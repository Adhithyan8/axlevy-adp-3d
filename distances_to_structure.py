import argparse
import logging
import os
import pickle
import time

import numpy as np
import torch
from Bio import PDB
from chroma import Chroma, Protein
from chroma.layers.structure.rmsd import CrossRMSD
from src.plots import plot_metric

# atomic radii for various atom types
ATOM_RADII = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "F": 1.47,
    "P": 1.80,
    "CL": 1.75,
    "MG": 1.73,
}


def count_clashes(structure, clash_cutoff=0.63, sequential_exclude=1):
    clash_cutoffs = {
        i + "_" + j: (clash_cutoff * (ATOM_RADII[i] + ATOM_RADII[j]))
        for i in ATOM_RADII
        for j in ATOM_RADII
    }

    atoms = [x for x in structure.get_atoms() if x.element in ATOM_RADII]
    coords = np.array([a.coord for a in atoms], dtype="d")

    kdt = PDB.kdtrees.KDTree(coords)
    clashes = []

    for atom_1 in atoms:
        kdt_search = kdt.search(
            np.array(atom_1.coord, dtype="d"), max(clash_cutoffs.values())
        )
        potential_clash = [(a.index, a.radius) for a in kdt_search]

        for ix, atom_distance in potential_clash:
            atom_2 = atoms[ix]
            res1 = atom_1.parent
            res2 = atom_2.parent

            # exclude same or nearby residues
            if res1.parent == res2.parent:
                res_diff = abs(res1.id[1] - res2.id[1])
                if res_diff <= sequential_exclude:
                    continue

            # exclude peptide bonds
            if (atom_2.name == "C" and atom_1.name == "N") or (
                atom_2.name == "N" and atom_1.name == "C"
            ):
                continue

            # exclude disulfide bridges
            if (atom_2.name == "SG" and atom_1.name == "SG") and atom_distance > 1.88:
                continue

            # check if it's a clash
            clash_key = atom_2.element + "_" + atom_1.element
            if atom_distance < clash_cutoffs.get(clash_key, float("inf")):
                clashes.append((atom_1, atom_2))

    return len(clashes) // 2


def parse_args():
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--cif", type=str, required=True)

    # model parameters
    parser.add_argument("--weights-backbone", type=str, default=None)
    parser.add_argument("--weights-design", type=str, default=None)

    # distance constraint parameters
    parser.add_argument("--n-distances", type=int, default=-1)
    parser.add_argument("--distance-threshold", type=float, default=6.0)
    parser.add_argument("--noise-std", type=float, default=0.5)

    # optimization parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--lr-distance", type=float, default=0.01)
    parser.add_argument("--rho-distance", type=float, default=0.99)
    parser.add_argument(
        "--temporal-schedule",
        type=str,
        default="sqrt",
        choices=["linear", "sqrt", "constant"],
    )
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--use-diffusion", type=int, default=1, choices=[0, 1])

    # initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-gt", type=int, default=0, choices=[0, 1])

    # logging parameters
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


def setup_ground_truth(protein, device="cuda"):
    X_gt, C_gt, S_gt = protein.to_XCS(all_atom=False)
    X_gt = X_gt[torch.abs(C_gt) == 1][None]
    S_gt = S_gt[torch.abs(C_gt) == 1][None]
    C_gt = C_gt[torch.abs(C_gt) == 1][None]
    X_gt -= X_gt.mean(dim=(0, 1, 2))
    return X_gt, C_gt, S_gt


def X_to_distance_matrix(X):
    nodes = X[:, :, 1, :]  # [N, R, 3] - CA atoms
    distance_matrix = torch.linalg.norm(nodes[:, :, None] - nodes[:, None], dim=-1)
    return distance_matrix  # [N, R, R]


def create_distance_mask(distance_matrix_gt, C_gt, n_distances, distance_threshold):
    n_residues = distance_matrix_gt.shape[1]

    if n_distances == -1:
        mask_dist = torch.ones_like(distance_matrix_gt)
    else:
        idxs = torch.arange(n_residues, device=distance_matrix_gt.device)
        idxs = idxs[(C_gt == 1).reshape(-1)]
        mask_dist = torch.zeros_like(distance_matrix_gt)
        n_ones = 0

        while n_ones < n_distances:
            i = idxs[np.random.randint(len(idxs))]
            j = idxs[np.random.randint(len(idxs))]
            if i < j:
                if mask_dist[0, i, j] < 0.5:
                    if distance_matrix_gt[0, i, j] < distance_threshold:
                        mask_dist[0, i, j] = 1.0
                        n_ones += 1

    return mask_dist


def add_noise_to_distances(Y_dist, mask_dist, noise_std):
    batch_size, n_res, _ = Y_dist.shape

    noise_upper = (
        torch.randn(batch_size, n_res, n_res, device=Y_dist.device) * noise_std
    )
    noise_upper = torch.triu(noise_upper, diagonal=1)
    noise_symm = noise_upper + noise_upper.transpose(-2, -1)

    Y_dist_noisy = Y_dist + mask_dist * noise_symm
    return Y_dist_noisy


def temporal_schedule(epoch, epochs, schedule_type, t_init):
    if schedule_type == "linear":
        return (-t_init + 0.001) * epoch / epochs + t_init
    elif schedule_type == "sqrt":
        return (1.0 - 0.001) * (1.0 - np.sqrt(epoch / epochs)) + 0.001
    elif schedule_type == "constant":
        return t_init
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def compute_distance_gradient(Z, C, Y_dist, mask_dist, multiply_R, lr_distance):
    if lr_distance > 0.0:
        with torch.enable_grad():
            Z.requires_grad_(True)
            X = multiply_R(Z, C)
            distance_matrix = X_to_distance_matrix(X)
            loss_dist = (
                ((Y_dist - mask_dist * distance_matrix) ** 2).mean(dim=(1, 2)).sum()
            )
            loss_dist.backward()

            grad_Z = Z.grad
        Z.requires_grad_(False)
    else:
        grad_Z = torch.zeros_like(Z)
        loss_dist = torch.tensor([0.0], device=Z.device)

    return grad_Z, loss_dist


def compute_rmsd_metrics(X_samples, X_gt, C_gt):
    population_size = X_samples.shape[0]
    rmsds = []
    rmsds_ca = []

    for i in range(population_size):
        # all-atom RMSD
        rmsd, _ = CrossRMSD().pairedRMSD(
            torch.clone(X_samples[i, C_gt[0] == 1]).cpu().reshape(1, -1, 3),
            torch.clone(X_gt[0, C_gt[0] == 1]).cpu().reshape(1, -1, 3),
            compute_alignment=True,
        )
        # CA-only RMSD
        rmsd_ca, _ = CrossRMSD().pairedRMSD(
            torch.clone(X_samples[i, C_gt[0] == 1, 1, :]).cpu().reshape(1, -1, 3),
            torch.clone(X_gt[0, C_gt[0] == 1, 1, :]).cpu().reshape(1, -1, 3),
            compute_alignment=True,
        )
        rmsds.append(rmsd.item())
        rmsds_ca.append(rmsd_ca.item())

    return np.array(rmsds), np.array(rmsds_ca)


def compute_elbo_metrics(X_samples, C_gt, backbone_network):
    population_size = X_samples.shape[0]
    elbos = []

    for i in range(population_size):
        elbo = backbone_network.estimate_elbo(X=X_samples[i : i + 1], C=C_gt)
        elbos.append(elbo.item())

    return np.array(elbos)


def compute_clash_metrics(X_samples, X_gt, C_gt, S_gt, outdir):
    population_size = X_samples.shape[0]
    parser = PDB.PDBParser(QUIET=True)
    all_clashes = []

    for i in range(population_size):
        # align structure
        _, X_aligned_temp = CrossRMSD().pairedRMSD(
            torch.clone(X_samples[i]).cpu().reshape(1, -1, 3),
            torch.clone(X_gt[0]).cpu().reshape(1, -1, 3),
            compute_alignment=True,
        )
        X_aligned_temp = X_aligned_temp.reshape(1, -1, 4, 3)

        # save temporary PDB
        protein_temp = Protein.from_XCS(X_aligned_temp, C_gt, S_gt)
        temp_pdb_path = f"{outdir}/temp_{i}.pdb"
        protein_temp.to_PDB(temp_pdb_path)

        # count clashes
        structure_temp = parser.get_structure("protein", temp_pdb_path)
        clashes = count_clashes(structure_temp, clash_cutoff=0.63, sequential_exclude=1)
        all_clashes.append(clashes)

    return np.array(all_clashes)


def optimize_structure(
    X_init,
    C_gt,
    S_gt,
    X_gt,
    Y_dist,
    mask_dist,
    backbone_network,
    multiply_R,
    multiply_R_inverse,
    args,
):
    X = X_init
    Z = multiply_R_inverse(X, C_gt)
    V_dist = torch.zeros_like(Z)

    metrics = {"epoch": [], "rmsd": [], "rmsd_ca": [], "t": [], "loss_dist": []}

    logging.info("Starting optimization...")

    for epoch in range(args.epochs):
        t = torch.tensor(
            temporal_schedule(epoch, args.epochs, args.temporal_schedule, args.t),
            device=X.device,
        ).float()

        # denoise if using diffusion
        if args.use_diffusion:
            with torch.no_grad():
                X0 = backbone_network.denoise(X.detach(), C_gt, t)
        else:
            X0 = X

        Z0 = multiply_R_inverse(X0, C_gt)

        # Compute gradient and update
        grad_Z_dist, loss_dist = compute_distance_gradient(
            Z0, C_gt, Y_dist, mask_dist, multiply_R, args.lr_distance
        )
        V_dist = args.rho_distance * V_dist + args.lr_distance * grad_Z_dist
        Z0 = Z0 - V_dist

        # Apply noise if using diffusion
        if args.use_diffusion:
            tm1 = torch.tensor(
                temporal_schedule(
                    epoch + 1, args.epochs, args.temporal_schedule, args.t
                ),
                device=X.device,
            ).float()
            alpha, sigma, _, _, _, _ = (
                backbone_network.noise_perturb._schedule_coefficients(tm1)
            )
            X = multiply_R(alpha * Z0 + sigma * torch.randn_like(Z0), C_gt)
        else:
            X = multiply_R(Z0, C_gt)

        # log metrics periodically
        if (epoch + 1) % args.log_every == 0:
            rmsds, rmsds_ca = compute_rmsd_metrics(X, X_gt, C_gt)
            idx_best = np.argmin(rmsds)

            metrics["epoch"].append(epoch)
            metrics["t"].append(t.item())
            metrics["loss_dist"].append(loss_dist.item())
            metrics["rmsd"].append(rmsds.tolist())
            metrics["rmsd_ca"].append(rmsds_ca.tolist())

            logging.info(
                f"Epoch {epoch + 1}/{args.epochs} - "
                f"Best RMSD: {rmsds[idx_best]:.4f} Å, "
                f"Mean RMSD: {rmsds.mean():.4f} Å"
            )

    return X, metrics


def main():
    args = parse_args()

    # configure logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    start_time = time.time()

    # set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # save arguments
    logging.info("Saving configuration...")
    args_dict = vars(args)
    with open(f"{args.outdir}/config.txt", "w") as f:
        for key in sorted(args_dict.keys()):
            f.write(f"{key}: {args_dict[key]}\n")

    # load Chroma model
    logging.info("Loading Chroma model...")
    if args.weights_backbone is not None and args.weights_design is not None:
        chroma = Chroma(
            weights_backbone=args.weights_backbone, weights_design=args.weights_design
        )
    else:
        chroma = Chroma()

    backbone_network = chroma.backbone_network

    def multiply_R(Z, C):
        return backbone_network.noise_perturb.base_gaussian._multiply_R(Z, C)

    def multiply_R_inverse(X, C):
        return backbone_network.noise_perturb.base_gaussian._multiply_R_inverse(X, C)

    # load ground truth
    logging.info(f"Loading ground truth structure from {args.cif}...")
    protein = Protein.from_CIF(args.cif, device="cuda")
    X_gt, C_gt, S_gt = setup_ground_truth(protein)
    n_residues = X_gt.shape[1]
    logging.info(f"Number of residues: {n_residues}")

    # compute distance matrix and constraints
    distance_matrix_gt = X_to_distance_matrix(X_gt)
    n_close = (
        (distance_matrix_gt < args.distance_threshold).sum().item() - n_residues
    ) // 2
    logging.info(f"Number of distances < {args.distance_threshold} Å: {n_close}")

    # create distance mask
    mask_dist = create_distance_mask(
        distance_matrix_gt, C_gt, args.n_distances, args.distance_threshold
    )
    Y_dist = mask_dist * distance_matrix_gt

    # add noise to observations
    Y_dist = add_noise_to_distances(Y_dist, mask_dist, args.noise_std)

    # initialize population
    logging.info(f"Initializing population of size {args.population_size}...")
    C_gt_expanded = C_gt.expand(args.population_size, -1)
    S_gt_expanded = S_gt.expand(args.population_size, -1)

    if args.init_gt:
        X_init = torch.clone(X_gt).expand(args.population_size, -1, -1, -1)
        logging.info("Initialized from ground truth structure")
    else:
        Z_init = torch.randn(args.population_size, *X_gt.shape[1:], device=X_gt.device)
        X_init = multiply_R(Z_init, C_gt_expanded)
        logging.info("Initialized from random noise")

    # run optimization
    X_final, metrics = optimize_structure(
        X_init,
        C_gt_expanded,
        S_gt_expanded,
        X_gt,
        Y_dist,
        mask_dist,
        backbone_network,
        multiply_R,
        multiply_R_inverse,
        args,
    )

    # compute final metrics
    logging.info("=" * 70)
    logging.info("Computing Final Metrics")
    logging.info("=" * 70)

    # RMSD
    rmsds_final, rmsds_ca_final = compute_rmsd_metrics(X_final, X_gt, C_gt_expanded)
    idx_best = np.argmin(rmsds_final)

    logging.info(f"RMSD Statistics (Å):")
    logging.info(f"Best RMSD: {rmsds_final.min():.4f}")
    logging.info(f"Mean RMSD: {rmsds_final.mean():.4f}")
    logging.info(f"Std RMSD: {rmsds_final.std():.4f}")

    # ELBO
    C_gt_single = C_gt_expanded[0:1]
    S_gt_single = S_gt_expanded[0:1]
    elbos_final = compute_elbo_metrics(X_final, C_gt_single, backbone_network)

    logging.info(f"ELBO Statistics:")
    logging.info(f"Best sample ELBO: {elbos_final[idx_best]:.4e}")
    logging.info(f"Mean ELBO: {elbos_final.mean():.4e}")
    logging.info(f"Std ELBO: {elbos_final.std():.4e}")
    logging.info(f"Max ELBO: {elbos_final.max():.4e}")

    # distance constraint error (L2)
    with torch.no_grad():
        distance_matrices_final = X_to_distance_matrix(X_final)
        l2_errors = (
            ((Y_dist - mask_dist * distance_matrices_final) ** 2).sum(dim=(1, 2)).sqrt()
        )

    logging.info(f"Distance Constraint L2 Error:")
    logging.info(f"Mean L2: {l2_errors.mean().item():.4e}")
    logging.info(f"Std L2: {l2_errors.std().item():.4e}")

    # clashes
    clashes_final = compute_clash_metrics(
        X_final, X_gt, C_gt_single, S_gt_single, args.outdir
    )

    logging.info(f"Clash Statistics:")
    logging.info(f"Mean clashes: {clashes_final.mean():.2f}")
    logging.info(f"Std clashes: {clashes_final.std():.2f}")
    logging.info(f"Min clashes: {clashes_final.min()}")
    logging.info(f"Max clashes: {clashes_final.max()}")
    logging.info(f"Best sample clashes: {clashes_final[idx_best]}")

    # align and save best structure
    _, X_aligned = CrossRMSD().pairedRMSD(
        torch.clone(X_final[idx_best]).cpu().reshape(1, -1, 3),
        torch.clone(X_gt[0]).cpu().reshape(1, -1, 3),
        compute_alignment=True,
    )
    X_aligned = X_aligned.reshape(1, -1, 4, 3).cuda()

    # save results
    logging.info("Saving results...")
    with open(f"{args.outdir}/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    logging.info(f"Saved metrics to {args.outdir}/metrics.pkl")

    output_name = args.outdir.split("/")[-1]
    protein_out = Protein.from_XCS(X_aligned, C_gt_single, S_gt_single)
    protein_out.to_PDB(f"{args.outdir}/{output_name}.pdb")
    logging.info(f"Saved best structure to {args.outdir}/{output_name}.pdb")

    protein_gt = Protein.from_XCS(X_gt, C_gt_single, S_gt_single)
    protein_gt.to_PDB(f"{args.outdir}/{output_name}_gt.pdb")
    logging.info(f"Saved ground truth to {args.outdir}/{output_name}_gt.pdb")

    # save plots
    for key in metrics.keys():
        if key != "epoch":
            plot_path = f"{args.outdir}/{key}.png"
            plot_metric(metrics, key, plot_path)
            logging.info(f"Saved plot to {plot_path}")

    elapsed_time = (time.time() - start_time) / 60
    logging.info(f"Total execution time: {elapsed_time:.2f} minutes")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
