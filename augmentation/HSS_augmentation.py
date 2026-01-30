import time as tm
import numpy as np
from datetime import datetime
import os
import sys

# Importation de la configuration des chemins
ROOT_DIR = "/gpfs/users/godardc/hinge/Donnees_Guillaume_path/Donnees_Guillaume_path"
import pandas as pd



import numpy as np
import matplotlib.pyplot as plt

import importlib
import POD
import RBF


from POD import POD 
pod = POD()

from RBF import RBFInterpolator



def run_POD_RBF_for_L(snapshots_matrix_L, t_min=10, t_max=18, n_interp=100):
    """
    Applique POD + RBF + reconstruction pour un L donné.

    Paramètres
    ----------
    snapshots_matrix_L : np.ndarray
        Matrice 2D ou 3D contenant les snapshots (n_points, n_t) ou (n_points, n_t, 1)
    t_min, t_max : int
        Intervalle réel de temps correspondant aux snapshots
    n_interp : int
        Nombre de valeurs interpolées pour l'évaluation RBF
    plot : bool
        Si True, affiche quelques snapshots reconstruits

    Retour
    ------
    states_reconstructed : np.ndarray
        États reconstruits interpolés : shape (n_interp, n_points)
    """

    # --- Mise en forme ---
    if snapshots_matrix_L.ndim == 3:
        snapshots_matrix_L = snapshots_matrix_L[:, :, 0]  # retirer axe inutile

    n_points, n_t = snapshots_matrix_L.shape


    # --- Détection des snapshots artificiels ---
    mask_bad = np.all(snapshots_matrix_L == -15, axis=0)
    weights = np.ones(n_t)
    weights[mask_bad] = 0.6   # fort downweight des faux snapshots


    # Paramètres t utilisés pour l'entraînement
    t_train = np.linspace(t_min, t_max, n_t).reshape(-1, 1)

    # --- POD ---
    pod = POD()
    pod.compute(snapshots_matrix_L, energy_ratio=0.999, weights=weights)

    coeffs = pod.project(snapshots_matrix_L)

    # --- RBF interpolation ---
    rbf = RBFInterpolator()
    rbf.fit(t_train, coeffs, sample_weight=weights)

    # Paramètres intermédiaires
    t_interp = np.linspace(t_min, t_max, n_interp).reshape(-1, 1)

    # Prédiction des coefficients POD interpolés
    coeffs_interp = rbf.predict(t_interp)

    # Reconstruction des snapshots
    states = pod.reconstruct(coeffs_interp)

    # Mise en forme (n_interp, n_points)
    states = states
    
    return states


def run_all_L_shell(num_storm) :

    # Chargement du fichier
    snapshots_matrix = np.load(f"/gpfs/users/godardc/HSS_completed/snapshots_matrix_HSS{num_storm}.npy")

    # Affichage des dimensions
    print("Dimensions :", snapshots_matrix.shape)


        # --- Liste des L à traiter ---
    L_vals = [2, 3, 4, 5]

    states_all ={}

    for idx_L, L in enumerate(L_vals):

        # Extraire la bonne "couche" de snapshots via l'index vertical L
        snap_L = snapshots_matrix[:, :, idx_L]  # shape (n_points, n_t)

        # Reconstruire les états interpolés POD+RBF
        states = run_POD_RBF_for_L(
            snap_L,
            t_min=10,
            t_max=18,
            n_interp=100
        )  # shape (100, n_points)

        # On stocke
        states_all[str(L)] = states

    # Chemin du fichier d'enregistrement 
    
    saved_folder = "/gpfs/users/godardc/HSS_augmented"

    np.savez_compressed(saved_folder + f"/HSS{num_storm}_augmented.npz", **states_all)
    print(f"Storm n°{num_storm} saved at {saved_folder}/HSS{num_storm}_augmented.npz")


if __name__ == "__main__":
    print("=============== Début POD-RBF ===============")
    # for storm_i in range(1,32):
    #     print(f"=== Début HHS n°{storm_i} ===")
    #     run_all_L_shell(storm_i)
    #     print(f"=== Fin HHS n°{storm_i} ===")
    #     print("----------------------------------------------------")

    # test pour une tempête
    # print(f"=== Début HHS n°{25} ===")
    # run_all_L_shell(25)
    # print(f"=== Fin HHS n°{25} ===")
    print("================== FIN POD-RBF ==================")

