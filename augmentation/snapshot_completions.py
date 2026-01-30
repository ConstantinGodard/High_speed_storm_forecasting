import time as tm
import numpy as np
from datetime import datetime
import os
import sys
import pandas as pd


# Chemin vers les HSS
# À changer en focntion 
ROOT_DIR = "/gpfs/users/godardc/hinge/Donnees_Guillaume_path/Donnees_Guillaume_path"


#----------------------------------------------------------------------------------------------------------------
#Chargement de la HSS au format Dataframe
#----------------------------------------------------------------------------------------------------------------
def init_storm(num_storm):
    """
    Initialise les données d'un HSS donné :
       - charge le fichier .dat
       - construit df
       - calcule les valeurs et dimensions (t, alpha, E)
       - construit la grille (alpha, 10^E)

    Paramètre
    ---------
    num_storm : int
        Numéro du HSS (0–31)

    Retourne
    --------
    dict contenant :
        df, all_t, alpha_vals, E_vals, n_t, n_alpha, n_E, grid_points
    """

    fichier_obs_path = ROOT_DIR + f"/test_constantin/Entrainement_RNN/32STORMS_DATA/HSS{num_storm}.dat"

    # Charger
    data = np.loadtxt(fichier_obs_path)
    df = pd.DataFrame(data, columns=['Kp','t','L','alpha','E','Daa'])
    df[['Kp','t','L']] = df[['Kp','t','L']].astype(int)

    # Valeurs distinctes
    all_t = sorted(df.t.unique())
    alpha_vals = sorted(df.alpha.unique())
    E_vals = sorted(df.E.unique())

    # Dimensions
    n_t = len(all_t)
    n_alpha = len(alpha_vals)
    n_E = len(E_vals)

    # Grille (alpha, 10^E)
    alpha_grid, E_grid = np.meshgrid(alpha_vals, E_vals, indexing='ij')
    grid_points = np.column_stack((alpha_grid.ravel(), 10**E_grid.ravel()))

    return {
        "df": df,
        "all_t": all_t,
        "alpha_vals": alpha_vals,
        "E_vals": E_vals,
        "n_t": n_t,
        "n_alpha": n_alpha,
        "n_E": n_E,
        "grid_points": grid_points
    }

def init_storm2():
    """
    Initialise les données d'un HSS donné :
       - charge le fichier .dat
       - construit df
       - calcule les valeurs et dimensions (t, alpha, E)
       - construit la grille (alpha, 10^E)

    Paramètre
    ---------
    num_storm : int
        Numéro du HSS (0–31)

    Retourne
    --------
    dict contenant :
        df, all_t, alpha_vals, E_vals, n_t, n_alpha, n_E, grid_points
    """

    fichier_obs_path = "/gpfs/users/godardc/HSS31_debug.dat"

    # Charger
    data = np.loadtxt(fichier_obs_path)
    df = pd.DataFrame(data, columns=['Kp','t','L','alpha','E','Daa'])
    df[['Kp','t','L']] = df[['Kp','t','L']].astype(int)

    # Valeurs distinctes
    all_t = sorted(df.t.unique())
    alpha_vals = sorted(df.alpha.unique())
    E_vals = sorted(df.E.unique())

    # Dimensions
    n_t = len(all_t)
    n_alpha = len(alpha_vals)
    n_E = len(E_vals)

    # Grille (alpha, 10^E)
    alpha_grid, E_grid = np.meshgrid(alpha_vals, E_vals, indexing='ij')
    grid_points = np.column_stack((alpha_grid.ravel(), 10**E_grid.ravel()))

    return {
        "df": df,
        "all_t": all_t,
        "alpha_vals": alpha_vals,
        "E_vals": E_vals,
        "n_t": n_t,
        "n_alpha": n_alpha,
        "n_E": n_E,
        "grid_points": grid_points
    }

#----------------------------------------------------------------------------------------------------------------
# Création d'une fonction remplisssage pour lisser les snapshots ayant des points manquants
# Permet également de compléter un snapshot vide par un snapshot dont toutes les valeurs sont égales à -15.0
#----------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from itertools import product

# fonction utilitaire nearest neighbor
def nearest_value(a, e, df):
    if df.empty or df['Daa'].dropna().empty:
        return np.nan  # ou une autre valeur par défaut
    d = np.sqrt((df['alpha'] - a)**2 + (df['E'] - e)**2)
    return df.loc[d.idxmin(), 'Daa']


def remplissage(sous_df, df_ref):

    sous_df = sous_df.copy()

    # Vérification rapide : aucune valeur disponible
    if sous_df['Daa'].dropna().empty:
        # Matrice de taille attendue : alpha × E
        def_range = df_ref[(df_ref["L"] == 3) & (df_ref["t"] == 10)] # Pour toutes les storms prendre (L,t) = (2,10) SAUF pour 31 : prendre L = 3 et t =10
        alphas = np.sort(def_range['alpha'].unique())
        Es = np.sort(def_range['E'].unique())

        n_total = len(alphas) * len(Es)
        return np.full(n_total, -15.0)

    # Sinon : on continue le code normal

    def_range = df_ref[(df_ref["L"] == 3) & (df_ref["t"] == 10)]
    alphas = np.sort(def_range['alpha'].unique())
    Es = np.sort(def_range['E'].unique())

    full = pd.DataFrame(list(product(alphas, Es)), columns=['alpha', 'E'])
    merged = full.merge(sous_df[['alpha','E','Daa']], on=['alpha','E'], how='left')

    missing_idx = merged[merged['Daa'].isna()].index

    for idx in missing_idx:
        a = merged.at[idx, 'alpha']
        e = merged.at[idx, 'E']

        grp_E = sous_df[sous_df['alpha'] == a].sort_values('E')
        if len(grp_E) >= 2 and grp_E['E'].min() <= e <= grp_E['E'].max():
            merged.at[idx, 'Daa'] = np.interp(e, grp_E['E'].values, grp_E['Daa'].values)
            continue

        grp_a = sous_df[sous_df['E'] == e].sort_values('alpha')
        if len(grp_a) >= 2 and grp_a['alpha'].min() <= a <= grp_a['alpha'].max():
            merged.at[idx, 'Daa'] = np.interp(a, grp_a['alpha'].values, grp_a['Daa'].values)
            continue

        val = nearest_value(a, e, sous_df)
        merged.at[idx, 'Daa'] = -15.0 if pd.isna(val) else val

    pivot = merged.pivot(index='alpha', columns='E', values='Daa')
    matrice = pivot.values
    return matrice.ravel()




#----------------------------------------------------------------------------------------------------------------
# Application de la complétion de données à l'ensemble des snapshots d'une HSS (4 lignes, 9 colonnes => 36 snapshot par HSS)
#----------------------------------------------------------------------------------------------------------------


def remplissage_full(df, L_min=2, L_max=5, t_min=10, t_max=18):
    """
    Construit une matrice de snapshots remplis pour tous les couples (L, t).

    Paramètres
    ----------
    df : DataFrame
        Données contenant les colonnes ['L','t','alpha','E','Daa'].
    L_min, L_max : int
        Bornes de L (inclusif).
    t_min, t_max : int
        Bornes de t (inclusif).

    Retour
    ------
    snapshots : np.ndarray
        Tableau 3D de taille (n_points, n_t, n_L)
    """

    snapshots = []

    L_values = list(range(L_min, L_max + 1))
    t_values = list(range(t_min, t_max + 1))

    for L in L_values:
        snapshots_L = []
        for t in t_values:
            snapshot_df = df[(df["L"] == L) & (df["t"] == t)]
            filled = remplissage(snapshot_df,df)  # ta fonction existante
            print(f"      Snapshot L ={L}, t ={t} : shape ={filled.shape}")
            snapshots_L.append(filled)
        snapshots.append(snapshots_L)
        print(f"L-shell n°{L} completed")

    # Convertir en numpy array
    snapshots = np.array(snapshots)   # shape : (n_L, n_t, n_points)

    # On remet dans l'ordre (n_points, n_t, n_L) comme avant
    snapshots = snapshots.transpose(2, 1, 0)

    return snapshots


#----------------------------------------------------------------------------------------------------------------
# Application de la complétion de données à l'ensemble des HSS, sauvegardé dans "/gpfs/users/godardc/HSS_completed"
#----------------------------------------------------------------------------------------------------------------

def completion_all_storm():

    save_dir = "/gpfs/users/godardc/HSS_completed"

    for num_storm in range(7,32):
        if num_storm in [28, 30]:
            continue
        print(f"------------------------------------------\nDébut de la completion de la HSSn°{num_storm}")
        snapshots_matrix = remplissage_full(init_storm(num_storm)['df'])

        save_path = os.path.join(save_dir, f"snapshots_matrix_HSS{num_storm}.npy")
        np.save(save_path, snapshots_matrix)

        print(f"\nStorm {num_storm} sauvegardé.")
        print(f" → Fichier enregistré dans : {save_path}")
        print(f" → Format de la HSS : {snapshots_matrix.shape}")

def completion_one_storm(num_storm):

    save_dir = "/gpfs/users/godardc/HSS_completed"

    print(f"------------------------------------------\nDébut de la completion de la HSSn°{num_storm}")
    snapshots_matrix = remplissage_full(init_storm(num_storm)['df'])

    save_path = os.path.join(save_dir, f"snapshots_matrix_HSS{num_storm}.npy")
    np.save(save_path, snapshots_matrix)

    print(f"\nStorm {num_storm} sauvegardé.")
    print(f" → Fichier enregistré dans : {save_path}")
    print(f" → Format de la HSS : {snapshots_matrix.shape}")

def completion_one_storm_debug(num_storm):

    save_dir = "/gpfs/users/godardc/HSS_completed"

    print(f"------------------------------------------\nDébut de la completion de la HSSn°{num_storm}")
    snapshots_matrix = remplissage_full(init_storm2()['df'])

    save_path = os.path.join(save_dir, f"snapshots_matrix_HSS{num_storm}.npy")
    np.save(save_path, snapshots_matrix)

    print(f"\nStorm {num_storm} sauvegardé.")
    print(f" → Fichier enregistré dans : {save_path}")
    print(f" → Format de la HSS : {snapshots_matrix.shape}")


if __name__ == "__main__":
    print("=== Début de la complétion des snapshots ===")
    completion_all_storm()

    # Debug pour une storm 
    #completion_one_storm(25)
    #completion_one_storm_debug(25)
    print("=== Terminé ===")
