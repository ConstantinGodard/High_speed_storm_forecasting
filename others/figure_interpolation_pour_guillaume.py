###################### IMPORTATIONS OF EXISTING PYTHON MODULES
# cd /Users/constantingodard/Documents/Stage_ENS/rapport_et_soutenance_Guillaume_Tran/Donnees_Guillaume_path
# python

import time as tm
import numpy as np
from datetime import datetime
import os
import sys

# Importation de la configuration des chemins
from config import ROOT_DIR  # ROOT_DIR = Donnees_Guillaume_path

# Définir le chemin vers pourJF-2
path_pourJF = os.path.join(ROOT_DIR, "pourJF")



"""
@Guillaume : 
J'ai créé un fichier config pour intialiser ROOT_DIR.
Sur mon ordinateur, ROOT_DIR = /Users/constantingodard/Documents/Stage_ENS/rapport_et_soutenance_Guillaume_Tran/Donnees_Guillaume_path

Où Donnees_Guillaume_path est juste une sauvegarde annexe de Donnees_Guillaume

Pour la sauvegarde des figures, j'ai stocker ça dans la variable : dossier_sauvegarde

Tu peux chercher les @Guillaume pour voir les variables à modifier pour reproduire figures que tu veux
"""

dossier_sauvegarde = ROOT_DIR + "/test_constantin/resultat_juillet"

# Ajout des dossiers nécessaires au PATH Python
sys.path.append(os.path.join(path_pourJF, "LIBdjinn3"))
import djinn as dj

from tensorflow.keras.models import load_model



sys.path.append(os.path.join(path_pourJF, "Reproduction_fig_Kluh2022", "fig6_Daa2D"))
from Daa2DLast_new_lib import (
    Daa2D_AlphaE_DATA,
    Daa2D_KpT_DATA,
    Daa2D_LT_DATA,
    Daa2D_AlphaE_light,
    Daa2D_KpT_light,
    Daa2D_LT_light,
    Daa2D_LT_light2,
    Daa2D_LT_light3,
    Daa2D_appli,
    Daa2D_appli_MAIN,
    Daa2D_appli_STD_MAIN,
    Daa2D_LT_appli_extrap_MAIN,
    Daa2D_3
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cmx
import matplotlib

## Chargement du modèle
#model_dnn = dj.load(model_name, model_path) 
model_path = path_pourJF +"/EntrainementDNN/LIBdjinn3/application/trained_on_stormsmeanKpTL/model9_10p_ep25000_ntrees20_lr0.002_bat5000/"
chargement_djinn = dj.load("dj9", model_path) 

#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
"""Coefficient de Diffusion dans le plan alpha, E """
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

"""
Utilisation : prendre la bonne séquence de kp_map qui t'intéresse parmis Kp_t_map_median, Kp_t_map_5th_percentile et Kp_t_map_95th_percentile
Changer manuellement le titre dans la fonction fig.suptitle 
Changer manuellement le titre de la sauvegarde .png avec la variable nom_tempete

voir les @Guillaume

"""

#--------------------------------------------------------------------------------------------------
### EXTRAPOLATION DE D_ALPHA EN UTILISANT LES INDICES KP EN CONTINU###
# On effectue une extrapolation en créant une fonction polynomiale passant 
# par les valeurs discrètes de Kp à un instant t donné. voir interpolationmap.ipynb
#--------------------------------------------------------------------------------------------------
def Daa2D_LT_light_interpolation(case_name, model_path_res, model_dnn) : 
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib 
    import matplotlib.ticker as ticker
    import matplotlib.cm as cmx

    from scipy.interpolate import interp1d

    # Dictionnaire Kp médian
    Kp_t_map_median = {
    10: 4, 11: 3, 12: 2, 13: 3, 14: 2, 15: 2, 16: 2, 17: 2, 18: 1
        }

    Kp_t_map_5th_percentile = {
        10: 3, 11: 2, 12: 1, 13: 1, 14: 1, 15: 0, 16: 0, 17: 0, 18: 0
    }

    Kp_t_map_95th_percentile = {
        10: 5, 11: 5, 12: 4, 13: 5, 14: 4, 15: 3, 16: 3, 17: 3, 18: 3
    }

    # @Guillaume : Changer kp_map en focntion de la tempête caractéristique à plotter
    kp_map = Kp_t_map_median

    # Données discrètes
    t_discrete_vals = np.array(list(kp_map.keys()))
    Kp_vals = np.array(list(kp_map.values()))


    # Création d'une interpolation spline cubique
    Kp_interp = interp1d(t_discrete_vals, Kp_vals, kind='cubic', fill_value='extrapolate')


    parameters = {'axes.labelsize':10 , 'axes.titlesize':8, 
        'figure.titlesize':8, 'xtick.labelsize':8, 'ytick.labelsize':8, 'axes.facecolor':'white'}
    plt.rcParams.update(parameters)

    # (L,t) grid
    L = np.arange(1.5,5.5,0.05)
    Time = np.arange(10,18,0.05)


    # Paramètres alpha et E fixés
    angle = np.array([30,49,69,86])
    E = np.array([np.log10(0.1),np.log10(0.5),np.log10(1.),np.log10(2)])

    # Initialisation de la figure
    fig, axs = plt.subplots(4,4,constrained_layout=True,figsize=(9,7))
    # @Guillaume : changer le titre de la figure en fonction de la HSS modélisé
    fig.suptitle(r"$Log_{10}(D_{\alpha\alpha}) (L,t)$ avec $K_p(t)$ $pour$ $HSS$ $5^{\mathrm{th}}$ $percentile$") # $5^{\mathrm{th}}$ percentile

    colorsMap = 'jet'
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=-9, vmax=-4)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig.colorbar(scalarMap,ax=axs.ravel().tolist(), label=r"$\log_{10}\left(D_{\alpha\alpha}\right)$ ($s^{-1}$)")

    index_fig = 0
    index_fig_y = 0

    for ie in E:
        for ia in angle:
            # Création de la grille (L,t) avec Kp dépendant de t
            indata = np.empty((L.shape[0] * Time.shape[0], 5))
            ind = 0

            for iL in L:
                for t_val in Time:
                    t_discrete = int(np.floor(t_val))
                    kp_val = kp_val = Kp_interp(t_val)

                    indata[ind, 0] = kp_val      # Kp
                    indata[ind, 1] = t_val       # t
                    indata[ind, 2] = iL          # L
                    indata[ind, 3] = ia          # angle
                    indata[ind, 4] = ie          # E
                    ind += 1

            pred = model_dnn.predict(indata)[:,0]
            indata[:,1] = (indata[:,1]-9.)/3.  # normalisation temps

            # Titre de la sous-figure
            ep = 10**ie
            titl = r"$E=$"+str("%2.1f"%ep)+" MeV, "+r"$\alpha=$"+str("%2.1f"%ia)
            
            if(index_fig%4==0 and index_fig != 0):
                index_fig = 0
                index_fig_y += 1

            axs[index_fig_y,index_fig].set_title(titl)
            axs[index_fig_y,index_fig].xaxis.set_major_locator(ticker.FixedLocator([1,2,3]))
            axs[index_fig_y,index_fig].xaxis.set_minor_locator(ticker.FixedLocator([0.33, 0.67, 1.33, 1.67, 2.33, 2.67]))
            axs[index_fig_y,index_fig].yaxis.set_major_locator(ticker.FixedLocator([2,3,4,5]))
            axs[index_fig_y,index_fig].yaxis.set_minor_locator(ticker.FixedLocator([1.5, 2.5, 3.5, 4.5, 5.5]))

            axs[index_fig_y,index_fig].scatter(indata[:,1],indata[:,2],c=scalarMap.to_rgba(pred),alpha=1, marker='.')

            if(index_fig_y==3):                                                     
                axs[index_fig_y,index_fig].set_xlabel("Days")  
            if(index_fig==0):
                axs[index_fig_y,index_fig].set_ylabel("L")      

            index_fig += 1

    # @Guillaume : Changer manuellement le nom de la figure sauvegarder avec la variable nom_tempete
    nom_tempete = "HSS_median"
    fig.savefig(model_path_res + "/HDaaLT_" + case_name + "_" + nom_tempete + ".png"  , dpi=300, format='png')
    print("Saved at : " + model_path_res + "/HDaaLT_" + case_name + "_" + nom_tempete + ".png")

    return ()

# Ligne à exécuter : 
Daa2D_LT_light_interpolation("model9_10p_ep25000_ntrees20_lr0.002_bat5000", dossier_sauvegarde, chargement_djinn)







#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
"""Coefficient de Diffusion dans le plan aplha, E """
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cmx
import matplotlib


"""
Utilisation : pour avoir un meilleure résolution des coefficients, je trouve que c'ets mieux de plot 3x avec chaque valeur de t.
Il faut donc modifier : Time = np.array([16, 17, 18])  avec des valeurs allant de 10 à 18
Il faut aussi changer kp_map en fonction de la tempête à représenter (Kp_t_map_median, Kp_t_map_5th_percentile, Kp_t_map_95th_percentile)
Changer manuellement le titre de la sauvegarde .png avec la variable nom_tempete

voir les @Guillaume
"""

def Daa2D_AlphaE_light_modifie2(case_name, model_path_res, model_dnn):

    # Configuration des paramètres de plot
    parameters = {'axes.labelsize': 10, 'axes.titlesize': 8,
                  'figure.titlesize': 8, 'xtick.labelsize': 8,
                  'ytick.labelsize': 8, 'axes.facecolor': 'white'}
    plt.rcParams.update(parameters)

    # Grilles alpha et E
    angle = np.arange(10, 89, 0.05)
    E = np.arange(np.log10(0.1), np.log10(6), 0.05)

    # Choix des valeurs
    L = np.array([2, 3, 4, 5])  # 4 lignes
    # @Guillaume : changer les valeurs de Time pour plot les bons t (allant de 10 à 18)
    Time = np.array([16, 17, 18])  # 3 colonnes

    # Dictionnaire Kp médian
    Kp_t_map_median = {
    10: 4, 11: 3, 12: 2, 13: 3, 14: 2, 15: 2, 16: 2, 17: 2, 18: 1
        }

    Kp_t_map_5th_percentile = {
        10: 3, 11: 2, 12: 1, 13: 1, 14: 1, 15: 0, 16: 0, 17: 0, 18: 0
    }

    Kp_t_map_95th_percentile = {
        10: 5, 11: 5, 12: 4, 13: 5, 14: 4, 15: 3, 16: 3, 17: 3, 18: 3
    }

    # @Guillaume : Changer le mapping en fonction de la tempête caractéristique à étudier
    kp_map = Kp_t_map_median

    # Figure
    fig, axs = plt.subplots(nrows=4, ncols=3, layout='constrained', figsize=(12, 8))
    fig.suptitle(r"$Log_{10}(D_{\alpha\alpha}) (\alpha,E)$")

    # Color map
    colorsMap = 'jet'
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=-9, vmax=-4)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig.colorbar(scalarMap, ax=axs.ravel().tolist(),
                 label=r"$\log_{10}\left(D_{\alpha\alpha}\right)$ ($s^{-1}$)")

    for i_row, iL in enumerate(L):
        for i_col, it in enumerate(Time):
            iKp = kp_map[it]
            indata = np.empty((angle.shape[0]*E.shape[0], 5))
            nE = E.shape[0]
            for i, ia in enumerate(angle):
                indata[i*nE:(i+1)*nE, 0] = iKp
                indata[i*nE:(i+1)*nE, 1] = it
                indata[i*nE:(i+1)*nE, 2] = iL
                indata[i*nE:(i+1)*nE, 3] = ia
                indata[i*nE:(i+1)*nE, 4] = E

            pred = model_dnn.predict(indata)[:, 0]
            indata[:, 1] = (indata[:, 1] - 9.) / 3.

            ax = axs[i_row, i_col]
            title = f"L={iL} $K_p$={iKp} t={((it - 9.) / 3.):.2f}"
            ax.set_title(title)
            ax.scatter(indata[:, 3], 10**indata[:, 4],
                       c=scalarMap.to_rgba(pred), alpha=1, marker='.')
            ax.set_yscale("log")
            ax.set_xlim(0, 90)
            ax.set_ylim(10**np.min(E), 10**np.max(E))

            if i_row == 3:
                ax.set_xlabel(r"$\alpha$ ($°$)")
            if i_col == 0:
                ax.set_ylabel(r"$E$ (MeV)")

    # @Guillaume : Changer manuellement le nom de la figure sauvegarder avec la variable nom_tempete
    nom_tempete = "HSS_median"
    fig.savefig(f"{model_path_res}/HDaaAlphaE_{case_name}_{nom_tempete}.png", dpi=300, format='png')
    print(f"Saved at : {model_path_res}/HDaaAlphaE_{case_name}_{nom_tempete}.png")
    return ()

# Ligne à exécuter : 
Daa2D_AlphaE_light_modifie2("model9_10p_ep25000_ntrees20_lr0.002_bat5000",dossier_sauvegarde, chargement_djinn)



### -----------------------------------------
### Affichage des observations
### -----------------------------------------

fichier_obs_path = ROOT_DIR + "/test_constantin/Entrainement_RNN/32STORMS_DATA/HSS28.dat"

#Closest to median: 28 1.8027756377319946
#Closest to 5th: 3 2.2588713996153036
#Closest to 95th: 30 1.2267844146385298

def Daa2D_AlphaE_light_obs(num_storm, part):

    fichier_obs_path = ROOT_DIR + f"/test_constantin/Entrainement_RNN/32STORMS_DATA/HSS{num_storm}.dat"

    # Chargement du fichier .dat
    # Colonnes : 0=Kp, 1=t, 2=L, 3=alpha, 4=E, 5=Daa (log10)
    data = np.loadtxt(fichier_obs_path)
    Kp_col, t_col, L_col = data[:, 0], data[:, 1], data[:, 2]
    alpha_col, E_col, Daa_col = data[:, 3], data[:, 4], data[:, 5]

    parameters = {'axes.labelsize': 10, 'axes.titlesize': 8,
                  'figure.titlesize': 8, 'xtick.labelsize': 8,
                  'ytick.labelsize': 8, 'axes.facecolor': 'white'}
    plt.rcParams.update(parameters)


    L_vals = np.array([2, 3, 4, 5])

    if part == 1:
        Time = np.array([10,11,12])
    elif part == 2:
        Time = np.array([13,14,15])
    elif part == 3:
        Time = np.array([16,17,18])


    fig, axs = plt.subplots(nrows=4, ncols=3, layout='constrained', figsize=(12, 8))
    fig.suptitle(r"$Log_{10}(D_{\alpha\alpha})$ Observations")

    colorsMap = 'jet'
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=-9, vmax=-4)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig.colorbar(scalarMap, ax=axs.ravel().tolist(),
                 label=r"$\log_{10}\left(D_{\alpha\alpha}\right)$ ($s^{-1}$)")
    
    for i_row, iL in enumerate(L_vals):
        for i_col, it in enumerate(Time):

            # Filtrage des lignes correspondant à L et t
            mask_lt = (L_col == iL) & (t_col == it)
            if not np.any(mask_lt):
                continue

            # Récupération des valeurs Kp présentes pour ce (L, t)
            kp_vals = np.unique(Kp_col[mask_lt])

            ax = axs[i_row, i_col]
            for iKp in kp_vals:
                mask = mask_lt & (Kp_col == iKp)
                x = alpha_col[mask]
                y = 10**E_col[mask]   # E en log10 dans le fichier
                c = Daa_col[mask]     # déjà en log10

                ax.scatter(x, y, c=scalarMap.to_rgba(c), alpha=1, marker='.')

            # Construction du titre avec L, t et la/les valeurs de Kp
            kp_str = ", ".join([str(int(k)) for k in kp_vals])
            title = f"L={iL}  t={((it - 9.) / 3.):.2f}  Kp={kp_str}"
            ax.set_title(title)

            ax.set_yscale('linear') # essayer log et linear
            ax.set_xlim(0, 90)
            ax.set_ylim(y.min(), y.max())

            if i_row == 3:
                ax.set_xlabel(r"$\alpha$ ($°$)")
            if i_col == 0:
                ax.set_ylabel(r"$E$ (MeV)")



    nom_tempete = "HSS_obs"
    fig.savefig(f"{ROOT_DIR}/test_constantin/resultat_juillet/InterpolationKp/observation_HSS{num_storm}_part{part}.png", dpi=300, format='png')
    print(f"Saved at : {ROOT_DIR}/test_constantin/resultat_juillet/InterpolationKp/observation_HSS{num_storm}_part{part}.png")
    return ()


Daa2D_AlphaE_light_obs(3,1)
Daa2D_AlphaE_light_obs(3,2)
Daa2D_AlphaE_light_obs(3,3)



import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable




### -----------------------------------------
### Affichage des différenciation ti+1 -ti
### -----------------------------------------

def Daa2D_diff_t_obs_combined(num_storm, part=1, ncols=3):
    """
    Affiche toutes les différences Δlog10(Daa) pour chaque L sur une figure,
    avec les colonnes correspondant aux triplets t → t+1 dans la plage définie par part.
    Tous les Kp sont affichés.
    """

    if part not in [1, 2, 3]:
        raise ValueError("part doit etre 1,2 ou 3")


    fichier_obs_path = ROOT_DIR + f"/test_constantin/Entrainement_RNN/32STORMS_DATA/HSS{num_storm}.dat"
    
    data = np.loadtxt(fichier_obs_path)
    df = pd.DataFrame(data, columns=['Kp','t','L','alpha','E','Daa'])
    df[['Kp','t','L']] = df[['Kp','t','L']].astype(int)


    all_times = sorted(df.t.unique())
    triplets_all = [(t, t+1) for t in all_times if t+1 in all_times]

    times = np.arange(10 + (part-1)*3, 10 + part*3)
    triplets_part = [(t, t1) for (t, t1) in triplets_all if t in times]


    L_vals = sorted(df.L.unique())
    nrows = len(L_vals)

    # Créer figure
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4*nrows), layout='constrained')
    if nrows==1: axs = axs[np.newaxis,:]

    cmap = plt.get_cmap('bwr')

    global_vmin = np.inf
    global_vmax = -np.inf

    for i_row, L in enumerate(L_vals):
        for j, (t, t_next) in enumerate(triplets_part):  # on fait t -> t+1
            if j >= ncols:
                break

            t_next = t + 1
            ax = axs[i_row, j]
            df_t = df[(df.L==L)&(df.t==t)]
            df_t1 = df[(df.L==L)&(df.t==t_next)]
            
            # Merge uniquement sur L, alpha, E
            merged = pd.merge(df_t, df_t1, on=['L','alpha','E'], suffixes=('_t','_t1'))
            if merged.empty:
                ax.axis('off')
                continue

            merged['diff'] = merged['Daa_t1'] - merged['Daa_t']

            x = merged['alpha'].values
            y = 10**merged['E'].values
            dv = merged['diff'].values
            kp_str = ", ".join(map(str, np.unique(merged.Kp_t)))

            vmin = min(dv.min(), 0)
            vmax = max(dv.max(), 0)
            #print(f"vmin = {vmin}, vmax =. {vmax}")

            global_vmin = min(global_vmin, vmin)
            global_vmax = max(global_vmax, vmax)


            if vmin >= 0 or vmax <= 0 or vmin == vmax:
                # Cas dégénéré : toutes les valeurs identiques
                norm = None
            else:
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                
            sc = ax.scatter(x, y, c=dv, cmap='seismic', norm=norm, marker='.')

            ax.set_yscale("linear") # essayer log et linear
            ax.set_xlim(0,90)
            ax.set_ylim(y.min(), y.max())
            ax.set_title(f"L={L}  d{t-9}: {t}->{t_next}\nKp={kp_str}", fontsize=8)
            if i_row==nrows-1: ax.set_xlabel(r"$\alpha$ ($°$)")
            if j==0: ax.set_ylabel(r"$E$ (MeV)")

        # remplir les colonnes vides
        for j in range(len(times)-1, ncols):
            axs[i_row,j].axis('off')


    global_vmin = np.floor(global_vmin)
    global_vmax = np.ceil(global_vmax)

    cb = fig.colorbar(
        ScalarMappable(norm=TwoSlopeNorm(vmin=global_vmin, vcenter=0, vmax=global_vmax), cmap=cmap),
        ax=axs.ravel().tolist(),
        label='Δ valeur',
        extend='neither'
    )
    ticks = np.linspace(global_vmin,global_vmax, 5)
    cb.set_ticks(ticks)
    cb.ax.set_yticklabels([f"{t:.2f}" for t in ticks])


    out_path = f"{ROOT_DIR}/test_constantin/resultat_juillet/InterpolationKp/diff_HSS{num_storm}_part{part}_combined.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")


Daa2D_diff_t_obs_combined(30,1)
Daa2D_diff_t_obs_combined(30,2)
Daa2D_diff_t_obs_combined(30,3)