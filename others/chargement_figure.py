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


""" main_extrap(
    "dj9",
    "model9_10p_ep25000_ntrees20_lr0.002_bat5000",
    "LIBdjinn3",
    None,
    None,
    False,
    False,
    "mean",
    89,
    10
) """

# Pour info, signature main_extrap complète :
# def main_extrap(model_name, case_name, origin, num_STORM, storm_trained,
#                 bool_train_by_L, bool_train_by_storm, str_type, angle_max, time_step):

#model_dnn = dj.load(model_name, model_path) 
model_path = path_pourJF +"/EntrainementDNN/LIBdjinn3/application/trained_on_stormsmeanKpTL/model9_10p_ep25000_ntrees20_lr0.002_bat5000/"
chargement_djinn = dj.load("dj9", model_path) 

def count_parameters_per_tree(model):
    """
    Compte le nombre de paramètres pour chaque arbre dans le modèle DJINN.

    Args:
        model (DJINN_Regressor): Modèle DJINN chargé.

    Returns:
        dict: Dictionnaire contenant le nombre de paramètres pour chaque arbre.
    """
    tree_params = {}
    for p in range(model._DJINN_Regressor__n_trees):  # Accéder au nombre d'arbres
        with model._DJINN_Regressor__sess[p].graph.as_default():  # Accéder au graphe de chaque arbre
            variables = tf.compat.v1.trainable_variables()  # Utiliser la version compat.v1
            total_params = sum([np.prod(var.shape.as_list()) for var in variables])
            tree_params[f"tree_{p}"] = total_params
            print(f"Nombre de paramètres pour l'arbre {p} : {total_params}")
    return tree_params

# Charger le modèle DJINN
chargement_djinn = dj.load("dj9", model_path)

# Compter les paramètres pour chaque arbre
tree_parameters = count_parameters_per_tree(chargement_djinn)

# Afficher les résultats
print("Nombre de paramètres par arbre :", tree_parameters)

def count_total_paramètre(tree_parameters):
    total = sum(tree_parameters.values())
    print(f"Nombre total de paramètres dans le modèle DJINN : {total}")
    return total

total_paramètres = count_total_paramètre(tree_parameters)
print("Nombre total de paramètres dans le modèle DJINN :", total_paramètres)

model_rnn_path = ROOT_DIR + "/test_constantin/Entrainement_RNN/lstm_storm_model.h5"
model_rnn = load_model(model_rnn_path)

# /Users/constantingodard/Documents/Stage_ENS/rapport_et_soutenance_Guillaume_Tran/Donnees_Guillaume_path/
# pourJF/EntrainementDNN/LIBdjinn3/application/trained_on_stormsmeanKpTL/model9_10p_ep25000_ntrees20_lr0.002_bat5000/dj9.pkl

#Daa2D_KpT_light(case_name, model_path_res, model_dnn)
Daa2D_KpT_light("model9_10p_ep25000_ntrees20_lr0.002_bat5000", ROOT_DIR + "/test_constantin/resultat_juillet", chargement_djinn)

Daa2D_KpT_light("model9_10p_ep25000_ntrees20_lr0.002_bat5000", ROOT_DIR + "/test_constantin/resultat_juillet", model_rnn)
#Daa2D_appli_MAIN(case_name, origin, storm_trained, bool_train_by_L, 
#                   bool_train_by_storm, model_path_res, str_type) 


#Daa2D_3(model_name, case_name, origin) 
#        model_path = "ROOT_DIR + "/pourJF/EntrainementDNN/" + origin + "/" + case_name + '/'
origin = "LIBdjinn3/application/trained_on_stormsmeanKpTL"
Daa2D_3("dj9", "model9_10p_ep25000_ntrees20_lr0.002_bat5000", origin) 

### valeur de Kp pour chaque heure
Kp_t_map_median = {
    10: 4, 11: 3, 12: 2, 13: 3, 14: 2, 15: 2, 16: 2, 17: 2, 18: 1
}

Kp_t_map_5th_percentile = {
    10: 3, 11: 2, 12: 1, 13: 1, 14: 1, 15: 0, 16: 0, 17: 0, 18: 0
}

Kp_t_map_95th_percentile = {
    10: 5, 11: 5, 12: 4, 13: 5, 14: 4, 15: 3, 16: 3, 17: 3, 18: 3
}


def Daa2D_LT_light_modifie(case_name, model_path_res, model_dnn) : 
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import matplotlib.ticker as ticker
    import matplotlib.cm as cmx

    parameters = {'axes.labelsize':10 , 'axes.titlesize':8, 
        'figure.titlesize':8, 'xtick.labelsize':8, 'ytick.labelsize':8, 'axes.facecolor':'white'}
    plt.rcParams.update(parameters)

    # (L,t) grid
    L = np.arange(1.5,5.5,0.05)
    Time = np.arange(10,18,0.05)

    # Kp discret par heure
    Kp_t_map = {
        10: 5, 11: 5, 12: 4, 13: 5, 14: 4, 15: 3, 16: 3, 17: 3, 18: 3
    }

    # Paramètres fixes
    angle = np.array([30,49,69,86])
    E = np.array([np.log10(0.1),np.log10(0.5),np.log10(1.),np.log10(2)])

    # Initialisation de la figure
    fig, axs = plt.subplots(4,4,constrained_layout=True,figsize=(9,7))
    fig.suptitle(r"$Log_{10}(D_{\alpha\alpha}) (L,t)$ avec $K_p(t)$ pour HSS $95^{\mathrm{th}}$ percentile")

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
                    kp_val = Kp_t_map.get(t_discrete, 2)  # default Kp=2

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

    fig.savefig(model_path_res + "/HDaaLT_" + case_name + ".png", dpi=300, format='png')
    print("Saved at : " + model_path_res + "/HDaaLT_" + case_name + ".png")

    return ()


Daa2D_LT_light_modifie("model9_10p_ep25000_ntrees20_lr0.002_bat5000", ROOT_DIR + "/test_constantin/resultat_juillet", chargement_djinn)

#--------------------------------------------------------------------------------------------------
### EXTRAPOLATION DE D_ALPHA EN UTILISANT LES INDICES KP EN CONTINU###
# On effectue une extrapolation en créant une fonction polynomiale passant 
# par les valeurs discrètes de Kp à un instant t donné. voir /test_constantin/test_data_set.ipynb
#--------------------------------------------------------------------------------------------------


def Daa2D_LT_light_interpolation(case_name, model_path_res, model_dnn) : 
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib 
    import matplotlib.ticker as ticker
    import matplotlib.cm as cmx

    from scipy.interpolate import interp1d

    # Données discrètes
    t_discrete_vals = np.array([10,10.5,11,12,13,14,15,16,17,18])
    Kp_vals = np.array([3, 2.5, 2, 1, 1, 1, 0, 0, 0, 0])

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

    fig.savefig(model_path_res + "/HDaaLT_" + case_name + ".png", dpi=300, format='png')
    print("Saved at : " + model_path_res + "/HDaaLT_" + case_name + ".png")

    return ()

Daa2D_LT_light_interpolation("model9_10p_ep25000_ntrees20_lr0.002_bat5000", ROOT_DIR + "/test_constantin/resultat_juillet", chargement_djinn)




###------------------------------------------------------------------------------
### reproduction figure 6 KLUTH avec Kp = [1,6,7,8,9]
###------------------------------------------------------------------------------

# def Daa2D_LT_light3(case_name, model_path_res, model_dnn, list_Kp) : 

Daa2D_LT_light3("model9_10p_ep25000_ntrees20_lr0.002_bat5000", ROOT_DIR + "/test_constantin/resultat_juillet", chargement_djinn, [0])

#Daa2D_appli(case_name, origin, storm_trained, bool_train_by_L, bool_train_by_storm, str_type) 
# Problème de chevauchement des légendes sur les figures
import matplotlib
matplotlib.use("PS")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.backends.backend_pdf
import matplotlib.cm as cmx
from matplotlib import rc
rc('text',usetex=False)
parameters = {'axes.labelsize':12 , 'axes.titlesize':8, 
		'figure.titlesize':10, 'xtick.labelsize':10, 'ytick.labelsize':10, 'axes.facecolor':'white'}
plt.rcParams.update(parameters)



def Daa2D_LT_light3_modife(case_name, model_path_res, model_dnn, list_Kp) : 
        
        for iKp in list_Kp :

                # (L,t) grid
                L = np.arange(1.5,5.5,0.05)
                Time = np.arange(10,18,0.05)

                # chosen values
                angle = np.array([30, 49, 69, 86])
                E = np.array([np.log10(0.1),np.log10(0.5), np.log10(1), np.log10(2) ])

                # figure initialisation
                fig, axs = plt.subplots(4,4,constrained_layout=True,figsize=(9,7))
                fig.suptitle(r"$Log_{10}(D_{\alpha\alpha}) (L,t)$ ")

                colorsMap = 'jet'
                cm = plt.get_cmap(colorsMap)
                cNorm = matplotlib.colors.Normalize(vmin=-9, vmax=-4)
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
                scalarMap.set_array([-9,-4])
                fig.colorbar(scalarMap,ax=axs.ravel().tolist(), label=r"$\log_{10}\left(D_{\alpha\alpha}\right)$ ($s^{-1}$)")

                index_fig =0
                index_fig_y =0
                for ia in angle:
                        for ie in E:
                                indata = np.empty((Time.shape[0]*L.shape[0],5))
                                ind=0
                                nT = Time.shape[0]
                                for iL in L:
                                        indata[ind*nT:(ind+1)*nT,0] = iKp
                                        indata[ind*nT:(ind+1)*nT,1] = Time
                                        indata[ind*nT:(ind+1)*nT,2] = iL
                                        indata[ind*nT:(ind+1)*nT,3] = ia
                                        indata[ind*nT:(ind+1)*nT,4] = ie
                                        ind+=1
                                pred = model_dnn.predict(indata)[:,0]
                                indata[:,1] = (indata[:,1]-9.)/3.
                                
                                iep = 10**ie

                                #font_size = 12
                                titl = r"$K_p=%2.1f\quad E=%2.1f\,MeV\quad \alpha=%2.1f$" % (iKp, iep, ia)

                                if(index_fig%4==0 and index_fig !=0):
                                        index_fig=0
                                        index_fig_y +=1
                                axs[index_fig_y,index_fig].set_title(titl)
                                axs[index_fig_y,index_fig].xaxis.set_major_locator(ticker.FixedLocator([1,2,3]))
                                axs[index_fig_y,index_fig].xaxis.set_minor_locator(ticker.FixedLocator([0.33, 0.67, 1.33, 1.67, 2.33, 2.67]))

                                axs[index_fig_y,index_fig].yaxis.set_major_locator(ticker.FixedLocator([2,3,4,5]))
                                axs[index_fig_y,index_fig].yaxis.set_minor_locator(ticker.FixedLocator([1.5, 2.5, 3.5, 4.5, 5.5]))
                                
                                print(pred.shape)
                                axs[index_fig_y,index_fig].scatter(indata[:,1],indata[:,2],c=scalarMap.to_rgba(pred),alpha=1, marker='.')
                                if(index_fig_y==4):                                                     
                                        axs[index_fig_y,index_fig].set_xlabel("Days")  
                                if(index_fig==0):
                                        axs[index_fig_y,index_fig].set_ylabel("L")      
                                index_fig +=1

                fig.savefig(model_path_res + "/HDaaLT3_" + case_name + "_Kp%s.png"%iKp,dpi=300,format='png')
                print("Saved at : " + model_path_res + "/HDaaLT3_" + case_name + "_Kp%s.png"%iKp)
        
        return ()

Daa2D_LT_light3_modife("model9_10p_ep25000_ntrees20_lr0.002_bat5000", ROOT_DIR + "/test_constantin/resultat_juillet", chargement_djinn, [0,5])




def Daa2D_appli_modif(case_name, origin, storm_trained, bool_train_by_L, bool_train_by_storm, str_type) :
        """
        Args :
                - case_name (str) : for file paths.
                - origin (str): LIBdjinn*, MMDNN_hybrid. for file paths.
                - storm_trained (str) : "trained_on_stormsmeanKpTL", "trained_on_stormsmeanKpT". for file paths.
                - bool_train_by_L (bool) : True if NN was trained at fixed L
                - bool_train_by_storm (bool) : True if NN was trained on a single storm (ex : mean storm, 1 case out of millions)
                - str_type (str) : "95per", "mean", "5per". the storm you want to play.
        Returns : 
        """ 
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib
        import matplotlib.ticker as ticker
        import matplotlib.cm as cmx
        start = tm.time()

        parameters = {'axes.labelsize':10 , 'axes.titlesize':8, 
		'figure.titlesize':8, 'xtick.labelsize':8, 'ytick.labelsize':8, 'axes.facecolor':'white'}
        plt.rcParams.update(parameters)
        
        # creating output directory
        if str_type=='95per' :
                storm = "95per_Storm"
        elif str_type=='mean' :
                storm = "Mean_Storm"
        elif str_type=='5per' :
                storm = "5per_Storm"

        if storm_trained == None :
                model_path_res = ROOT_DIR + "/pourJF/Reproduction_fig_Kluh2022/fig6_Daa2D/" + origin + "/" + case_name + "/"
        else :
                if bool_train_by_L == True :
                        if bool_train_by_storm == True :
                                model_path_res = ROOT_DIR + "/pourJF/Reproduction_fig_Kluh2022/fig6_Daa2D/" + origin + "/application/" + storm_trained + "/train_by_L/train_by_storm/" + storm + "/" + case_name + "/"
                        else :
                                model_path_res = ROOT_DIR + "/pourJF/Reproduction_fig_Kluh2022/fig6_Daa2D/" + origin + "/application/" + storm_trained + "/train_by_L/" + storm + "/" + case_name + "/"
                else :
                        model_path_res = ROOT_DIR + "/pourJF/Reproduction_fig_Kluh2022/fig6_Daa2D/" + origin + "/application/" + storm_trained + "/" + storm + "/" + case_name + "/"
        os.system("mkdir -p " + model_path_res)

        Time = np.arange(10, 19, 1)
        for i,iT in enumerate(Time) :
                
                # figure initialization. title below.
                fig, axs = plt.subplots(nrows=1, ncols=4, layout='constrained', figsize = (9,2.3))
                itp = (iT-9.)/3.
                if str_type=='95per' :
                        fig.suptitle(r"$95^{th}$ percentile storm. t=%s j"%("%2.1f"%itp))
                if str_type=='mean' :
                        fig.suptitle(r"Mean storm. t=%s j"%("%2.1f"%itp))
                if str_type=='5per' :
                        fig.suptitle(r"$5^{th}$ percentile storm. t=%s j"%("%2.1f"%itp))

                # colors
                cm = plt.get_cmap('jet')
                cNorm = matplotlib.colors.Normalize(vmin=-9, vmax=-4)
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
                fig.colorbar(scalarMap,ax=axs.ravel().tolist(), label = r"$\log_{10}\left(D_{\alpha\alpha}\right)$ ($s^{-1}$)")
                
                L = [2,3,4,5]
                for index_fig, iL in enumerate(L) :
                        print("L : ", iL)
                        if storm_trained == None :
                                data_model_path = ROOT_DIR + "/pourJF/data_application/data/PREDS/" + origin + "/" + case_name 
                        else :
                                if bool_train_by_L==True :
                                        case_nameL = case_name + "_L%s"%iL
                                        if bool_train_by_storm == True :
                                                data_model_path = ROOT_DIR + "/pourJF/data_application/data/PREDS/" + origin + "/" + storm_trained + "/train_by_L/train_by_storm/" + storm + "/" + case_nameL + "/full_data_%s_L%s.dat"%(str_type, iL)
                                        else :
                                                data_model_path = ROOT_DIR + "/pourJF/data_application/data/PREDS/" + origin + "/" + storm_trained + "/train_by_L/" + storm + "/" + case_nameL + "/full_data_%s_L%s.dat"%(str_type, iL)
                                else :
                                        data_model_path = ROOT_DIR + "/pourJF/data_application/data/PREDS/" + origin + "/" + storm_trained + "/" + case_name + "/full_data_%s_L%s.dat"%(str_type, iL)
                        Path_Kp = np.loadtxt(ROOT_DIR + "/pourJF/data_application/data/DATA/Path_Kp_%s_L%s.dat"%(str_type, iL))

                        # Kp, t, L, alpha, log10E, log10Daa(data), log10Daa(pred)
                        dataL = np.loadtxt(data_model_path)
                        # these two below should be the same. I checked : yes.
                        dataLT = dataL[np.where(dataL[:,1]==iT)[0],:]
                        dataLTK = dataLT[np.where(dataLT[:,0]==Path_Kp[i])[0],:]
                        #print("shape LT, LTK : ", dataLT.shape, dataLTK.shape)
                        #print(dataLT==dataLTK, np.sum(dataLT==dataLTK*1))

                        angle = np.unique(dataLTK[:,3])  # depends on L
                        E = np.unique(dataLTK[:,4])     # always the same
                        print('angle : ', np.shape(angle))
                        print('E : ', np.shape(E))

                #### can do without this and plot dataLTK directly instead of data_to_plot

                        #alpha, E, log10Daa(pred)
                        data_to_plot = np.empty((0,7))
                        for ia in angle :
                                print('ia : ', ia)
                                dataLTKA = dataLTK[np.where(dataLTK[:,3]==ia)[0],:]
                                print(np.shape(dataLTKA))
                                print('E LTKA : ', np.shape(np.unique(dataLTKA[:,4])))
                                print('Daa LTKA : ', np.shape(dataLTKA[:,6]))
                                shape = np.shape(dataLTKA)
                                data_to_plot = np.append(data_to_plot, dataLTKA, axis=0)

                ###### can do without this and plot dataLTK directly instead of data_to_plot
                
                        # plot
                        if data_to_plot.size > 0:
                                # Scatter plot : alpha (col 3) vs 10^E (col 4), colorée selon col 6
                                axs[index_fig].scatter(
                                        data_to_plot[:, 3], 
                                        10 ** data_to_plot[:, 4], 
                                        c=scalarMap.to_rgba(data_to_plot[:, 6]), 
                                        alpha=1, 
                                        marker='o'
                                )

                                # Titre du subplot
                                axs[index_fig].set_title(
                                        r"$L=%s$" % iL + r"  $K_p=$" + str("%2.1f" % Path_Kp[i])
                                )

                                # Axe x : angle alpha
                                axs[index_fig].set_xlim(0, 90)
                                axs[index_fig].set_xlabel(r"$\alpha$ ($°$)")
                                axs[index_fig].xaxis.set_major_locator(
                                        ticker.FixedLocator([int(np.min(angle)), 50, int(np.max(angle))])
                                )
                                axs[index_fig].xaxis.set_minor_locator(
                                        ticker.FixedLocator([25, 75])
                                )

                                # Axe y : énergie en échelle log
                                axs[index_fig].set_yscale("log")
                                axs[index_fig].set_ylim(10 ** np.min(E), 10 ** np.max(E))
                                axs[index_fig].set_yticks(10 ** np.arange(np.floor(np.min(E)), np.ceil(np.max(E)) + 1))

                                # Optionnel : pour forcer un aspect ratio spécifique
                                # axs[index_fig].set_aspect('auto')  # ou 'equal', ou ratio numérique

                        else:
                                print(f"Aucune donnée à tracer pour subplot index_fig = {index_fig}, iL = {iL}")

                        
                axs[0].set_ylabel(" E ($MeV$)")       

                fig.savefig(model_path_res + "/HDaaAlphaE_" + str_type + "_t%s_"%str(iT) + case_name + ".png", dpi=300, format='png')
                print("Saved at : " +  model_path_res + "/HDaaAlphaE_" + str_type + "_t%s_"%str(iT) + case_name + ".png")

        end = tm.time()
        with open(model_path_res + "RUNTIME.txt", "a") as f :
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                f.write(dt_string + " --> RUNTIME " + case_name + str(end-start) + "\n")
        
        return ()

origin = "LIBdjinn3/trained_on_stormsmeanKpTL/model9_10p_ep25000_ntrees20_lr0.002_bat5000"
#model9_10p_ep25000_ntrees20_lr0.002_bat5000
Daa2D_appli_modif("full_data_mean_L2.dat",origin,None,False, False,'median')



def Daa2D_appli_MAIN_modifie(case_name, origin, storm_trained, bool_train_by_L, bool_train_by_storm, model_path_res, str_type) :
        """
        Args :
                - case_name (str) : for file paths.
                - origin (str): LIBdjinn*, MMDNN_hybrid. for file paths.
                - storm_trained (str) : "trained_on_stormsmeanKpTL", "trained_on_stormsmeanKpT". for file paths.
                - bool_train_by_L (bool) : True if NN was trained at fixed L
                - bool_train_by_storm (bool) : True if NN was trained on a single storm (ex : mean storm, 1 case out of millions)
                - str_type (str) : "95per", "mean", "5per". the storm you want to play.
        Returns : 
        """ 
        # matplotlib parameters
##        parameters = {'axes.labelsize':12 , 'axes.titlesize':10, 
##                'figure.titlesize':12, 'xtick.labelsize':10, 'ytick.labelsize':10, 'axes.facecolor':'white'}
##        plt.rcParams.update(parameters)

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib
        import matplotlib.ticker as ticker
        import matplotlib.cm as cmx

        start = tm.time()
        parameters = {'axes.labelsize':10 , 'axes.titlesize':8, 
		'figure.titlesize':8, 'xtick.labelsize':8, 'ytick.labelsize':8, 'axes.facecolor':'white'}
        plt.rcParams.update(parameters)
        
        if str_type=='95per' :
                storm = "95per_Storm"
        elif str_type=='mean' :
                storm = "Mean_Storm"
        elif str_type=='median' :
                storm = "Median_Storm"
        elif str_type=='5per' :
                storm = "5per_Storm"

        Time = np.arange(10, 19, 1)
        for i,iT in enumerate(Time) :
                
                # figure initialization. title below.
                fig, axs = plt.subplots(nrows=1, ncols=4, layout='constrained', figsize = (9,2.3))
                itp = (iT-9.)/3.
                if str_type=='95per' :
                        fig.suptitle(r"$95^{th}$ percentile storm. t=%s j"%("%2.1f"%itp))
                if str_type=='mean' :
                        fig.suptitle(r"Mean storm. t=%s j"%("%2.1f"%itp))
                if str_type=='median' :
                        fig.suptitle(r"Median storm. t=%s j"%("%2.1f"%itp))
                if str_type=='5per' :
                        fig.suptitle(r"$5^{th}$ percentile storm. t=%s j"%("%2.1f"%itp))

                # colors
                cm = plt.get_cmap('jet')
                cNorm = matplotlib.colors.Normalize(vmin=-9, vmax=-4)
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
                fig.colorbar(scalarMap,ax=axs.ravel().tolist(), label = r"$\log_{10}\left(D_{\alpha\alpha}\right)$ ($s^{-1}$)")
                
                L = [2,3,4,5]
                for index_fig, iL in enumerate(L) :
                        print("L : ", iL)
                        if storm_trained == None :
                                data_model_path = ROOT_DIR + "/pourJF/data_application/data/PREDS/" + origin + "/" + case_name 
                        else :
                                if bool_train_by_L==True :
                                        case_nameL = case_name + "_L%s"%iL
                                        if bool_train_by_storm == True :
                                                data_model_path = ROOT_DIR + "/pourJF/data_application/data/PREDS/" + origin + "/" + storm_trained + "/train_by_L/train_by_storm/" + storm + "/" + case_nameL + "/full_data_%s_L%s.dat"%(str_type, iL)
                                        else :
                                                data_model_path = ROOT_DIR + "/pourJF/data_application/data/PREDS/" + origin + "/" + storm_trained + "/train_by_L/" + storm + "/" + case_nameL + "/full_data_%s_L%s.dat"%(str_type, iL)
                                else :
                                        data_model_path = ROOT_DIR + "/pourJF/data_application/data/PREDS/" + origin + "/" + storm_trained + "/" + case_name + "/full_data_%s_L%s.dat"%(str_type, iL)
                        Path_Kp = np.loadtxt(ROOT_DIR + "/pourJF/data_application/data/DATA/Path_Kp_%s_L%s.dat"%(str_type, iL))

                        # Kp, t, L, alpha, log10E, log10Daa(data), log10Daa(pred)
                        dataL = np.loadtxt(data_model_path)
                        # these two below should be the same. I checked : yes.
                        dataLT = dataL[np.where(dataL[:,1]==iT)[0],:]
                        dataLTK = dataLT[np.where(dataLT[:,0]==Path_Kp[i])[0],:]
                        #print("shape LT, LTK : ", dataLT.shape, dataLTK.shape)
                        #print(dataLT==dataLTK, np.sum(dataLT==dataLTK*1))

                        angle = np.unique(dataLTK[:,3])  # depends on L
                        E = np.unique(dataLTK[:,4])     # always the same
                        print('angle : ', np.shape(angle))
                        print('E : ', np.shape(E))

                #### can do without this and plot dataLTK directly instead of data_to_plot

                        #alpha, E, log10Daa(pred)
                        data_to_plot = np.empty((0,7))
                        for ia in angle :
                                print('ia : ', ia)
                                dataLTKA = dataLTK[np.where(dataLTK[:,3]==ia)[0],:]
                                print(np.shape(dataLTKA))
                                print('E LTKA : ', np.shape(np.unique(dataLTKA[:,4])))
                                print('Daa LTKA : ', np.shape(dataLTKA[:,6]))
                                shape = np.shape(dataLTKA)
                                data_to_plot = np.append(data_to_plot, dataLTKA, axis=0)

                ###### can do without this and plot dataLTK directly instead of data_to_plot
                
                        # plot
                        if data_to_plot.size > 0:
                                # Scatter plot
                                axs[index_fig].scatter(
                                        data_to_plot[:, 3],                  # angle α
                                        10 ** data_to_plot[:, 4],            # énergie (log→linéaire)
                                        c=scalarMap.to_rgba(data_to_plot[:, 6]), 
                                        alpha=1, 
                                        marker='o'
                                )

                                # Titre du subplot
                                axs[index_fig].set_title(
                                        r"$L=%s$" % iL + r"  $K_p=$" + str("%2.1f" % Path_Kp[i])
                                )

                                # Axe X (alpha)
                                axs[index_fig].set_xlim(0, 90)
                                axs[index_fig].set_xlabel(r"$\alpha$ ($°$)")
                                axs[index_fig].xaxis.set_major_locator(
                                        ticker.FixedLocator([int(np.min(angle)), 50, int(np.max(angle))])
                                )
                                axs[index_fig].xaxis.set_minor_locator(
                                        ticker.FixedLocator([25, 75])
                                )

                                # Axe Y (Energie) – log scale
                                axs[index_fig].set_yscale("log")
                                axs[index_fig].set_ylim(10 ** np.min(E), 10 ** np.max(E))
                                axs[index_fig].set_yticks(10 ** np.arange(np.floor(np.min(E)), np.ceil(np.max(E)) + 1))

                        else:
                                axs[index_fig].set_visible(False)  # Cache le subplot vide (optionnel)
                                print(f" data_to_plot est vide pour index_fig = {index_fig}, iL = {iL}, Kp = {Path_Kp[i]}")

                        
                axs[0].set_ylabel(" E ($MeV$)")       

                fig.savefig(model_path_res + "/HDaaAlphaE_" + str_type + "_t%s_"%str(iT) + case_name + ".png", dpi=300, format='png')
                print("Saved at : " +  model_path_res + "/HDaaAlphaE_" + str_type + "_t%s_"%str(iT) + case_name + ".png")

        end = tm.time()
        with open(model_path_res + "RUNTIME.txt", "a") as f :
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                f.write(dt_string + " --> RUNTIME " + case_name + str(end-start) + "\n")
        
        return ()

Daa2D_appli_MAIN_modifie("preds_mean_L2.dat",origin,None,False, False,ROOT_DIR + "/test_constantin/resultat_juillet",'median')

# HDaaAlphaE_model9_10p_ep25000_ntrees20_lr0.002_bat5000

# def Daa2D_AlphaE_light(case_name, model_path_res, model_dnn)  : 
Daa2D_AlphaE_light("model9_10p_ep25000_ntrees20_lr0.002_bat5000",ROOT_DIR + "/test_constantin/resultat_juillet", chargement_djinn)






def Daa2D_AlphaE_light_modifie(case_name, model_path_res, model_dnn)  : 
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib
        import matplotlib.ticker as ticker
        import matplotlib.cm as cmx


        # Paramètres graphiques
        parameters = {
                'axes.labelsize': 10,
                'axes.titlesize': 8,
                'figure.titlesize': 8,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'axes.facecolor': 'white'
        }
        plt.rcParams.update(parameters)

        # Grille (alpha, E)
        angle = np.arange(10, 89, 0.05)
        E = np.arange(np.log10(0.1), np.log10(6), 0.05)

        # Valeurs choisies
        Time = np.array([10,11, 12, 13,14,15,16,17,18])
        L = np.array([2, 3, 4, 5])
        Kp_t_map_median = {
                10: 4, 11: 3, 12: 2, 13: 3, 14: 2, 15: 2, 16: 2, 17: 2, 18: 1
        }
        # <-- Mettre ici la séauence de Kp appropriée
        Kp = np.array([Kp_t_map_median[t] for t in Time])

        # Détermination dynamique du nombre de sous-figures
        n_panels = len(L) * len(Kp) * len(Time)
        ncols = 3
        nrows = int(np.ceil(n_panels / ncols))

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, layout='constrained', figsize=(10, 2.5 * nrows))
        fig.suptitle(r"$\log_{10}(D_{\alpha\alpha}) (\alpha,E)$ ")

        # Color map
        colorsMap = 'jet'
        cm = plt.get_cmap(colorsMap)
        cNorm = matplotlib.colors.Normalize(vmin=-9, vmax=-4)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        fig.colorbar(scalarMap, ax=axs.ravel().tolist(), label=r"$\log_{10}\left(D_{\alpha\alpha}\right)$ ($s^{-1}$)")

        axs = np.array(axs).reshape(-1)  # Conversion à un vecteur plat
        index_fig = 0

        for iL in L:
                for iKp in Kp:
                        for it in Time:
                                # Préparation des données d'entrée
                                indata = np.empty((angle.shape[0] * E.shape[0], 5))
                                ind = 0
                                nE = E.shape[0]

                                for ia in angle:
                                        indata[ind * nE:(ind + 1) * nE, 0] = iKp
                                        indata[ind * nE:(ind + 1) * nE, 1] = it
                                        indata[ind * nE:(ind + 1) * nE, 2] = iL
                                        indata[ind * nE:(ind + 1) * nE, 3] = ia
                                        indata[ind * nE:(ind + 1) * nE, 4] = E
                                        ind += 1

                                pred = model_dnn.predict(indata)[:, 0]
                                indata[:, 1] = (indata[:, 1] - 9.) / 3.
                                itp = (it - 9.) / 3.

                                titl = f"L={iL:.1f}  $K_p={iKp:.1f}$  t={itp:.1f}"
                                ax = axs[index_fig]

                                ax.set_title(titl)
                                ax.scatter(indata[:, 3], 10 ** indata[:, 4], c=scalarMap.to_rgba(pred), alpha=1, marker='.')
                                ax.set_yscale("log")

                                if index_fig // ncols == nrows - 1:
                                        ax.set_xlabel(r"$\alpha$ ($°$)")
                                if index_fig % ncols == 0:
                                        ax.set_ylabel(r"$E$ ($MeV$)")

                                index_fig += 1
        # Sauvegarde
        fig.savefig(model_path_res + "/HDaaAlphaE_" + case_name + ".png", dpi=300, format='png')
        print("Saved at : " + model_path_res + "/HDaaAlphaE_" + case_name + ".png")
        return ()


Daa2D_AlphaE_light_modifie("model9_10p_ep25000_ntrees20_lr0.002_bat5000",ROOT_DIR + "/test_constantin/resultat_juillet", chargement_djinn)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cmx
import matplotlib

Kp_t_map_median = {
    10: 4, 11: 3, 12: 2, 13: 3, 14: 2, 15: 2, 16: 2, 17: 2, 18: 1
}

Kp_t_map_5th_percentile = {
    10: 3, 11: 2, 12: 1, 13: 1, 14: 1, 15: 0, 16: 0, 17: 0, 18: 0
}

Kp_t_map_95th_percentile = {
    10: 5, 11: 5, 12: 4, 13: 5, 14: 4, 15: 3, 16: 3, 17: 3, 18: 3
}

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
    Time = np.array([16, 17, 18])  # 5 colonnes

    # Dictionnaire Kp médian
    Kp_t_map_95th_percentile = {
    10: 5, 11: 5, 12: 4, 13: 5, 14: 4, 15: 3, 16: 3, 17: 3, 18: 3
        }


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
            iKp = Kp_t_map_95th_percentile[it]
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

    fig.savefig(f"{model_path_res}/HDaaAlphaE_{case_name}.png", dpi=300, format='png')
    print(f"Saved at : {model_path_res}/HDaaAlphaE_{case_name}.png")
    return ()


Daa2D_AlphaE_light_modifie2("model9_10p_ep25000_ntrees20_lr0.002_bat5000",ROOT_DIR + "/test_constantin/resultat_juillet", chargement_djinn)
