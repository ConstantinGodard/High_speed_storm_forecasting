# High-Speed Storm Forecasting using C-LSTM

## Description
Prédiction des coefficients de diffusion des électrons qui driftent dans la ceinture de Van Allen, pour différentes L-shells.
Dataset reprenant 32 tempêtes géomagnétiques (HSS) sur 3 jours (1 mesure toutes les 8 heures => 9 mesures par tempête).
5 variables observées : Kp (indice géomagnétique), t (temps 1->9), L (L-shell : 2->5), alpha (angle de pitch : alpha_min(L) -> 90°), E (niveau d'énergie des électrons en MeV)
1 variable calculée par J-F Ripoll : coefficient de diffusion D_alpha

## Prérequis pour utilisation sur le meso-centre
- Python 3.9.23
- PyTorch 1.12.1
- CUDA 11.3
- ffmpeg 6.1.2 (installé via conda-forge, nécessaire pour l'utilisation de `video_player.py` permettant de visualiser les HSS)
- GLIBC 2.17 (!!!! nécessaire pour l'utilisation du serveur via clé SSH => downgrade VSCode version <=1.85, voir documentation : https://mesocentre.pages.centralesupelec.fr/user_doc/)
- Voir le `requirements.txt` ou `torch_env_spec.txt`

## Détails ffmpeg
Built with gcc 13.3.0 (conda-forge), configuration includes: libx264, libx265, libvpx, libfreetype, libass, libopenh264, libaom, libsvtav1...


## Project structure C-LSTM
src/
 - CLSTM.py             : création du C-LSTM, chargement des données et fonction auxiliaires pour le training
 - test_CLSTM.py        : Lancement de l'entraînement en appelant les fonctions de `CLSTM.py`, de `loading_model.py` et de `videoplayer.py`
 - loading_model.py     : fonction permettant de visualiser les prédictions et les performances d'un modèle après entraînement
 - videoplayer.py       : fonction permettant la création d'une vidéo comparant le groundtruth et la prediction, affichant également la différence. Utilisation nécessaire de ffmpeg. 
 - tools.py             : fonction utilitaires pour le C-LSTM (vient de ce repo : https://github.com/chao-tan/FORECAST-CLSTM)

batch/
 - run_file_LSTM.batch  : permet l'execution de `test_CLSTM.py` sur un des GPU du serveur (GPU A100 ou Tesla V100), affiche les résultats dans `out.txt`
 - loading_model.batch  : permet l'execution de `loading_model.py` sur un des GPU du serveur, affiche les résultats dans `out_loading_model.txt`

augmentation/
 - snapshot_completions.py       : permet l'uniformisation des données (certains points manquants dans les données brutes), rajoute une plage de données égale à -15 pour des valeurs de L sans données
 - HSS_augmentation.py           : utilise les données complétées pour les augmenter selon la variable t (passe de 9 incréments de temps à n) en utilisant une ACP puis une RBF
 - POD.py                        : fonctions nécessaire pour effectuer une Analyse ne composantes principales crée par Rémy Vallot
 - RBF.py                        : fonction nécessaire pour interpoler les données crée par Rémy Vallot
 - energy_ratio_scrapping.ipynb  : étude des différentes manière d'effectuer l'ACP sur le jeu de donné

data_analysis/
 - interpolationmap.ipynb                      : vue d'ensemble sur la répartition des valeurs des Kp par HSS, et différentes interpolation
 - test_data_set.ipynb                         : visualisation de la répartiton des 6 metrics du jeu de donné

data/
 - HSS_completed/ : 32 HSS complétées (9 time increments) au format .npz
 - HSS_augmented/ : 32 HSS complétées (100 time increment) post ACP au format .npz
 - Kp_seq         : 32 séquences des Kp, relatives aux HSS au format .dat

output_text_files/
 - out_loading_model.txt
 - out.txt

others/
 - LSTM.py                                     : test de LSTM sur le jeu de données augmenté (100 incréments de temps)
 - test_LSTM.py                                : exécute l'entraînement de `LSTM.py`
 - Data_completion_script.ipynb                : carnet jupyter pour faire des tests POD-RBF sur une HSS
 - chargement_figure.py                        : chargement du meilleur modèle DJINN de Guillaume et permet la visualisation des prédictions grâce à une fonction interpolée des Kp (au lieu des valeurs discrètes avant)
 - figure_interpolation_pour_guillaume.py      : pareil que `chargement_figure.py` mais adapté au besoin spéciaux de Guillaume

