import sys
sys.path.append("/gpfs/users/godardc")
import os
import importlib

# --------------------------
# Chargement du forecaster CLSTM
# --------------------------
print("\n=== Import CLSTM forecaster ===")
import CLSTM   
importlib.reload(CLSTM)

from CLSTM import CLSTM_M, StormDataset, HSSForecaster
print("OK : CLSTM_M importé")

# --------------------------
# Chargement vidéo player
# --------------------------
import video_player
importlib.reload(video_player)
from video_player import creation_video_comparison
print("OK : video_player importé")

import numpy as np
import torch
from torch.utils.data import DataLoader
import loading_model 
importlib.reload(loading_model)

DATA_DIR = "/gpfs/users/godardc/HSS_augmented/"
ROOT_DIR = "/gpfs/users/godardc/hinge/Donnees_Guillaume_path/Donnees_Guillaume_path"

print("\n=== Initialisation du Forecaster CLSTM ===")
forecaster = HSSForecaster(DATA_DIR, DATA_DIR)
print("Forecaster initialisé")

# --------------------------
# Loaders
# --------------------------
train_idx, val_idx = forecaster.folds[0]
test_idx = forecaster.test_idx

print(f"Train idx: {train_idx}")
print(f"Val idx:   {val_idx}")
print(f"Test idx:  {test_idx}")

print("\n=== Construction des DataLoaders ===")
train_loader, val_loader = forecaster.build_loaders(train_idx, val_idx)
forecaster.scaler = train_loader.dataset.scaler

print("Scaler mean min/max :", forecaster.scaler.mean_.min(), forecaster.scaler.mean_.max())
print("Scaler var min/max  :", forecaster.scaler.var_.min(), forecaster.scaler.var_.max())

x_sample, y_sample, kp_sample, _ = next(iter(train_loader))
print("x_sample min/max :", x_sample.min().item(), x_sample.max().item())
print("y_sample min/max :", y_sample.min().item(), y_sample.max().item())


print("Scaler stats:")
print(" mean shape :", forecaster.scaler.mean_.shape)
print(" var shape  :", forecaster.scaler.var_.shape)
print("Scaler OK\n")

test_ds = StormDataset(test_idx, forecaster.kp_all_tensor,
                       forecaster.DATA_DIR, forecaster.L_vals,
                       scaler=forecaster.scaler)

test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
print("Test loader OK")
print("Scaler utilisé dans test :", test_loader.dataset.scaler)

# --------------------------
# Entraînement
# -------------------------- 
print("\n=== Lancement de l'entraînement CLSTM ===")
train_hist, val_hist, fixed_val_hist, val_mse_per_frame_last_epoch, log_dir = forecaster.train_one_fold(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=1
)
os.makedirs(os.path.join(log_dir, "MSE"), exist_ok=True)
os.makedirs(os.path.join(log_dir, "videos"), exist_ok=True)

print("Fin entraînement")
print("Longueur train_hist :", len(train_hist))
print("Longueur val_hist   :", len(val_hist))

save_path = f"{log_dir}/MSE/CLSTM_rollout_loss_plot.png"

loading_model.plot_loss(train_hist, val_hist,fixed_val_hist,save_path)
print("Courbes de loss sauvegardées")

loading_model.plot_val_mse_per_frame(val_mse_per_frame_last_epoch, "Loss_per_frame.png")

# --------------------------
# Évaluation test
# --------------------------
print("\n=== Évaluation finale sur test ===")
batch = next(iter(test_loader))
x, y, kp, idx = batch

print("Batch shapes :")
print(" x :", x.shape)
print(" y :", y.shape)

print("Début plot MSE per frame")
loading_model.plot_mse_cumulated(train_loader, "train", n_batches=26, save_path=f"{log_dir }/MSE/train_mse_per_frame.png")
loading_model.plot_mse_cumulated(val_loader,   "val",   n_batches=3, save_path=f"{log_dir }/MSE/val_mse_per_frame.png")
loading_model.plot_mse_cumulated(test_loader,  "test",  n_batches=3, save_path=f"{log_dir }/MSE/test_mse_per_frame.png")
print("Fin plot MSE per frame")
print("Début perf  CLSTM vs naïf")
loading_model.plot_naive_vs_clstm_three_splits(train_loader, val_loader, test_loader,filename=f"{log_dir }/MSE/naive_vs_CLSTM.png")
print("Fin perf  CLSTM vs naïf")
print("Début génération vidéos")
loading_model.generate_videos(train_loader, f"{log_dir }/videos/train", n_batches=3)
loading_model.generate_videos(val_loader,   f"{log_dir }/videos/val", n_batches=3)
loading_model.generate_videos(test_loader,  f"{log_dir }/videos/test", n_batches=3)
print("Fin génération vidéos")


print("---------------------------\n")
print("    Vidéos sauvegardées    \n")
print("---------------------------")

# ==============================================================================
# Commandes bash SLURM
# ==============================================================================
# source activate torch_env (surtout pas conda activate sur SLURM)

# ------------
# TRAIN
# sbatch run_file_LSTM.batch
# tail -f out.txt
# squeue -u godardc (USER) 
# scancel 11481788 (JOBID)
# ------------

# ------------
# INFO GPU
# sinfo
# squeue -p gpu
# sinfo -N -p gpua100
# ------------

# ------------
# MONITORING TENSORBOARD
# Filtre regex tensorboard : ^\d{2}_\d{2}__\d{2}h\d{2}$
# Launch Tensorboard : tensorboard --logdir runs --port 6007 --host 0.0.0.0
# ------------
