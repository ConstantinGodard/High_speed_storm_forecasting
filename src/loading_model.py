import torch
import importlib
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os
from scipy import stats

# --------------------------
# Imports projet
# --------------------------
import CLSTM
import video_player

importlib.reload(CLSTM)
importlib.reload(video_player)

from CLSTM import HSSForecaster, StormDataset
from video_player import creation_video_comparison

# --------------------------
# Initialisation forecaster
# --------------------------
DATA_DIR = "/gpfs/users/godardc/HSS_augmented/"
ROOT_DIR = "/gpfs/users/godardc/hinge/Donnees_Guillaume_path/Donnees_Guillaume_path"

forecaster = HSSForecaster(DATA_DIR, ROOT_DIR)
# --------------------------
# Load modèle entraîné
# -------------------------- 
# objectif performance auato-encodeur fonction identité : 15_12__13h18
# Best model : 23_01__00h02 train :h15h6 - MSE 10^-1
# 2nd best model : 23_01__17h27 train historique + horizon variable MSE = 1.5 10^-1
# 3rd best : 21_01__15h22
# 
model_dir = "/gpfs/users/godardc/runs/23_01__00h02/"
ckpt = torch.load(model_dir+"best_model.pt")
forecaster.model.load_state_dict(ckpt["model_state_dict"], strict=False)
# model_dict = forecaster.model.state_dict()
# ckpt_dict = ckpt["model_state_dict"]

# filtered = {
#     k: v for k, v in ckpt_dict.items()
#     if k in model_dict and v.shape == model_dict[k].shape
# }

# model_dict.update(filtered)
# forecaster.model.load_state_dict(model_dict)

warmup_len_test = 15
horizon_test = 6
# --------------------------
# Loaders
# --------------------------
train_idx, val_idx = forecaster.folds[0]
test_idx = forecaster.test_idx

train_loader, val_loader = forecaster.build_loaders(train_idx, val_idx)
forecaster.scaler = train_loader.dataset.scaler

test_ds = StormDataset(
    test_idx,
    forecaster.kp_all_tensor,
    forecaster.DATA_DIR,
    forecaster.L_vals,
    scaler=forecaster.scaler
)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)


def split_states_by_L(state_tensor, L_vals=[2,3,4,5], ny=254):
    state_np = state_tensor.cpu().numpy() if hasattr(state_tensor, "cpu") else state_tensor
    T, total = state_np.shape
    nL = len(L_vals)
    pts_per_L = total // nL
    nx = pts_per_L // ny

    out = {}
    for i, L in enumerate(L_vals):
        arr = state_np[:, i*pts_per_L:(i+1)*pts_per_L].reshape(T, ny, nx)
        out[L] = arr
    return out

def process_test_batch_rollout_sliding(
    x_init, y_true, kp_seq, storm_idx, forecaster, scaler,
    warmup_len=warmup_len_test, horizon=horizon_test
):
    print("\n=== process_test_batch_rollout_sliding ===")
    print("x_init shape :", x_init.shape)
    print("y_true shape :", y_true.shape)
    print("kp_seq shape :", kp_seq.shape)

    B, T_total, D = y_true.shape

    pred_blocks = []

    # Boucle glissante : toujours du GT en entrée
    #for t in range(0, T_total - warmup_len, horizon):
    t = 0
    while t + warmup_len + horizon <= T_total:
        
        x_warmup = y_true[:, t : t + warmup_len, :]           # (B, 30, D)
        kp_future = kp_seq[:, t + warmup_len : t + warmup_len + horizon, :]
        
        # sécurité fin de séquence
        if kp_future.shape[1] < horizon:
            break

        with torch.no_grad():
            y_pred_6 = forecaster.rollout(
                x_warmup.cuda(),
                kp_future.cuda(),
                training=False
            ).cpu()                                           # (B, 6, D)

        pred_blocks.append(y_pred_6)

        t +=  horizon


    # Concaténation des prédictions
    y_pred_all = torch.cat(pred_blocks, dim=1)                # (B, T_pred, D)

    # -------------------------
    # Dénormalisation (1ère tempête)
    # -------------------------
    y_true_denorm = scaler.inverse_transform(y_true[0])
    y_pred_denorm = scaler.inverse_transform(y_pred_all[0])

    print("y_true_denorm min/max:", y_true_denorm.min(), y_true_denorm.max())
    print("y_pred_denorm min/max:", y_pred_denorm.min(), y_pred_denorm.max())

    # -------------------------
    # Split par L-shell
    # -------------------------
    states_true_L = split_states_by_L(y_true_denorm)

    states_pred_L = {}
    for L in [2, 3, 4, 5]:
        gt_warmup = states_true_L[L][:warmup_len]              # (30, ny, nx)
        pred_future = split_states_by_L(y_pred_denorm)[L]      # (T_pred, ny, nx)

        states_pred_L[L] = np.concatenate(
            [gt_warmup, pred_future],
            axis=0
        )

    # -------------------------
    # Kp aligné timeline vidéo
    # -------------------------
    total_frames = states_pred_L[2].shape[0]
    y_spline = kp_seq[0, :total_frames, 0].cpu().numpy()

    print("n_frames states :", total_frames)
    print("len y_spline   :", len(y_spline))
    print("Sliding rollout OK\n")

    return states_true_L, states_pred_L, y_spline, storm_idx

# --------------------------
# Génération vidéos
# --------------------------

def generate_videos(loader, split_name, n_batches=3):
    for b, (x, y, kp, storm_idx) in enumerate(loader):
        if b >= n_batches:
            break

        B, T_total, D = y.shape
        pred_all = []
        t = 0

        #print(f"\n--- Batch {b+1}, storm {storm_idx[0].item()} ---")
        while t + warmup_len_test + horizon_test <= T_total:
            # Warmup à partir des frames réelles
            x_warmup = y[:, t:t + warmup_len_test, :]
            kp_future = kp[:, t + warmup_len_test : t + warmup_len_test + horizon_test, :]

            #print(f"t = {t}, warmup frames {t}-{t+warmup_len_test-1}, horizon frames {t+warmup_len_test}-{t+warmup_len_test+horizon_test-1}")

            # Rollout autoregressif sur l'horizon
            with torch.no_grad():
                y_pred = forecaster.rollout(
                    x_warmup.cuda(),
                    kp_future.cuda(),
                    training=False
                ).cpu()

            pred_all.append(y_pred)

            # Avancer l'index strictement d'un horizon
            t += horizon_test

        # Concaténation des blocs prédites
        y_pred_all = torch.cat(pred_all, dim=1)

        # Dénormalisation
        y_true_denorm = forecaster.scaler.inverse_transform(y[0])
        y_pred_denorm = forecaster.scaler.inverse_transform(y_pred_all[0])

        # Split par L-shell
        states_true_L = split_states_by_L(y_true_denorm)
        states_pred_L = {}
        for L in [2, 3, 4, 5]:
            gt_warmup = states_true_L[L][:warmup_len_test]  # warmup initial
            pred_future = split_states_by_L(y_pred_denorm)[L]
            states_pred_L[L] = np.concatenate([gt_warmup, pred_future], axis=0)

        # Kp aligné pour vidéo
        total_frames = states_pred_L[2].shape[0]
        y_spline = kp[0, :total_frames, 0].cpu().numpy()

        storm_id = storm_idx[0].item()
        filename = f"{split_name}_storm_{storm_id}.mp4"
        print(f"Création vidéo : {filename}")

        creation_video_comparison(
            states_true={L: states_true_L[L] for L in [2,3,4,5]},
            states_pred={L: states_pred_L[L] for L in [2,3,4,5]},
            y_spline=y_spline,
            num_storm=storm_id,
            filename=filename
        )


# --------------------------
# Calcul MSE par frame pour un batch
# --------------------------
def compute_val_mse_per_frame(x, y, kp, forecaster, scaler, warmup_len=warmup_len_test, horizon=horizon_test):
    """
    Calcule la MSE par frame pour un batch, avec sliding rollout.
    """
    B, T_total, D = y.shape
    mse_per_frame = []

    pred_blocks = []

    #for t in range(0, T_total - warmup_len, horizon):
    t = 0
    while t + warmup_len + horizon <= T_total:

        x_warmup = y[:, t:t+warmup_len, :]
        kp_future = kp[:, t+warmup_len:t+warmup_len+horizon, :]

        # sécurité fin de séquence
        if kp_future.shape[1] < horizon:
            break

        with torch.no_grad():
            y_pred = forecaster.rollout(
                x_warmup.cuda(),
                kp_future.cuda(),
                training=False
            ).cpu()

        # Découpe y_true exactement pour correspondre à la prédiction
        y_true_slice = y[:, t+warmup_len:t+warmup_len+horizon, :].cpu()

        T = min(y_pred.shape[1], y_true_slice.shape[1])

        for i in range(T):
            mse_frame = F.mse_loss(
                y_pred[:, i, :],
                y_true_slice[:, i, :]
            )
            mse_per_frame.append(mse_frame.item())
        t += horizon


    return np.array(mse_per_frame)  # shape = nombre total de frames prédites

def compute_naive_mse_per_frame(y, warmup_len=warmup_len_test, horizon=horizon_test):
    B, T_total, D = y.shape
    mse_per_frame = []

    for t in range(0, T_total - warmup_len, horizon):

        if t + warmup_len + horizon > T_total:
            break

        last_frame = y[:, t + warmup_len - 1, :].unsqueeze(1)
        y_pred = last_frame.repeat(1, horizon, 1)

        y_true_slice = y[:, t+warmup_len:t+warmup_len+horizon, :]

        T_pred = y_true_slice.shape[1]

        for i in range(T_pred):
            mse = F.mse_loss(y_pred[:, i, :], y_true_slice[:, i, :])
            mse_per_frame.append(mse.item())

    return np.array(mse_per_frame)



# --------------------------
# Génération et plot MSE cumulée
# --------------------------
def plot_mse_cumulated(loader, split_name, n_batches,save_path):
    """
    Calcule et plot les MSE frame par frame pour les n_batches
    et les affiche cumulées sur la même figure
    """
    #os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8,5))
    for b, (x, y, kp, storm_idx) in enumerate(loader):
        if b >= n_batches:
            break
        mse_frames = compute_val_mse_per_frame(x, y, kp, forecaster, forecaster.scaler)
        plt.plot(range(warmup_len_test+1, warmup_len_test+1+len(mse_frames)), mse_frames, label=f"{split_name} batch {b+1}")

    plt.yscale('log')
    plt.xlabel("Frame")
    plt.ylabel("MSE")
    plt.title(f"MSE par frame - {split_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure sauvegardée dans {save_path}")

def plot_mse_cumulated_naive(loader, split_name, n_batches, save_path):
    """
    Plot MSE frame par frame pour le prédicteur naïf.
    """
    plt.figure(figsize=(8,5))

    for b, (x, y, kp, storm_idx) in enumerate(loader):
        if b >= n_batches:
            break

        mse_frames = compute_naive_mse_per_frame(y)
        plt.plot(
            range(warmup_len_test+1, warmup_len_test+1 + len(mse_frames)),
            mse_frames,
            label=f"{split_name} batch {b+1}"
        )

    plt.yscale("log")
    plt.xlabel("Frame")
    plt.ylabel("MSE")
    plt.title(f"MSE par frame – prédicteur naïf ({split_name})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Figure sauvegardée dans {save_path}")


# --------------------------
# Prédictuer naïf persisitent 
# --------------------------

def process_test_batch_naive_persistence(
    y_true, kp_seq, storm_idx, scaler,
    warmup_len=warmup_len_test, horizon=horizon_test
):
    """
    Prédicteur naïf : répète la dernière frame du warmup
    pour les frames futures (persistence).
    """
    B, T_total, D = y_true.shape
    pred_blocks = []

    for t in range(0, T_total - warmup_len, horizon):

        # dernière frame du warmup
        last_frame = y_true[:, t + warmup_len - 1, :].unsqueeze(1)  # (B, 1, D)

        # réplication sur l'horizon
        y_pred_6 = last_frame.repeat(1, horizon, 1)

        pred_blocks.append(y_pred_6)

    y_pred_all = torch.cat(pred_blocks, dim=1)

    # dénormalisation (1ère tempête)
    y_true_denorm = scaler.inverse_transform(y_true[0])
    y_pred_denorm = scaler.inverse_transform(y_pred_all[0])

    # split par L
    states_true_L = split_states_by_L(y_true_denorm)

    states_pred_L = {}
    for L in [2, 3, 4, 5]:
        gt_warmup = states_true_L[L][:warmup_len]
        pred_future = split_states_by_L(y_pred_denorm)[L]
        states_pred_L[L] = np.concatenate([gt_warmup, pred_future], axis=0)

    total_frames = states_pred_L[2].shape[0]
    y_spline = kp_seq[0, :total_frames, 0].cpu().numpy()

    return states_true_L, states_pred_L, y_spline, storm_idx


# --------------------------
# Comparaison performance, Best CLSTM vs naïf
# --------------------------
#model_dir = "/gpfs/users/godardc/ztest/"
def plot_naive_vs_clstm_three_splits(train_loader, val_loader, test_loader,
                                     warmup_len=warmup_len_test, horizon=horizon_test,
                                     filename=model_dir+"MSE/naive_vs_clstm_all_splits.png"):
    """
    Crée une figure avec 3 sous-plots verticaux : train / val / test
    MSE par frame, Naïf vs CLSTM, avec IC 90%.
    """
    def get_mse(loader):
        all_mse_clstm = []
        all_mse_naive = []
        for x, y, kp, storm_idx in loader:
            # CLSTM
            mse_clstm = compute_val_mse_per_frame(x, y, kp, forecaster, forecaster.scaler,
                                                  warmup_len=warmup_len, horizon=horizon)
            all_mse_clstm.append(mse_clstm)

            # Naïf
            mse_naive = compute_naive_mse_per_frame(y, warmup_len=warmup_len, horizon=horizon)
            all_mse_naive.append(mse_naive)

        all_mse_clstm = np.array(all_mse_clstm)
        all_mse_naive = np.array(all_mse_naive)
        return all_mse_clstm, all_mse_naive

    def mean_ci90(arr, small_sample=False):
        """
        Pour small_sample=True (val/test), retourne mean et min/max exact pour IC.
        """
        mean = arr.mean(axis=0)
        if small_sample or arr.shape[0] < 2:
            ci_lower = arr.min(axis=0)
            ci_upper = arr.max(axis=0)
        else:
            ci_val = stats.t.ppf(0.95, arr.shape[0]-1) * arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
            ci_lower = mean - ci_val
            ci_upper = mean + ci_val
        return mean, ci_lower, ci_upper

    # loaders et noms
    loaders = [train_loader, val_loader, test_loader]
    split_names = ["train", "val", "test"]

    plt.figure(figsize=(12, 15))

    for i, (loader, name) in enumerate(zip(loaders, split_names)):
        ax = plt.subplot(3,1,i+1)
        mse_clstm, mse_naive = get_mse(loader)

        #small_sample=True pour val et test
        small_sample = (name != "train")
        
        mean_clstm, ci_low_clstm, ci_up_clstm = mean_ci90(mse_clstm, small_sample=(name!="train"))
        mean_naive, ci_low_naive, ci_up_naive = mean_ci90(mse_naive, small_sample=(name!="train"))
        global_mse_clstm = mean_clstm.mean()
        global_mse_naive = mean_naive.mean()

        frames = np.arange(warmup_len+1, warmup_len+1 + mean_clstm.shape[0])

        ax.fill_between(range(len(mean_clstm)), ci_low_clstm, ci_up_clstm, color="blue", alpha=0.2)
        ax.fill_between(range(len(mean_naive)), ci_low_naive, ci_up_naive, color="red", alpha=0.2)
        ax.plot(mean_clstm, color="blue", label="CLSTM")
        ax.plot(mean_naive, color="red", label="Naive")

        text_str = (
            f"CLSTM mean MSE: {global_mse_clstm:.3f}\n"
            f"Naive mean MSE: {global_mse_naive:.3f}"
        )

        ax.text(
            0.98, 0.02, text_str,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )
        ax.set_yscale('log')
        ax.set_xlabel("Frame prédite")
        ax.set_ylabel("MSE moyenne par frame")
        ax.set_title(f"MSE par frame - {name}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Figure sauvegardée : {filename}")


def plot_loss(train_hist, val_hist,fixed_val_hist, save_path=None):
    """
    Plot train / validation loss curves.

    Args:
        train_hist (list or array): training loss per epoch
        val_hist   (list or array): validation loss per epoch
        save_path  (str, optional): path to save the figure (e.g. 'loss.png')
    """
    epochs = range(1, len(train_hist) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_hist, label="Train loss", linewidth=2)
    plt.plot(epochs, val_hist, label="Val loss", linewidth=2)
    plt.plot(epochs, fixed_val_hist, label="Val loss fixed", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        #os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Loss plot saved to {save_path}")

    plt.show()


def plot_val_mse_per_frame(val_mse_per_frame_last_epoch, save_path=None):
    """
    Plot MSE par frame sur l'horizon de validation.

    Args:
        val_mse_per_frame_last_epoch (torch.Tensor ou np.array): MSE par frame (shape = Horizon)
        save_path (str, optional): chemin pour sauvegarder la figure
    """
    # Si c'est un tensor PyTorch, on convertit en numpy
    if hasattr(val_mse_per_frame_last_epoch, "cpu"):
        vals = val_mse_per_frame_last_epoch.cpu().numpy()
    else:
        vals = val_mse_per_frame_last_epoch

    plt.figure(figsize=(8, 4))
    plt.plot(vals, marker='o', linewidth=2)
    plt.xlabel("Frame (horizon)")
    plt.ylabel("MSE validation")
    plt.yscale("log")
    plt.title("MSE par frame sur horizon de validation")
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée dans {save_path}")

    plt.show()

# --------------------------
# Fonction Vilon plot
# --------------------------


def plot_violin_rmse_per_hss_loader(
    loader,
    split_name,
    forecaster,
    scaler,
    n_batches=3,
    warmup_len=warmup_len_test,
    horizon=horizon_test,
    save_path=None
):
    """
    Violin plot RMSE horizon pour chaque HSS dans un loader (train/val/test).
    Deux violons par HSS : CLSTM vs Naïf, couleurs différentes.
    """
    all_rmse_clstm = []
    all_rmse_naive = []

    batch_count = 0
    for x, y, kp, storm_idx in loader:
        batch_count += 1
        if batch_count > n_batches:
            break

        B, T_total, D = y.shape

        rmse_clstm_per_hss = [[] for _ in range(B)]
        rmse_naive_per_hss = [[] for _ in range(B)]

        t = 0
        while t + warmup_len + horizon <= T_total:
            x_warmup = y[:, t:t+warmup_len, :]
            kp_future = kp[:, t+warmup_len:t+warmup_len+horizon, :]
            y_true_slice = y[:, t+warmup_len:t+warmup_len+horizon, :]

            # CLSTM
            with torch.no_grad():
                y_pred_clstm = forecaster.rollout(
                    x_warmup.cuda(),
                    kp_future.cuda(),
                    training=False
                ).cpu()

            # Naïf
            last_frame = y[:, t+warmup_len-1, :].unsqueeze(1)
            y_pred_naive = last_frame.repeat(1, horizon, 1)

            for b in range(B):
                mse_clstm = F.mse_loss(y_pred_clstm[b], y_true_slice[b]).item()
                mse_naive = F.mse_loss(y_pred_naive[b], y_true_slice[b]).item()
                rmse_clstm_per_hss[b].append(np.sqrt(mse_clstm))
                rmse_naive_per_hss[b].append(np.sqrt(mse_naive))

            t += horizon

        all_rmse_clstm.extend(rmse_clstm_per_hss)
        all_rmse_naive.extend(rmse_naive_per_hss)

    # =========================
    # Violin plot
    # =========================
    plt.figure(figsize=(10, 6))
    positions = []
    data = []
    colors = []
    labels = []

    pos = 1
    for i, (clstm_vals, naive_vals) in enumerate(zip(all_rmse_clstm, all_rmse_naive)):
        # CLSTM
        data.append(clstm_vals)
        positions.append(pos)
        colors.append("blue")
        labels.append(f"HSS {i+1}\nCLSTM")
        pos += 1
        # Naïf
        data.append(naive_vals)
        positions.append(pos)
        colors.append("red")
        labels.append(f"HSS {i+1}\nNaif")
        pos += 2  # espace entre HSS

    parts = plt.violinplot(data, positions=positions, showmeans=True, showextrema=True)

    # Appliquer les couleurs
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.5)

    plt.xticks(positions, labels)
    plt.ylabel("RMSE horizon")
    plt.title(f"RMSE par horizon – {split_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Violin plot sauvegardé dans {save_path}")

    plt.show()


# --------------------------
# Appels pour chaque split
# --------------------------

if __name__ == "__main__":
    print("------------------------")
    print("Début plot MSE per frame")
    # plot_mse_cumulated(train_loader, "train", n_batches=26, save_path=model_dir+"MSE/train_mse_per_frame.png")
    # plot_mse_cumulated(val_loader,   "val",   n_batches=3, save_path= model_dir+"MSE/val_mse_per_frame.png")
    # plot_mse_cumulated(test_loader,  "test",  n_batches=3, save_path=model_dir+"MSE/test_mse_per_frame.png")
    # print("Fin plot MSE per frame")
    # print("------------------------")
    # print("Test prédicteur naïf")
    # print("Début plot MSE per frame")
    # plot_mse_cumulated_naive(train_loader, "train", 26, save_path=model_dir+"MSE/naive_train_mse.png")
    # plot_mse_cumulated_naive(val_loader,   "val",   3,  save_path=model_dir+"MSE/naive_val_mse.png")
    # plot_mse_cumulated_naive(test_loader,  "test",  3,  save_path=model_dir+"MSE/naive_test_mse.png")
    # plot_naive_vs_clstm_three_splits(train_loader, val_loader, test_loader)
    # print("Fin plot MSE per frame")
    # print("fin test prédicteur naïf")
    # print("------------------------")
    # print("Début génération vidéos")
    # generate_videos(train_loader, f"{model_dir}videos/train", n_batches=3) 
    # generate_videos(val_loader,   f"{model_dir}videos/val",   n_batches=3)
    # generate_videos(test_loader,  f"{model_dir}videos/test",  n_batches=3)
    # print("Fin génération vidéos")
    # Récupérer le premier batch du test_loader


    # Train
    plot_violin_rmse_per_hss_loader(
        train_loader, "train",
        forecaster, forecaster.scaler,
        n_batches=26,
        save_path=model_dir+"MSE/violin_rmse_train.png"
    )

    # Val
    plot_violin_rmse_per_hss_loader(
        val_loader, "val",
        forecaster, forecaster.scaler,
        n_batches=3,
        save_path=model_dir+"MSE/violin_rmse_val.png"
    )

    # Test
    plot_violin_rmse_per_hss_loader(
        test_loader, "test",
        forecaster, forecaster.scaler,
        n_batches=3,
        save_path=model_dir+"MSE/violin_rmse_test.png"
    )

    print("------------------------")


# ==============================================================================
# Commandes bash SLURM
# ==============================================================================
# sbatch loading_model.batch
# tail -f out_loading_model.txt
    