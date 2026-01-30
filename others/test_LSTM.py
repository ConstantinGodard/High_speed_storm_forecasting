import sys
sys.path.append("/gpfs/users/godardc")

import importlib
import hss_forecaster
importlib.reload(hss_forecaster)
# hss_forecaster_Lfixed ou hss_forecaster
from hss_forecaster import HSSForecaster, LSTMForecasterModel, StormDataset
import video_player
from torch.utils.data import Dataset, DataLoader


importlib.reload(video_player)

from video_player import creation_video_comparison
import numpy as np
import torch



DATA_DIR = "/gpfs/users/godardc/HSS_augmented/"
ROOT_DIR = "/gpfs/users/godardc/hinge/Donnees_Guillaume_path/Donnees_Guillaume_path"

forecaster = HSSForecaster(DATA_DIR, ROOT_DIR)
#folds = forecaster.prepare_folds(n_splits=1)
#train_idx, val_idx = folds[0]

train_idx, val_idx = forecaster.folds[0]  
test_idx = forecaster.test_idx            

train_loader, val_loader = forecaster.build_loaders(train_idx, val_idx)
forecaster.scaler = train_loader.dataset.scaler
print(forecaster.scaler.mean_.shape)
print("Scaler fitted OK")


test_ds = StormDataset(test_idx, forecaster.kp_all_tensor,
                       forecaster.DATA_DIR, forecaster.L_vals,
                       scaler=forecaster.scaler)

test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
print(test_loader.dataset.scaler)

train_hist, val_hist = forecaster.train_one_fold(
    fold_idx=0,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100
)

save_path = "loss_single_run.png"
forecaster.plot_loss(train_hist, val_hist, fold_idx="Pas de fold", save_path=save_path)

# -------------------------
# Évaluation finale sur test
# -------------------------
batch = next(iter(test_loader))
x, y = batch  
print(x.shape, y.shape)

test_true_states, test_pred_states, test_kp_seq = forecaster.evaluate_on_test(test_loader)

def split_states_by_L(state_tensor, L_vals=[2,3,4,5], ny=254):
    """
    state_tensor : (T, 60960)
    return dict[L] = (T, ny, nx)
    """
    state_np = state_tensor.cpu().numpy() if hasattr(state_tensor, "cpu") else state_tensor
    T, total = state_np.shape

    nL = len(L_vals)
    pts_per_L = total // nL
    nx = pts_per_L // ny

    out = {}

    for i, L in enumerate(L_vals):
        start = i * pts_per_L
        end   = (i+1) * pts_per_L
        arr = state_np[:, start:end].reshape(T, ny, nx)
        out[L] = arr

    return out


def process_test_batch(x, y, model, scaler):
    """
    x : (1, 99, 60961)
    y : (1, 99, 60960)
    """
    # y_true
    y_true = y.squeeze(0)            # (99, 60960)

    # prédiction
    with torch.no_grad():
        y_pred = model(x.cuda()).cpu().squeeze(0)   # (99, 60960)

    # Dénormalisation
    y_true_denorm = scaler.inverse_transform(y_true)
    y_pred_denorm = scaler.inverse_transform(y_pred)


    # Kp
    y_spline = x[0,:, -1].cpu().numpy()             # (99,)

    # split by L
    states_true_L = split_states_by_L(y_true_denorm)       # dict[L] = (99, 254, 60)
    states_pred_L = split_states_by_L(y_pred_denorm)

    return states_true_L, states_pred_L, y_spline


batch = next(iter(test_loader))
x, y = batch

states_true_L, states_pred_L, y_spline = process_test_batch(
    x, 
    y, 
    forecaster.model, 
    forecaster.scaler
)

for L in [2,3,4,5]:
    print(L, states_true_L[L].shape)

file_name_video = "test_loss.mp4"
creation_video_comparison(
    states_true=states_true_L,
    states_pred=states_pred_L,
    y_spline=y_spline,
    filename=file_name_video
)
print(f"Video saved at {file_name_video}")


# squeue -u godardc -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
# sbatch run_file_LSTM.batch
# tail -f out.txt
# scancel 9911286
# sinfo
# squeue -p gpu

# ------------
# Filtre regex tensorboard : ^\d{2}_\d{2}__\d{2}h\d{2}$
# Launch Tensorboard : tensorboard --logdir runs --port 6007 --host 0.0.0.0
# ------------
