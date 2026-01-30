import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import os
import pandas as pd
from scipy.interpolate import UnivariateSpline
import sys


# -------------------------------------------------------------------------------------------------------------------
        # Création du LSTM
# -------------------------------------------------------------------------------------------------------------------

input_dim  = 4*15240 + 1   # 60961
hidden_dim = 512
num_layers = 4
output_dim = 4*15240       # 60960, uniquement les états

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()

        self.in_norm  = nn.LayerNorm(input_dim)
        self.encoder  = nn.Linear(input_dim, 1024)
        self.dropout  = nn.Dropout(0.3)

        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.in_norm(x)
        x = self.encoder(x)
        x = self.dropout(x)

        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

    

model = LSTMForecaster(input_dim, hidden_dim, num_layers,output_dim).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()


# -------------------------------------------------------------------------------------------------------------------
        #
# -------------------------------------------------------------------------------------------------------------------


DATA_DIR = "/gpfs/users/godardc/HSS_augmented/"
L_vals = ['2', '3', '4', '5']

def to_tensor(storms):
    # storms: (N,100,60960)
    x = torch.from_numpy(storms).float()
    return x

def make_xy(states, kp):
    # states: (N, T, 60960)
    # kp    : (N, T, 1)

    x = torch.cat([states, kp], dim=-1)   # (N, T, 60961) -> entrée du modèle
    x_in  = x[:, :-1, :]                  # (N, T-1, 60961)
    y_out = states[:, 1:, :]              # (N, T-1, 60960) -> sortie cible
    return x_in, y_out


# -------------------------------------------------------------------------------------------------------------------
        # Charger les HSS
# -------------------------------------------------------------------------------------------------------------------

class StormDataset(torch.utils.data.Dataset):
    def __init__(self, indices, kp_tensor):
        self.indices = indices
        self.kp_tensor = kp_tensor  # torch.Size [32, 100, 1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        data = np.load(f"{DATA_DIR}/HSS{i}_augmented.npz", allow_pickle=True, mmap_mode='r')
        arr = np.stack([data[k] for k in L_vals], axis=0)      # (4,100,15240)
        arr = arr.transpose(1,0,2).reshape(100, -1)            # (100,60960)
        arr = torch.from_numpy(arr).float()                    # (100,60960)
        #print(f"HSS{i}: {arr.shape}")
        kp_seq = self.kp_tensor[i]                             # (100,1)
        x_in, y_out = make_xy(arr.unsqueeze(0), kp_seq.unsqueeze(0))  # ajoute batch dim
        return x_in.squeeze(0), y_out.squeeze(0)              # (99, 60961), (99, 60960)


ROOT_DIR = "/gpfs/users/godardc/hinge/Donnees_Guillaume_path/Donnees_Guillaume_path"

# -------------------------------------------------------------------------------------------------------------------
        # Chargement des séquences des Kp
# -------------------------------------------------------------------------------------------------------------------

def load_Kp_sequences():
    base_dir = os.path.join(
        ROOT_DIR, 
        "Frontiers21", 
        "UNI_PARIS_SACLAY_ARTICLE", 
        "32STORMS_DATA"
    )

    data_list = []
    # DAA2D_HSS1_Storm
    for i in range(1, 33):
        sequence_file = os.path.join(base_dir, f"DAA2D_HSS{i}_Storm", "Kp_sequence.dat")
        #print(f"Loading {sequence_file}...")

        if not os.path.exists(sequence_file):
            #print(f"Warning: {sequence_file} not found.")
            data_list.append({"storm_id": i, "sequence": None})
            continue

        try:
            seq = np.loadtxt(sequence_file)
        except Exception as e:
            #print(f"Error reading {sequence_file}: {e}")
            seq = None

        data_list.append({
            "storm_id": i,
            "sequence": seq
        })

    return pd.DataFrame(data_list)


def interpolate_to_100(seq):
    if seq is None:
        return None

    x = np.arange(len(seq))
    spline = UnivariateSpline(x, seq, s=0)

    x_new = np.linspace(0, len(seq)-1, 100)
    y_new = spline(x_new)

    return y_new  # shape (100,)


df_kp = load_Kp_sequences()

kp_all = []

for i in range(32):
    seq = df_kp.loc[df_kp.storm_id == (i+1), "sequence"].iloc[0]
    kp_interp = interpolate_to_100(seq)
    kp_all.append(kp_interp)

kp_all = np.array(kp_all)   # (32,100)
#print("kp_all shape :", kp_all.shape)

kp_all_tensor = torch.from_numpy(kp_all).float()   # (32,100)
kp_all_tensor = kp_all_tensor.unsqueeze(-1)        # (32,100,1)
#print("kp_all_tensor shape :", kp_all_tensor.shape)



# -------------------------------------------------------------------------------------------------------------------
        # Chargement des données et création des folds
# -------------------------------------------------------------------------------------------------------------------

ALL = np.arange(32) # [0..31] 
TEST_SIZE = 4 
test_idx = ALL[-TEST_SIZE:] # array([28,29,30,31]) 
trainval_idx = ALL[:-TEST_SIZE] # array([0..27]) 

kf = KFold(n_splits=7, shuffle=False) 

folds = [] 

for train_sub, val_sub in kf.split(trainval_idx): 
    train_idx = trainval_idx[train_sub] 
    val_idx = trainval_idx[val_sub] 
    folds.append((train_idx, val_idx))


def build_loaders(train_idx, val_idx, batch_size=1):

    train_ds = StormDataset(train_idx, kp_all_tensor)
    val_ds   = StormDataset(val_idx, kp_all_tensor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



# -------------------------------------------------------------------------------------------------------------------
        # Entrainement d'un fold
# -------------------------------------------------------------------------------------------------------------------


def train_one_fold(num_fold, model, train_loader, val_loader, optimizer, loss_fn, epochs=3):

    train_hist = []
    val_hist   = []

    for epoch in range(epochs):

        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.cuda()
            y = y.cuda()

            optimizer.zero_grad()

            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()

            # === Gradient clipping (très important pour LSTM avec gros vecteurs) ===
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.cuda()
                y = y.cuda()
                out = model(x)
                val_loss += loss_fn(out, y).item()

        train_epoch = total_loss / len(train_loader)
        val_epoch   = val_loss   / len(val_loader)

        train_hist.append(train_epoch)
        val_hist.append(val_epoch)

        print(f"Fold n°{num_fold} | Epoch {epoch+1} | Train {train_epoch:.4f} | Val {val_epoch:.4f}")

    return train_hist, val_hist



# -------------------------------------------------------------------------------------------------------------------
        # Rollout
# -------------------------------------------------------------------------------------------------------------------


@torch.no_grad()
def rollout(model, x0, horizon):
    # x0: (1,1,60960)
    model.eval()

    preds = []
    x_t = x0

    for _ in range(horizon):
        out = model(x_t)          # (1,1,60960)
        preds.append(out)
        x_t = out

    return torch.cat(preds, dim=1)   # (1, horizon, 60960)



import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------------
        # Affichage de la train_loss et val_loss
# -------------------------------------------------------------------------------------------------------------------


def plot_loss(train_history, val_history, fold_idx=0, save_path=None):
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_history)+1)
    
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_history, label="Train Loss", marker='o')
    plt.plot(epochs, val_history, label="Validation Loss", marker='x')
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Fold {fold_idx} - Training vs Validation Loss")
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()



# -------------------------------------------------------------------------------------------------------------------
        # Lancement du train
# -------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    optimizer1 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn1 = nn.MSELoss()

    fold_idx = int(sys.argv[1])        # numéro du fold à entraîner
    epochs   = 50        # nombre d'époques

    # récupérer indices du fold
    train_idx, val_idx = folds[fold_idx]

    # construire les DataLoaders
    print(f"Fold {fold_idx} - Construction des DataLoaders...")
    train_loader, val_loader = build_loaders(train_idx, val_idx, batch_size=1)
    print(f"Fold {fold_idx} - Train loader: {len(train_loader)} batches")
    print(f"Fold {fold_idx} - Val loader  : {len(val_loader)} batches")

    # entraîner
    print(f"Fold {fold_idx} - Début de l'entraînement...")
    train_hist, val_hist = train_one_fold(fold_idx, model, train_loader, val_loader, optimizer1, loss_fn1, epochs=epochs)
    print(f"Fold {fold_idx} - Entraînement terminé.")

    # sauvegarder le modèle
    model_path = f"lstm_fold{fold_idx}.pt"
    torch.save(model.state_dict(), model_path )
    print(f"Fold {fold_idx} - Modèle sauvegardé dans {model_path}")

    # Afficher la loss
    loss_plot_path = f"/gpfs/users/godardc/loss_fold{fold_idx}.png"
    print(f"Fold {fold_idx} - Sauvegarde du graphe des pertes dans {loss_plot_path}...")
    plot_loss(train_hist, val_hist, fold_idx=fold_idx, save_path=loss_plot_path)
    print(f"Fold {fold_idx} - Terminé.")


# squeue -u godardc -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
# sbatch run_file.batch
# scancel 9911286
# tail -f out.txt