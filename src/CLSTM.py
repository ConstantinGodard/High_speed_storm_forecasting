# -*- coding: utf-8 -*-
# /gpfs/users/godardc/hss_forecaster.py
import os
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch.optim as opti
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json
import tools


# -------------------- Modèle CLSTM --------------------
class CLSTM_M(nn.Module):
    def __init__(self):
        super().__init__()

        # INPUT : (4, 254, 60)
        self.h1, self.w1 = 254, 60

        # After pool1 (factor 2)
        self.h2, self.w2 = self.h1 // 2, self.w1 // 2

        # After pool2
        self.h3, self.w3 = self.h2 // 2, self.w2 // 2

        # --- Embedding du Kp ---
        self.embC = 32 
        self.kp_embed = nn.Sequential(
            nn.Linear(1, self.embC),
            nn.ReLU(),
            nn.Linear(self.embC, self.embC)
        )

        # --- ENCODER ---
        self.spconv1 = tools.FSCONV2D(4, 64, (3,3), stride=(1,1), padding=1)
        self.convlstm1 = tools.ConvLSTM(input_size=(self.h1, self.w1),
                                        input_dim=64, hidden_dim=64, kernel_size=(3,3))
        self.spool1 = tools.FSPOOL2D()

        self.spconv2 = tools.FSCONV2D(64, 128, (3,3), stride=(1,1), padding=1)
        self.convlstm2 = tools.ConvLSTM(input_size=(self.h2, self.w2),
                                        input_dim=128, hidden_dim=128, kernel_size=(3,3))
        self.spool2 = tools.FSPOOL2D()

        self.spconv3 = tools.FSCONV2D(128, 256, (3,3), stride=(1,1), padding=1)
        self.convlstm3 = tools.ConvLSTM(input_size=(self.h3, self.w3),
                                        input_dim=256 + self.embC, hidden_dim=256, kernel_size=(3,3), return_one=True)

        # --- DECODER ---
        self.dconv1 = tools.FSDCONV2D(256, 128, (3,3), stride=(1,1), padding=1)
        self.upool1 = tools.FSUNPOOLING()

        self.dconv2 = tools.FSDCONV2D(128, 64, (3,3), stride=(1,1), padding=1)
        self.upool2 = tools.FSUNPOOLING()

        self.dconv3 = tools.FSDCONV2D(64, 64, (3,3), stride=(1,1), padding=1)
        self.outlayer = tools.FSDCONV2D(64, 4, (1,1), stride=(1,1), padding=0)

        # --- U-Net additional convs ---
        self.up_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.merge2 = nn.Conv2d(256, 128, kernel_size=1)
        self.merge1 = nn.Conv2d(128, 64, kernel_size=1)

        self.skip1_alpha = nn.Parameter(torch.tensor(0.1))
        self.skip2_alpha = nn.Parameter(torch.tensor(0.1))
        self.skip2_logit = nn.Parameter(torch.tensor(-2.5))  # alpha ≈ 0.05


        # Structure Basique 
        # 60 960 -> 4 -> 32 -> 64 -> 128 		(encoder)
        # 128 -> 64 -> 32 -> 32 -> 4 -> 60 960	(decoder)


    def forward_step(self, x_t, kp_t, h3=None, c3=None):
        """
        Traite une seule frame (B, 60960) avec ConvLSTM uniquement au bottleneck.

        Entrées :
            x_t : (B, 60960)
            kp_t : (B,1) ou (B,)
            h3, c3 : hidden states du ConvLSTM bottleneck

        Retour :
            out_flat : (B, 60960)
            h3, c3   : hidden states mis à jour
        """
        device = x_t.device
        B = x_t.shape[0]

        # ------------------------------------------------
        # Reshape entrée
        # ------------------------------------------------
        x_t = x_t.reshape(B, 4, self.h1, self.w1)

        # =================================================
        # ENCODER (CNN PUR)
        # Avec ou sans skip connection, structure U-Net, testé en focntion des performpances
        # =================================================

        # -------- Bloc 1 --------
        y1 = F.leaky_relu(self.spconv1(x_t), inplace=True)  # (B,32,254,60)
        skip1 = y1

        y1p, ind1 = self.spool1(y1)                         # (B,32,127,30)

        # -------- Bloc 2 --------
        y2 = F.leaky_relu(self.spconv2(y1p), inplace=True)  # (B,64,127,30)
        skip2 = y2

        y2p, ind2 = self.spool2(y2)                         # (B,64,63,15)

        # -------- Bottleneck --------
        y3 = F.leaky_relu(self.spconv3(y2p), inplace=True)  # (B,128,63,15)

        # ------------------------------------------------
        # Injection de Kp au bottleneck
        # ------------------------------------------------
        if kp_t.dim() == 1:
            kp_t = kp_t.unsqueeze(-1)
        kp_t = kp_t.float().to(device)

        kp_emb = self.kp_embed(kp_t)                         # (B, embC)
        kp_map = kp_emb.unsqueeze(-1).unsqueeze(-1)          # (B, embC,1,1)
        kp_map = kp_map.expand(-1, -1, y3.shape[2], y3.shape[3])

        kp_map = kp_map * .10 # Accentue la dépendance au Kp
        y3 = torch.cat([y3, kp_map], dim=1)                  # (B,128+embC,63,15)

        # =================================================
        # ConvLSTM (MEMOIRE TEMPORELLE ICI SEULEMENT)
        # =================================================
        y3_in = y3.unsqueeze(1)                              # (B,1,C,H,W)

        if h3 is None or c3 is None:
            h3 = torch.zeros(
                B, self.convlstm3.hidden_dim,
                y3.shape[2], y3.shape[3],
                device=device
            )
            c3 = torch.zeros_like(h3)

        y3_out, (h3, c3) = self.convlstm3(
            y3_in,
            hidden_state=(h3, c3)
        )

        y_enc3 = y3_out                                      # (B,128,63,15)

        # =================================================
        # DECODER (U-Net)
        # =================================================

        # -------- Up 1 --------
        y = F.leaky_relu(self.dconv1(y_enc3), inplace=True)  # (B,64,63,15)
        y = F.interpolate(y, scale_factor=2, mode="nearest")# (B,64,126,30)
        y = F.leaky_relu(self.up_conv1(y), inplace=True)    # (B,64,126,30)

        if y.shape[2:] != skip2.shape[2:]:
            y = F.interpolate(y, size=skip2.shape[2:], mode="nearest")

        #alpha2 = torch.sigmoid(self.skip2_alpha)  # param scalaire
        #y = torch.cat([y, alpha2 * skip2], dim=1)
        alpha2 = torch.sigmoid(self.skip2_logit)

        # skip2 agit comme une correction, pas comme une copie
        skip2_corr = skip2 - y.detach()

        y = torch.cat([y, alpha2 * skip2_corr], dim=1)
        y = F.leaky_relu(self.merge2(y), inplace=True)       # (B,64,127,30)

        # -------- Up 2 --------
        y = F.leaky_relu(self.dconv2(y), inplace=True)       # (B,32,127,30)
        #y = self.upool2(y, ind1)                              # (B,32,254,60)
        y = F.interpolate(y, size=(self.h1, self.w1), mode="nearest")

        #if y.shape[2:] != skip1.shape[2:]:
        #    y = F.interpolate(y, size=skip1.shape[2:], mode="nearest")

        #alpha1 =  torch.sigmoid(self.skip1_alpha)
        #y = torch.cat([y, alpha1 * skip1], dim=1)                     # (B,64,254,60)
        #y = F.leaky_relu(self.merge1(y), inplace=True)       # (B,32,254,60)

        # -------- Output --------
        y = F.leaky_relu(self.dconv3(y), inplace=True)       # (B,32,254,60)
        y = self.outlayer(y)                                 # (B,4,254,60)

        out_flat = y.reshape(B, -1)                          # (B,60960)

        return out_flat, h3, c3

def horizon_schedule(epoch):
    if epoch < 5:
        return 20
    elif epoch < 10:
        return 30
    elif epoch < 20:
        return 40
    else:
        return 50

def get_alpha(horizon):
    if horizon <= 10:
        return 0.99
    elif horizon <= 20:
        return 0.97
    else:
        return 0.95


# -------------------- Dataset --------------------
class StormDataset(Dataset):
    def __init__(self, indices, kp_tensor, data_dir, L_vals, scaler):
        self.indices = indices
        self.kp_tensor = kp_tensor
        self.DATA_DIR = data_dir
        self.L_vals = L_vals
        self.scaler = scaler   # None = pas de normalisation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        data = np.load(
            f"{self.DATA_DIR}/HSS{i}_augmented.npz",
            allow_pickle=True,
            mmap_mode='r'
        )

        arr = np.stack([data[k] for k in self.L_vals], axis=0)
        arr = arr.transpose(1,0,2).reshape(100, -1)   # (100, 60960)

        if self.scaler is not None:
            arr = self.scaler.transform(arr)

        arr = torch.from_numpy(arr).float()

        kp_seq = self.kp_tensor[i]
        x_in, y_out = HSSForecaster.make_xy(arr.unsqueeze(0), kp_seq.unsqueeze(0))

        return x_in.squeeze(0), y_out.squeeze(0), kp_seq, i



# -------------------- Loss adaptée aux convolutions --------------------

def loss_exp(y_pred, y_true, reduction="mean"):
    """
    Implémente la loss Forecaster :
    exp(1 - min(|y|, |ŷ|)/20) * (y - ŷ)^2

    y_pred shape : torch.Size([1, 15240])
    y_true shape : torch.Size([1, 20, 60960])


    """
    #print(f" y_pred shape : {y_pred.shape}")
    #print(f" y_true shape : {y_true.shape}")

    abs_true = torch.abs(y_true)
    abs_pred = torch.abs(y_pred)

    min_val = torch.minimum(abs_true, abs_pred)

    weight = torch.exp(1 - min_val / 1.0) # Avec des données normalisées prendre 1, sinon prendre 20 puisque toutes les données sont comprises entre 

    loss_pixel = weight * (y_true - y_pred)**2

    if reduction == "mean":
        return loss_pixel.mean()
    elif reduction == "sum":
        return loss_pixel.sum()
    else:
        return loss_pixel

def loss_exp_temporal(y_pred, y_true, alpha=0.95, reduction="mean"):
    """
    Loss exp pondérée dans le temps.
    y_pred, y_true : (B, T, N)
    """
    # --- loss exp par pixel ---
    abs_true = torch.abs(y_true)
    abs_pred = torch.abs(y_pred)
    min_val = torch.minimum(abs_true, abs_pred)

    weight_pixel = torch.exp(1 - min_val / 1.0)
    loss_pixel = weight_pixel * (y_true - y_pred) ** 2   # (B,T,N)

    # --- pondération temporelle ---
    T = y_true.shape[1]
    time_weights = alpha ** torch.arange(T, device=y_true.device)  # (T,)
    time_weights = time_weights / time_weights.sum()

    loss_pixel = loss_pixel * time_weights[None, :, None]

    if reduction == "mean":
        return loss_pixel.mean()
    elif reduction == "sum":
        return loss_pixel.sum()
    else:
        return loss_pixel


# -------------------- Forecaster --------------------
class HSSForecaster:
    def __init__(self, data_dir, root_dir, 
                 L_vals=['2','3','4','5'], 
                 hidden_dim=32,num_layers=2):

        self.DATA_DIR = data_dir
        self.ROOT_DIR = root_dir
        self.L_vals = L_vals

        # Format d'entrée : 4 L-shells × 254 × 60 = 60960
        self.input_dim = 4*254*60
        self.output_dim = self.input_dim

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = 0.3

        # Normalisation
        self.scaler = StandardScaler()

        # -------------------- Modèle CLSTM --------------------
        self.model = CLSTM_M().cuda()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=7e-5, 
            weight_decay=5e-5
        )

        self.base_lr = 1e-4
        self.min_lr  = 1e-8

        # Tu mettras ici ta loss CLSTM
        self.loss_fn = nn.MSELoss()
        #self.loss_fn = loss_exp
        #self.loss_fn = lambda y_pred, y_true: loss_exp_temporal(y_pred, y_true, alpha=0.95)

        self.mse_info_fn = nn.MSELoss()

        # Scheduler
        """ 
        self.scheduler_min_lr = 1e-7
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.25,
            patience=15,
            min_lr=self.scheduler_min_lr
        )
         """
        self.kp_all_tensor = self.load_and_interpolate_kp()




        # Folds
        ALL = np.arange(32)
        np.random.seed(42)
        np.random.shuffle(ALL)

        self.train_idx = ALL[:26]
        self.val_idx   = ALL[26:29]
        self.test_idx  = ALL[29:32]
        self.folds = [(self.train_idx, self.val_idx)]

        # Stats
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: total={total:,}, trainable={trainable:,}")


    # -------------------- Méthodes --------------------
    @staticmethod
    def make_xy(states, kp):
        x = states
        x_in  = x[:, :-1, :]
        y_out = states[:, 1:, :]
        return x_in, y_out

    def load_and_interpolate_kp(self):
        df_kp = self.load_kp_sequences()
        kp_all = []
        for i in range(32):
            seq = df_kp.loc[df_kp.storm_id == (i+1), "sequence"].iloc[0]
            if seq is None:
                seq = np.zeros(100)  # ou np.nan si tu veux gérer NaN après
            else:
                seq = np.array(seq, dtype=float)

            kp_interp = self.interpolate_to_100(seq)
            kp_interp = np.array(kp_interp, dtype=float)  # s'assure du type float
            kp_all.append(kp_interp)

        kp_all = np.stack(kp_all)  # stack plutôt que np.array pour forcer shape uniforme
        kp_tensor = torch.from_numpy(kp_all).float().unsqueeze(-1)

        return kp_tensor

    def load_kp_sequences(self):
        base_dir = "/gpfs/users/godardc/Kp_seq"
        data_list = []

        for i in range(1, 33):
            sequence_file = os.path.join(
                base_dir,
                f"kp_sequence_hss{i}.dat"
            )

            if not os.path.exists(sequence_file):
                data_list.append({"storm_id": i, "sequence": None})
                continue

            try:
                seq = np.loadtxt(sequence_file)
            except Exception:
                seq = None

            data_list.append({
                "storm_id": i,
                "sequence": seq
            })

        return pd.DataFrame(data_list)


    @staticmethod
    def interpolate_to_100(seq):
        if seq is None:
            return None
        x = np.arange(len(seq))
        spline = UnivariateSpline(x, seq, s=0)
        x_new = np.linspace(0, len(seq)-1, 100)
        y_new = spline(x_new)
        return y_new


    def build_loaders(self, train_idx, val_idx, batch_size=1):

        # 1. Fit du scaler sur le train (une seule fois)
        scaler = StandardScaler()

        for i in train_idx:
            data = np.load(f"{self.DATA_DIR}/HSS{i}_augmented.npz",
                        allow_pickle=True, mmap_mode='r')
            arr = np.stack([data[k] for k in self.L_vals], axis=0)
            arr = arr.transpose(1,0,2).reshape(100, -1)
            scaler.partial_fit(arr)

        # 2. Recrée les datasets avec normalisation activée
        train_ds = StormDataset(train_idx, self.kp_all_tensor,
                                self.DATA_DIR, self.L_vals, scaler=scaler)

        val_ds = StormDataset(val_idx, self.kp_all_tensor,
                            self.DATA_DIR, self.L_vals, scaler=scaler)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    
    def get_shuffled_starts(self, T_indice, warmup_len, horizon):
        max_start = T_indice - (warmup_len + horizon)
        starts = torch.arange(0, max_start + 1)
        perm = torch.randperm(len(starts))
        return starts[perm]


    def train_one_fold(self, train_loader, val_loader, epochs=30):

        timestamp = datetime.now().strftime("%d_%m__%Hh%M")
        hparams = {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            #"scheduler_factor": self.scheduler.factor,
            #"scheduler_patience": self.scheduler.patience,
            #"scheduler_min_lr": self.scheduler_min_lr,
            "lr": self.optimizer.param_groups[0]["lr"],
            "weight_decay": self.optimizer.param_groups[0]["weight_decay"],
            "input_dim": self.input_dim,
            "output_dim": self.output_dim
        }

        log_dir = f"runs/{timestamp}"
        writer = SummaryWriter(log_dir=log_dir)
        with open(f"{log_dir}/hparams.json", "w") as f:
            json.dump(hparams, f, indent=4)

        train_hist, val_hist, fixed_val_hist = [], [], []
        best_val_loss = float("inf")
        best_val_mse_f1 = float("inf")
        epochs_no_improve = 0
        val_mse_per_frame_last_epoch = None
        epochs_per_window = 15

        # ============================================================
        # FIXED validation windows (constant across epochs)
        # ============================================================
        N_FIXED = 32
        warmup_len_fixed = 30
        horizon_fixed = 7

        # On suppose T_indice constant (sinon à prendre sur 1er batch)
        T_indice = next(iter(val_loader))[0].shape[1]
        max_start = T_indice - (warmup_len_fixed + horizon_fixed)

        fixed_starts = torch.linspace(
            0, max_start, steps=N_FIXED
        ).long()

        # fenêtres TRAIN
        T_indice_train = next(iter(train_loader))[0].shape[1]
        train_starts = self.get_shuffled_starts(T_indice_train, 30, 6)

        # fenêtres VAL
        T_indice_val = next(iter(val_loader))[0].shape[1]
        val_starts = self.get_shuffled_starts(T_indice_val, 30, 6)

        ema_mse = None
        beta = 0.95  # lissage  
        warmup_len_value = 15
        horizon_value = 6

        lambda_skip = 1e-3
        
        for epoch in range(epochs):

            # ----------------------------
            # Curriculum sur l'horizon
            # ----------------------------
            if epoch < 133:
                horizon_value = np.random.choice([3, 5])
            elif epoch < 266:
                horizon_value = np.random.choice([6, 8])
            else:
                horizon_value = np.random.choice([9, 12])

            if epoch % 50 == 0:
                print(f"[Epoch {epoch}] horizon = {horizon_value}")


            # ============================================================
            # TRAIN
            # ============================================================
            self.model.train()
            total_loss = 0

            # Comparaison avec les autres modèles avec MSE
            total_mse_info = 0.0

            train_mse_f1 = train_mse_f50 = train_mse_f99 = 0.0
            train_num_batches = 0

            # ----- scheduled sampling : exponential decay -----
            epsilon_start = 1.0          # teacher forcing initial
            epsilon_end   = 0.10         # valeur minimale souhaitée
            decay_rate    = 0.9955       # taux de décroissance

            # calcul de epsilon au début de chaque epoch
            effective_decay = decay_rate 
            current_eps = epsilon_start * (effective_decay ** epoch)

            # s’assure que current_eps n’est jamais inférieur à epsilon_end
            current_eps = max(current_eps, epsilon_end)
            #Horizon = horizon_schedule(epoch)
            p = epoch / max(1, epochs - 1)

            #lr = self.base_lr * (1 - p**1.5)
            lr = self.base_lr * (self.min_lr / self.base_lr) ** p
            lr = max(lr, self.min_lr)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            for x, y, kp, idx in train_loader:

                x = x.cuda()
                y = y.cuda()
                kp = kp.cuda()

                shuffle_kp = False

                warmup_len = warmup_len_value
                horizon = horizon_value
                T_indice = x.shape[1]

                position_idx = epoch // epochs_per_window
                #total_positions = T_indice - (warmup_len + horizon) + 1
                max_position = (T_indice - (warmup_len + horizon)) // epochs_per_window
                #p = position_idx / max(1, max_position)   # p ∈ [0, 1]

                max_start = T_indice - (warmup_len + horizon)
                start = torch.randint(0, max_start + 1, (1,)).item()


                x_warmup = x[:, start : start + warmup_len, :]
                kp_warmup = kp[:, start : start + warmup_len, :]
                y_target = y[:,start + warmup_len : start + warmup_len + horizon, :]


                #out = self.model(x, kp)
                #loss = self.loss_fn(out, y
                out = self.rollout(x_warmup, kp[:, start + warmup_len : start + warmup_len + horizon, :],y_true=y_target, training=True,tf_ratio=current_eps)

                #print(out.requires_grad)
                H = y_target.shape[1]
                alpha = get_alpha(H)

                #loss = loss_exp_temporal(out, y_target, alpha=alpha)

                loss = self.loss_fn(out, y_target)

                # ---- régularisation skip2 (TRAIN UNIQUEMENT) ----
                if hasattr(self.model, "skip2_logit"):
                    loss = loss + lambda_skip * torch.sigmoid(self.model.skip2_logit).pow(2)
                    
                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.80)
                self.optimizer.step()

                total_loss += loss.item()

                #print(f"Out.shape : {out.shape}")
                #print(f"Target.shape : {y_target.shape}")
                mse = (out - y_target)**2
                mse = mse.mean(dim=0).mean(dim=1)

                train_mse_f1  += mse[0].item()
                train_mse_f50 += mse[len(mse)//2].item()
                train_mse_f99 += mse[-1].item()

                    # ---- MSE informative (pas backward) ----
                mse_info = mse.mean().item()#self.mse_info_fn(out, y_target).item()
                total_mse_info += mse_info
                                
                train_num_batches += 1
            #print(f"train_num_batches : {train_num_batches}")
            train_loss = total_loss / train_num_batches
            train_MSE_info = total_mse_info / train_num_batches
            train_mse_f1  /= train_num_batches
            train_mse_f50 /= train_num_batches
            train_mse_f99 /= train_num_batches

            # ============================================================
            # VALIDATION (corrigé : x, y, kp)
            # ============================================================
            self.model.eval()
            val_loss = 0
            val_mse_f1 = val_mse_f50 = val_mse_f99 = 0.0
            val_num_batches = 0

            # stockage des erreurs par frame (uniquement dernière epoch)
            val_mse_frames_accum = []
            torch.manual_seed(epoch)
            with torch.no_grad():
                for x, y, kp, idx in val_loader:
                    x = x.cuda()
                    y = y.cuda()
                    kp = kp.cuda()

                    shuffle_kp = False

                    if shuffle_kp:
                        #perm = torch.randperm(kp.size(1), device=kp.device)
                        #kp = kp[:, perm, :]
                        kp = torch.zeros_like(kp)

                    # --- Configuration du palier ---
                    epochs_per_step = 3 
                    current_step = epoch // epochs_per_step
                    total_steps = epochs // epochs_per_step

                    warmup_len = warmup_len_value
                    horizon = horizon_value
                    T_indice = x.shape[1]

                    # position_idx = epoch // epochs_per_window

                    # if position_idx >= len(val_starts):
                    #     val_starts = self.get_shuffled_starts(T_indice, warmup_len, horizon)
                    #     position_idx = 0

                    # start = int(val_starts[position_idx])
                    max_start = T_indice - (warmup_len + horizon)
                    start = torch.randint(0, max_start + 1, (1,)).item()


                    x_warmup = x[:, start : start + warmup_len, :]
                    kp_warmup = kp[:, start : start + warmup_len, :]
                    y_target = y[:,start + warmup_len : start + warmup_len + horizon, :]


                    # Validation
                    out = self.rollout(x_warmup, kp[:, start + warmup_len : start + warmup_len + horizon, :],training=False)
                    #val_loss += self.loss_fn(out, y_target)#.item()
                    #loss_val_custom = loss_exp_temporal(out, y_target, alpha=alpha)
                    #val_loss += loss_val_custom.item()


                    mse = (out - y_target)**2
                    mse = mse.mean(dim=0).mean(dim=1)

                    #mse_info = mse.mean().item()
                    val_loss += mse.mean().item()

                
                    val_mse_frames_accum.append(mse.detach().cpu())

                    val_mse_f1  += mse[0].item()
                    val_mse_f50 += mse[len(mse)//2].item()
                    val_mse_f99 += mse[-1].item()

                    val_num_batches += 1

            #print(f"val_num_batches : {val_num_batches}")
            train_epoch = train_loss
            train_epoch_mse_info = train_MSE_info
            val_epoch = val_loss/val_num_batches
            train_hist.append(train_epoch)
            val_hist.append(val_epoch)

            val_mse_per_frame_last_epoch = torch.stack(val_mse_frames_accum, dim=0).mean(dim=0)
                # shape : (Horizon,)


            #self.scheduler.step(val_loss)

            val_mse_f1  /= val_num_batches
            val_mse_f50 /= val_num_batches
            val_mse_f99 /= val_num_batches


            # ============================================================
            # FIXED VALIDATION LOSS (stationnaire)
            # ============================================================
            self.model.eval()
            fixed_val_loss = 0.0
            fixed_val_batches = 0
            with torch.no_grad():
                for x, y, kp, idx in val_loader:
                    x = x.cuda()
                    y = y.cuda()
                    kp = kp.cuda()

                    batch_loss = 0.0

                    for start_fixed in fixed_starts:
                        start_fixed = int(start_fixed)

                        x_warmup = x[:, start_fixed:start_fixed+warmup_len_fixed, :]
                        kp_warmup = kp[:, start_fixed:start_fixed+warmup_len_fixed, :]

                        kp_fixed = kp[:, start_fixed+warmup_len_fixed :
                                        start_fixed+warmup_len_fixed+horizon_fixed, :]

                        out = self.rollout(x_warmup, kp_fixed, training=False)

                        H = out.shape[1]   # ← source de vérité
                        y_target = y[:, start_fixed+warmup_len_fixed :
                                        start_fixed+warmup_len_fixed+H, :]

                        mse = (out - y_target).pow(2).mean()
                        batch_loss += mse

                    batch_loss /= N_FIXED
                    fixed_val_loss += batch_loss.item()
                    fixed_val_batches += 1

            fixed_val_loss /= fixed_val_batches
            fixed_val_hist.append(fixed_val_loss)
            #scheduler.step(fixed_val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"\nEpoch {epoch+1}           | warmup_len={warmup_len_value} | horizon={horizon_value}\n"  #| warmup=[{start}:{start+warmup_len}]         | pred=[{start+warmup_len}:{start+warmup_len+ horizon}]\n"
                f"Train: loss={train_epoch_mse_info:.4f} | f1={train_mse_f1:.4f} | f50={train_mse_f50:.4f} | f99={train_mse_f99:.4f}\n"
                f"Val:   loss={val_epoch:.4f} | f1={val_mse_f1:.4f} | f50={val_mse_f50:.4f} | f99={val_mse_f99:.4f}\n"
                f"Val_fixed: loss={fixed_val_loss:.4f}         | N_seq_fixed = {N_FIXED}\n"
                f"-- loss_exp={train_epoch:.4f} | Current lr={current_lr:.1e}     | tf_ratio={current_eps:.2f} \n"
                f"-------------------------------------------------------------"
            )

            writer.add_scalar("Train/Loss", train_epoch_mse_info, epoch)
            writer.add_scalar("Val/Loss", val_epoch, epoch)
            writer.add_scalar("Train/MSE_frame1", train_mse_f1, epoch)
            writer.add_scalar("Train/MSE_frame50", train_mse_f50, epoch)
            writer.add_scalar("Train/MSE_frame99", train_mse_f99, epoch)
            writer.add_scalar("Val/MSE_frame1", val_mse_f1, epoch)
            writer.add_scalar("Val/MSE_frame50", val_mse_f50, epoch)
            writer.add_scalar("Val/MSE_frame99", val_mse_f99, epoch)
            writer.add_scalar("Val/Loss_fixed", fixed_val_loss, epoch)
            writer.add_scalar("LR", lr, epoch)


            # Early stopping logic
            early_stop_patience =  250
            if fixed_val_loss < best_val_loss:
                best_val_loss = fixed_val_loss
                epochs_no_improve = 0
                # Sauvegarder le meilleur modèle
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "hparams": hparams
                }, f"{log_dir}/best_model.pt")

            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    # Calculer val_mse_per_frame_last_epoch même si early stopping
                    val_mse_per_frame_last_epoch = torch.stack(val_mse_frames_accum, dim=0).mean(dim=0)
                    # Charger le meilleur modèle
                    #self.model.load_state_dict(torch.load("best_model.pt"))
                    break

            # Scheduler step
            #self.scheduler.step(val_loss)

        writer.close()

        # HPARAMS summary
        hparam_writer = SummaryWriter(log_dir + "/hparams")
        hparam_writer.add_hparams(
            hparam_dict=hparams,
            metric_dict={
                "best_val_loss": best_val_loss,
                "best_val_f1": best_val_mse_f1
            }
        )
        hparam_writer.close()


        return train_hist, val_hist, fixed_val_hist, val_mse_per_frame_last_epoch,log_dir

    #@torch.no_grad()
    def rollout(self, x0, kp_seq, y_true=None, training=True, tf_ratio=0.5):
        if training:
            self.model.train()
        else:
            self.model.eval()

        device = next(self.model.parameters()).device

        x0 = x0.to(device)              # (B, T_warmup, 60960)
        kp_seq = kp_seq.to(device)      # (B, horizon, 1)

        B, T_warmup, _ = x0.shape
        horizon = kp_seq.shape[1]

        preds = []

        # -------------------------
        # Init ConvLSTM state
        # -------------------------
        h3, c3 = None, None

        kp_dummy = torch.zeros(B, 1, device=device)

        # -------------------------
        # Warmup (IMPORTANT)
        # -------------------------
        for t in range(T_warmup):
            _, h3, c3 = self.model.forward_step(
                x0[:, t],        # vraie frame passée
                kp_dummy,        # PAS kp_seq ici
                h3, c3
            )
            #if t % 5 == 0 or t == T_warmup-1:
            #    print(f"Warmup frame {t}: h3 norm={h3.norm().item():.4f}, c3 norm={c3.norm().item():.4f}")
        # Dernière frame connue
        x_t = x0[:, -1]

        # -------------------------
        # Rollout autoregressif
        # -------------------------
        for t in range(horizon):
            out, h3, c3 = self.model.forward_step(
                x_t,
                kp_seq[:, t],
                h3, c3
            )
            preds.append(out.unsqueeze(1))

            # Teacher forcing
            if training and y_true is not None:
                use_tf = torch.rand(1).item() < tf_ratio
                x_t = y_true[:, t] if use_tf else out
            else:
                x_t = out

        return torch.cat(preds, dim=1)