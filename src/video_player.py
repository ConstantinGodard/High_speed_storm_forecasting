# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, colors
import numpy as np

def creation_video_comparison(states_true, states_pred, y_spline, num_storm, filename="video_all_L.mp4", fps=10):
    """
    Cree une video comparant True vs Pred pour L = 2,3,4,5 avec une barre Kp.
    """

    L_vals = [2, 3, 4, 5]
    n_frames = min(states_true[2].shape[0], len(y_spline)) #states_true[2].shape[0]   # 99 frames
    y_spline = y_spline[:n_frames]
    print("n_frames states :", states_true[2].shape[0])

    fig, axes_img = plt.subplots(
        len(L_vals), 3, figsize=(18, 20),
        gridspec_kw={'height_ratios': [1]*len(L_vals), 'width_ratios':[1,1,1]}
    )


    fig.suptitle(
    "$HSS$ n°{} \n $\log(D_{{\\alpha\\alpha}})$ $changes$ $over$ $time$".format(num_storm),
    fontsize=16,
    y=0.92
    )
    plt.subplots_adjust(hspace=0.25, wspace=0.3)

    # --- Colorbar ---
    cNorm = colors.Normalize(vmin=-9, vmax=-4)
    cmap = plt.cm.jet
    # --- Colorbar différence ---
    diff_vmax = 2.0  # à ajuster si besoin
    diff_norm = colors.TwoSlopeNorm(vmin=-diff_vmax, vcenter=0.0, vmax=diff_vmax)
    diff_cmap = plt.cm.seismic

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cNorm)
    sm.set_array([])
    """    cbar = fig.colorbar(sm, ax=axes_img.ravel().tolist(),
                        orientation='vertical', fraction=0.1, pad=0.1,
                        label=r"$\log(D_{\alpha\alpha})$")
    cbar.ax.yaxis.label.set_size(14)"""

    sm_diff = plt.cm.ScalarMappable(cmap=diff_cmap, norm=diff_norm)
    sm_diff.set_array([])
    """    cbar_diff = fig.colorbar(
        sm_diff,
        ax=axes_img[:, 2].ravel().tolist(),
        orientation='vertical',
        fraction=0.1,
        pad=0.1,
        label=r"$\log(D_{\alpha\alpha})_{true} - \log(D_{\alpha\alpha})_{pred}$"
    )
    cbar_diff.ax.yaxis.label.set_size(14)"""


    # --- Axes pour la barre Kp ---
    ax_kp = fig.add_axes([0.1, 0.02, 0.8, 0.05])
    ax_kp.set_xlim(0, 6)
    ax_kp.set_ylim(0, 1)
    ax_kp.set_yticks([])
    ax_kp.set_xlabel("$K_p$ index")

    bar = ax_kp.barh(0.5, y_spline[0], height=0.3, color='dodgerblue')
    kp_marker = ax_kp.axvline(y_spline[0], color='red', lw=2, linestyle='--')

    text_time = ax_kp.text(0.1, 1.2, "", ha='center', va='bottom', fontsize=12,
                           transform=ax_kp.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    text_kp = ax_kp.text(0.9, 1.2, "", ha='center', va='bottom', fontsize=12,
                         transform=ax_kp.transAxes, bbox=dict(facecolor='white', alpha=0.7))

    # --- Initialisation des images ---
    alpha_min_dict = {2: 18.8365, 3: 10.4515, 4: 7.6616, 5: 6.4779}
    alpha_max = 90
    y1 = 10**(-1.1327)
    y2 = 10**(0.77797)

    ims = []
    for i, L in enumerate(L_vals):
        # Colonne gauche  True
        ax = axes_img[i, 0]
        im = ax.imshow(np.rot90(states_true[L][0], k=1), cmap=cmap, norm=cNorm,
                       aspect='auto', extent=[alpha_min_dict[L], alpha_max, y1, y2])
        ax.set_ylabel("E (MeV)")
        ax.set_xlabel(r"$\alpha$ (°)")
        ax.set_title("$L =$ {}  Groundtruth".format(L))
        ax.set_xlim(6.47, alpha_max)
        ax.set_xticks([alpha_min_dict[L], 50, 90])
        ax.set_xticklabels(["{}".format(int(v)) for v in [alpha_min_dict[L], 50, 90]])
        ims.append(im)

        # Colonne droite → Pred
        ax = axes_img[i, 1]
        im = ax.imshow(np.rot90(states_pred[L][0], k=1), cmap=cmap, norm=cNorm,
                       aspect='auto', extent=[alpha_min_dict[L], alpha_max, y1, y2])
        ax.set_ylabel("E (MeV)")
        ax.set_xlabel(r"$\alpha$ (°)")
        ax.set_title("$L =$ {} - Pred".format(L))
        ax.set_xlim(6.47, alpha_max)
        ax.set_xticks([alpha_min_dict[L], 50, 90])
        ax.set_xticklabels(["{}".format(int(v)) for v in [alpha_min_dict[L], 50, 90]])
        ims.append(im)

        # Colonne 3 → Différence
        ax = axes_img[i, 2]
        diff0 = states_true[L][0] - states_pred[L][0]
        im = ax.imshow(
            np.rot90(diff0, k=1),
            cmap=diff_cmap,
            norm=diff_norm,
            aspect='auto',
            extent=[alpha_min_dict[L], alpha_max, y1, y2]
        )
        ax.set_ylabel("E (MeV)")
        ax.set_xlabel(r"$\alpha$ (°)")
        ax.set_title("$L =$ {}  True - Pred".format(L))
        ax.set_xlim(6.47, alpha_max)
        ax.set_xticks([alpha_min_dict[L], 50, 90])
        ax.set_xticklabels(["{}".format(int(v)) for v in [alpha_min_dict[L], 50, 90]])

        ims.append(im)

    # --- Colorbar principale (jet) entre Pred et Diff ---
    cax_main = fig.add_axes([0.635, 0.12, 0.015, 0.75])
    cbar = fig.colorbar(sm, cax=cax_main)
    cbar.set_label(r"$\log(D_{\alpha\alpha})$", fontsize=14)

    # --- Colorbar différence tout à droite ---
    cax_diff = fig.add_axes([0.92, 0.12, 0.015, 0.75])
    cbar_diff = fig.colorbar(sm_diff, cax=cax_diff)
    cbar_diff.set_label(
        r"$\log(D_{\alpha\alpha})_{true} - \log(D_{\alpha\alpha})_{pred}$",
        fontsize=14
    )


    # --- Fonction update ---
    def update(frame):
        t = 1 + 8 * frame / (n_frames - 1)
        Kp = y_spline[frame]

        # Assurer que Kp est un scalaire
        if isinstance(Kp, np.ndarray):
            Kp = Kp.item()

        for i, L in enumerate(L_vals):
            ims[3*i].set_array(np.rot90(states_true[L][frame], k=1))
            ims[3*i+1].set_array(np.rot90(states_pred[L][frame], k=1))
            diff = states_true[L][frame] - states_pred[L][frame]
            ims[3*i+2].set_array(np.rot90(diff, k=1))


        bar[0].set_width(Kp)
        kp_marker.set_xdata([Kp, Kp])
        text_time.set_text("$t$ = {:.2f}".format(t))
        text_kp.set_text("$K_p$ = {:.2f}".format(Kp))


        return ims + [bar[0], kp_marker, text_time, text_kp]

    # --- Animation ---
    print("n_frames =", n_frames)
    print("len(y_spline) =", len(y_spline))

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=100, blit=True)

    writer = animation.FFMpegWriter(fps=10, codec='mpeg4', bitrate=8000)
    ani.save("{}".format(filename), writer=writer, dpi=120)
 
    plt.close()
    print("Video sauvegardee : {}".format(filename))

# # Paramètres
# L_vals = [2, 3, 4, 5]
# n_frames = 10
# ny, nx = 254, 60

# # --- States True et Pred aleatoires ---
# states_true = {L: np.random.rand(n_frames, ny, nx) for L in L_vals}
# states_pred = {L: np.random.rand(n_frames, ny, nx) for L in L_vals}

# # --- Valeurs Kp aleatoires ---
# y_spline = np.random.rand(n_frames) * 6  # entre 0 et 6

# # Verification
# print(states_true[2].shape)  # (10, 254, 60)
# print(states_pred[3].shape)  # (10, 254, 60)
# print(y_spline)              # tableau Kp
# creation_video_comparison(states_true, states_pred, y_spline, filename="test_video.mp4")

# if __name__ == "__main__":
#     creation_video_comparison(states_true, states_pred, y_spline, filename="test_video.mp4")
