# THIS FILE REDEFINES THE RBF INTERPOLATION METHOD

import numpy as np
from scipy.interpolate import Rbf
from sklearn.preprocessing import StandardScaler

class RBFInterpolator:
    """
    RBF interpolator for latent space POD coefficients.
    """

    def __init__(self):
        self.interpolators = []
        self.scaler = None
        self.trained = False

    def fit(self, params, coeffs, sample_weight=None):
        """
        Train RBF interpolators.

        Args:
            params (np.ndarray): shape (N, nb_var), input parameter configurations
            coeffs (np.ndarray): shape (D, N), latent coefficients (e.g., POD modes)
        """
        if sample_weight is None:
            sample_weight = np.ones(coeffs.shape[1])

        W_sqrt = np.sqrt(sample_weight)
        coeffs_weighted = coeffs * W_sqrt   # pondère les sorties

        N, D = coeffs.shape[1], coeffs.shape[0]
        nb_var = params.shape[1]

        self.scaler = StandardScaler(with_std=True, with_mean=True)
        params_scaled = self.scaler.fit_transform(params)

        interp_inputs = [params_scaled[:, i] for i in range(nb_var)]
        self.interpolators = []

        for k in range(D):
            rbf = Rbf(*interp_inputs, coeffs_weighted[k, :])
            self.interpolators.append(rbf)

        self.trained = True

    def predict(self, new_params):
        """
        Predict new coefficients using trained RBF interpolators.

        Args:
            new_params (np.ndarray): shape (M, nb_var), new parameter points

        Returns:
            pred_coeffs (np.ndarray): shape (M, D), predicted latent coefficients
        """
        if not self.trained:
            raise RuntimeError("Interpolator has not been trained. Call `fit()` first.")

        new_params_scaled = self.scaler.transform(new_params)
        nb_var = new_params.shape[1]
        pred_inputs = [new_params_scaled[:, i] for i in range(nb_var)]

        D = len(self.interpolators)
        M = new_params.shape[0]
        pred_coeffs = np.zeros((M, D))

        for k in range(D):
            pred_coeffs[:, k] = self.interpolators[k](*pred_inputs)

        return pred_coeffs
