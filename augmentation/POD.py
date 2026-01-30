import numpy as np
from typing import Optional


class POD:
    """
    Compute a POD basis from a snapshot matrix and provide projection utilities.

    Attributes
    ----------
    basis : np.ndarray
        Matrix whose columns are the retained POD modes.
    singular_values : np.ndarray
        Singular values associated with the retained modes.
    retained_energy : float
        Fraction of energy preserved by the retained modes.
    projection_matrix : np.ndarray
        Matrix projecting a state vector onto the retained POD coefficients.
    mean_snapshot : Optional[np.ndarray]
        Mean snapshot if centering is enabled.
    """

    def __init__(self, center: bool = False):
        self.center = center
        self.basis: Optional[np.ndarray] = None
        self.singular_values: Optional[np.ndarray] = None
        self.retained_energy: Optional[float] = None
        self.projection_matrix: Optional[np.ndarray] = None
        self.mean_snapshot: Optional[np.ndarray] = None

    def compute(
        self,
        snapshots: np.ndarray,
        n_modes: Optional[int] = None,
        energy_ratio: Optional[float] = None,
        weights = None
    ) -> "POD":
        """
        Compute the POD basis from a snapshot matrix.

        Parameters
        ----------
        snapshots : np.ndarray
            Matrix of shape (n_state, n_snapshots) containing the states.
        n_modes : Optional[int]
            Number of modes to retain.
        energy_ratio : Optional[float]
            Energy ratio in (0, 1] used to choose the number of modes.

        Returns
        -------
        POD
            The current instance for chaining.
        """
        if snapshots.ndim != 2:
            raise ValueError("snapshots must be a 2D array (n_state, n_snapshots)")

        if n_modes is not None and energy_ratio is not None:
            raise ValueError("Specify either n_modes or energy_ratio, not both.")

        if energy_ratio is not None and not (0.0 < energy_ratio <= 1.0):
            raise ValueError("energy_ratio must be in (0, 1].")

        snapshots = np.asarray(snapshots, dtype=float)

        if self.center:
            self.mean_snapshot = np.mean(snapshots, axis=1, keepdims=True)
            snapshots_centered = snapshots - self.mean_snapshot
        else:
            self.mean_snapshot = None
            snapshots_centered = snapshots

        # --- WEIGHTS ---
        if weights is None:
            weights = np.ones(snapshots.shape[1])
        else:
            weights = np.asarray(weights, dtype=float)
            if weights.size != snapshots.shape[1]:
                raise ValueError("weights must have shape (n_snapshots,)")

        W_sqrt = np.sqrt(weights)[np.newaxis, :]
        snapshots_centered = snapshots_centered * W_sqrt


        U, S, _ = np.linalg.svd(snapshots_centered, full_matrices=False)
        energy = S**2
        cumulative_energy = np.cumsum(energy)
        total_energy = cumulative_energy[-1] if energy.size else 0.0

        if energy_ratio is not None:
            if total_energy == 0.0:
                raise ValueError("Cannot determine n_modes from energy ratio; total energy is zero.")
            # smallest r such that cumulative energy ratio >= target
            threshold = energy_ratio * total_energy
            n_modes = int(np.searchsorted(cumulative_energy, threshold) + 1)

        if n_modes is None:
            n_modes = S.size

        if not (1 <= n_modes <= S.size):
            raise ValueError("n_modes must satisfy 1 <= n_modes <= min(n_state, n_snapshots).")

        self.basis = U[:, :n_modes]
        self.singular_values = S[:n_modes]
        if total_energy > 0.0:
            self.retained_energy = cumulative_energy[n_modes - 1] / total_energy
        else:
            self.retained_energy = 0.0
        self.projection_matrix = self.basis.T
        return self

    def project(self, snapshots: np.ndarray) -> np.ndarray:
        """
        Project snapshots onto the retained POD subspace.

        Parameters
        ----------
        snapshots : np.ndarray
            State vector(s) with shape (n_state,) or (n_state, n_snapshots).

        Returns
        -------
        np.ndarray
            POD coefficients with shape (n_modes,) or (n_modes, n_snapshots).
        """
        if self.projection_matrix is None:
            raise RuntimeError("POD basis not computed. Call compute() first.")

        snapshots = np.asarray(snapshots, dtype=float)
        if snapshots.ndim == 1:
            snapshots = snapshots[:, np.newaxis]
            squeeze = True
        elif snapshots.ndim == 2:
            squeeze = False
        else:
            raise ValueError("snapshots must be 1D or 2D array.")

        if self.center and self.mean_snapshot is not None:
            centered = snapshots - self.mean_snapshot
        else:
            centered = snapshots

        coeffs = self.projection_matrix @ centered
        if squeeze:
            return coeffs[:, 0]
        return coeffs

    def reconstruct(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Reconstruct state vectors from POD coefficients.

        Parameters
        ----------
        coeffs : np.ndarray
            POD coefficients with shape (n_modes,) or (n_modes, n_snapshots),
            or their transpose (n_snapshots, n_modes).

        Returns
        -------
        np.ndarray
            Reconstructed state(s) with shape (n_state,) or (n_state, n_snapshots).
        """
        if self.basis is None:
            raise RuntimeError("POD basis not computed. Call compute() first.")

        coeffs = np.asarray(coeffs, dtype=float)
        transposed = False
        if coeffs.ndim == 1:
            coeffs = coeffs[:, np.newaxis]
            squeeze = True
        elif coeffs.ndim == 2:
            squeeze = False
            if coeffs.shape[0] != self.basis.shape[1] and coeffs.shape[1] == self.basis.shape[1]:
                coeffs = coeffs.T
                transposed = True
        else:
            raise ValueError("coeffs must be 1D or 2D array.")

        states = self.basis @ coeffs
        if self.center and self.mean_snapshot is not None:
            states = states + self.mean_snapshot

        if squeeze:
            return states[:, 0]
        if transposed:
            return states.T
        return states
