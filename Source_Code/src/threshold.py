"""
src/threshold.py
Adaptive Gaussian Kernel Density Estimation (KDE) Anomaly Threshold Scorer.

Replaces static top-k heuristics with a continuous non-parametric density
boundary. The KDE is fitted on session-level negative log-likelihood (NLL)
losses computed over normal validation sequences. The anomaly threshold tau
is derived by numerically integrating the fitted KDE's probability density
function to find the (1 - alpha) quantile of its CDF.

This means the KDE itself controls the decision boundary — not a raw
NumPy percentile on the input array.
"""

import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid
from typing import Tuple
import pickle


class AdaptiveKDEScorer:
    """
    Fits a Gaussian Kernel Density Estimator over the distribution of
    session-level negative log-likelihoods (NLL, specifically max per-event NLL)
    observed on normal validation data.

    The anomaly threshold tau is the value where the KDE's own CDF
    reaches (1 - alpha). Any test session whose NLL statistic exceeds tau
    is classified as anomalous.

    Mathematical formulation:
        Given normal session NLLs L = {l_1, ..., l_N}, the density is:
            f_hat(x) = (1 / Nh) * sum_i K((x - l_i) / h)
        where K is the Gaussian kernel and h is the bandwidth
        (auto-selected via Scott's rule by default).

        The threshold tau satisfies:
            integral_{-inf}^{tau} f_hat(u) du = 1 - alpha
    """

    def __init__(self, alpha: float = 0.02, bandwidth: str = "scott"):
        """
        Args:
            alpha:     significance level (target false positive rate).
                       Default 0.02 means ~2% of normal sessions may be
                       flagged as anomalous.
            bandwidth: KDE bandwidth selection rule ('scott', 'silverman',
                       or a float for manual override).
        """
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.kde = None
        self.threshold = None
        self.fitted_losses = None
        self._grid = None
        self._cdf = None

    def fit(self, session_nll_losses: np.ndarray) -> "AdaptiveKDEScorer":
        """
        Fits KDE on an array of session-level NLL losses (typically session
        max per-event NLL) and derives the anomaly threshold tau from the
        KDE's own CDF.

        Args:
            session_nll_losses: 1-D array of session NLL values (e.g. max-event NLL),
                                one per normal validation session.
        Returns:
            self (fitted scorer).
        """
        losses = np.array(session_nll_losses, dtype=np.float64)
        if losses.size < 5:
            raise ValueError(
                f"Need at least 5 session losses to fit KDE, got {losses.size}."
            )

        self.fitted_losses = losses
        self.kde = gaussian_kde(losses, bw_method=self.bandwidth)

        # Build a fine evaluation grid that extends beyond the observed
        # range so the tails of the density are captured.
        margin = 3.0 * self.kde.scotts_factor() * losses.std()
        lo = losses.min() - margin
        hi = losses.max() + margin
        self._grid = np.linspace(lo, hi, 20_000)

        # Evaluate the fitted KDE's PDF on this grid
        pdf_values = self.kde.evaluate(self._grid)

        # Numerically integrate the PDF to get the CDF
        cdf_raw = cumulative_trapezoid(pdf_values, self._grid, initial=0.0)
        # Normalize so it reaches 1.0 at the right boundary
        total_area = cdf_raw[-1]
        if total_area > 0:
            self._cdf = cdf_raw / total_area
        else:
            # Degenerate case: fall back to empirical percentile
            self._cdf = np.linspace(0, 1, len(self._grid))

        # Threshold tau: smallest grid point where CDF >= (1 - alpha)
        target_quantile = 1.0 - self.alpha
        idx = np.searchsorted(self._cdf, target_quantile)
        idx = min(idx, len(self._grid) - 1)
        self.threshold = float(self._grid[idx])

        return self

    def score(self, session_nll: float) -> Tuple[bool, float, float]:
        """
        Evaluates a single session's NLL statistic against the fitted threshold.

        Args:
            session_nll: the session-level NLL statistic (typically the max
                         per-event NLL across all transitions).
        Returns:
            is_anomaly:   True if session_nll > tau.
            severity:     normalized severity score in [0, 1].
            density:      the KDE-estimated probability density at this NLL.
        """
        if self.threshold is None or self.kde is None:
            raise RuntimeError("AdaptiveKDEScorer must be fitted before scoring.")

        nll = float(session_nll)
        density = float(self.kde.evaluate([nll])[0])

        is_anomaly = nll > self.threshold

        # Severity: how far above the threshold the NLL sits, normalized
        # against the observed range.
        if is_anomaly:
            overshoot = nll - self.threshold
            # Scale relative to the spread of normal losses
            spread = float(self.fitted_losses.std()) + 1e-7
            severity = float(np.clip(overshoot / (2.0 * spread), 0.0, 1.0))
        else:
            severity = 0.0

        return is_anomaly, severity, density

    def score_event(self, event_nll: float) -> Tuple[bool, float, float]:
        """
        Convenience method: scores a single transition's NLL.
        Used by the live API for per-event granularity while the session-
        level decision remains the authoritative anomaly signal.

        Same interface as score() but operates on a single event NLL
        rather than a session aggregate.
        """
        return self.score(event_nll)

    def save(self, file_path: str):
        """Serialize the fitted scorer to disk."""
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str) -> "AdaptiveKDEScorer":
        """Deserialize a previously fitted scorer."""
        with open(file_path, "rb") as f:
            return pickle.load(f)
