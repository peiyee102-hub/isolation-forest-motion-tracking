"""
quality_judge.py
================
Two-layer movement quality assessment for elbow flexion/extension rehab.

Layer 1 — Rule engine (interpretable, clinician-friendly):
  Uses patient-specific thresholds derived from calibration reps.
  Each dimension maps to a named fault for real-time feedback.

Layer 2 — Statistical anomaly gate (optional, additive):
  Mahalanobis distance from the calibration distribution.
  Flags reps that "feel wrong" in a combination of features
  even when no single rule is violated.

The architecture deliberately avoids Random Forest on this dataset because:
  - Only 20 labelled good reps from one patient (one session)
  - Bad CSVs represent entire sessions, not individual reps — 52 segmented bad reps
  - After per-rep segmentation SPARC, comp and velocity overlap heavily with good reps
  - A supervised classifier would memorise noise, not generalise to new patients
  - Patient-specific calibration captures individual baseline variation naturally
  - Rules are explainable to clinicians and can drive real-time cues
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from motion_analysis import RepFeatures, CalibrationResult, RepScore

# Try to import joblib for Isolation Forest; if missing, gate stays inactive
try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


# ──────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────

CALIB_QUALITY_FRACTION = 0.90   # target = 90% of calibration quality

# Isolation Forest score normalization (from training_summary.json)
# decision_function range: min=-0.1664  max=0.1803  threshold=0.0197
IF_SCORE_MIN = -0.1664
IF_SCORE_MAX = 0.1803


def calibrate(reps: List[RepFeatures],
              quality_fraction: float = CALIB_QUALITY_FRACTION) -> CalibrationResult:
    """
    Build patient-specific quality thresholds from the first N good reps.

    Threshold strategy:
      - ROM target        = quality_fraction × median ROM
      - SPARC target      = median SPARC (less negative reps are flagged)
      - Compensation      = median + 1 SD upper bound (lenient, motor-impaired patients move more)
      - Shoulder swing    = median + 1 SD of upper_rom_y
      - Jerk              = 90th percentile (catches tremor / speed-cheat outliers)
      - Duration          = [10th, 90th] percentile (catches speed-cheat and fatigue)
    """
    if not reps:
        raise ValueError("Need at least 1 calibration rep.")

    rom_vals       = np.array([r.rom           for r in reps])
    sparc_vals     = np.array([r.sparc         for r in reps])
    comp_vals      = np.array([r.comp_mean     for r in reps])
    ury_vals       = np.array([r.upper_rom_y   for r in reps])
    jerk_vals      = np.array([r.mean_jerk     for r in reps])
    dur_vals       = np.array([r.duration_s    for r in reps])

    return CalibrationResult(
        target_rom      = float(np.median(rom_vals)   * quality_fraction),
        target_sparc    = float(np.median(sparc_vals)),      # threshold: must be ≤ this (more negative)
        max_comp_mean   = float(np.median(comp_vals)  + np.std(comp_vals)),
        max_upper_rom_y = float(np.median(ury_vals)   + np.std(ury_vals)),
        max_jerk        = float(np.percentile(jerk_vals, 75)),
        min_duration_s  = float(np.percentile(dur_vals, 25)),
        max_duration_s  = float(np.percentile(dur_vals, 75) * 2.0),
        n_calib_reps    = len(reps),
        raw_reps        = reps,
    )


# ──────────────────────────────────────────────
# Isolation Forest anomaly gate (Layer 2)
# ──────────────────────────────────────────────

class IsolationForestGate:
    """
    Loads a pre-trained Isolation Forest model.
    Scores reps with decision_function; lower = more anomalous.
    """
    FEATURES = ['sparc', 'mean_jerk', 'peak_velocity', 'mean_velocity',
                'duration_s']
    # comp_norm and swing_norm computed on the fly
    DEFAULT_THRESHOLD = 0.0197   # 5th percentile of normal training scores

    def __init__(self, model_path: Optional[str] = None, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._model = None
        self._fitted = False

        if not _HAS_JOBLIB:
            return

        if model_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base, "models", "isolation_forest.joblib")

        if os.path.exists(model_path):
            try:
                self._model = joblib.load(model_path)
                self._fitted = True
            except Exception:
                self._model = None
                self._fitted = False

    def _to_vec(self, rep: RepFeatures) -> np.ndarray:
        rom = rep.rom if rep.rom > 0 else 1.0
        comp_norm = rep.comp_mean / rom
        swing_norm = rep.upper_rom_y / rom
        return np.array([
            rep.sparc,
            rep.mean_jerk,
            rep.peak_velocity,
            rep.mean_velocity,
            rep.duration_s,
            comp_norm,
            swing_norm,
        ]).reshape(1, -1)

    def score(self, rep: RepFeatures) -> float:
        """Return Isolation Forest decision_function. Higher = more normal."""
        if not self._fitted:
            return 1.0   # inactive gate = assume normal
        vec = self._to_vec(rep)
        return float(self._model.decision_function(vec)[0])

    def is_anomaly(self, rep: RepFeatures) -> bool:
        return self._fitted and self.score(rep) < self.threshold


# ──────────────────────────────────────────────
# Mahalanobis anomaly gate (legacy, kept for compatibility)
# ──────────────────────────────────────────────

class MahalanobisGate:
    """
    Fits a multivariate Gaussian to calibration reps.
    Flags reps with Mahalanobis distance > threshold as anomalous.
    Uses regularised covariance to handle small N.
    """
    FEATURES = ['rom', 'sparc', 'comp_mean', 'upper_rom_y', 'mean_jerk', 'duration_s']
    DEFAULT_THRESHOLD = 3.0   # chi² 99.9% for 6 DOF ≈ 22, but we use empirical z-score style

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.mean_: Optional[np.ndarray] = None
        self.inv_cov_: Optional[np.ndarray] = None
        self._fitted = False

    def _to_vec(self, rep: RepFeatures) -> np.ndarray:
        return np.array([getattr(rep, f) for f in self.FEATURES])

    def fit(self, reps: List[RepFeatures]):
        if len(reps) < 3:
            return   # not enough data to fit — gate stays inactive
        X = np.stack([self._to_vec(r) for r in reps])
        self.mean_ = X.mean(axis=0)
        cov = np.cov(X.T)
        # Regularise: shrink toward diagonal to avoid singular matrix with small N
        cov = cov + 1e-4 * np.eye(len(self.FEATURES))
        self.inv_cov_ = np.linalg.inv(cov)
        self._fitted = True

    def score(self, rep: RepFeatures) -> float:
        """Return Mahalanobis distance. Returns 0 if not fitted."""
        if not self._fitted:
            return 0.0
        diff = self._to_vec(rep) - self.mean_
        d2 = diff @ self.inv_cov_ @ diff
        return float(np.sqrt(max(d2, 0.0)))

    def is_anomaly(self, rep: RepFeatures) -> bool:
        return self._fitted and self.score(rep) > self.threshold


# ──────────────────────────────────────────────
# Rule engine
# ──────────────────────────────────────────────

# Each rule: (fault_name, test_fn, feedback_message)
# test_fn(features, calib) -> True means fault DETECTED
_RULES: List[Tuple[str, callable, str]] = [
    (
        "insufficient_rom",
        lambda f, c: f.rom < c.target_rom,
        "Range of motion too limited — try to straighten and bend your arm fully."
    ),
    (
        "shoulder_swing",
        lambda f, c: f.upper_rom_y > c.max_upper_rom_y,
        "Shoulder is swinging — keep your upper arm still and close to your body."
    ),
    (
        "trunk_lean",
        lambda f, c: f.upper_rom_y > (c.max_upper_rom_y * 2.5),
        "Trunk is leaning — sit upright and avoid rocking your body."
    ),
    (
        "compensation",
        lambda f, c: f.comp_mean > c.max_comp_mean,
        "Upper arm is compensating — isolate the movement to your elbow."
    ),
    (
        "tremor",
        lambda f, c: f.mean_jerk > c.max_jerk,
        "Movement is unsteady — try to move slowly and smoothly."
    ),
    (
        "speed_cheating",
        lambda f, c: (f.duration_s < c.min_duration_s * 0.92),
        "Rep is too fast — slow down for better muscle engagement."
    ),
    (
        "unsmooth",
        lambda f, c: (f.sparc > (c.target_sparc * 0.5) and f.mean_jerk > c.max_jerk),
        "Movement is jerky — aim for a smooth, controlled arc."
    ),
]


# ──────────────────────────────────────────────
# Quality scorer
# ──────────────────────────────────────────────

def _dimension_score(value: float, good_value: float,
                     direction: str = "higher_is_better",
                     tolerance: float = 0.15) -> float:
    """
    Returns 0–1 score for a single feature relative to the calibration value.
    direction: "higher_is_better" (ROM) | "lower_is_better" (jerk, comp)
    """
    if good_value == 0:
        return 1.0
    ratio = value / good_value
    if direction == "higher_is_better":
        return float(np.clip((ratio - (1 - tolerance)) / tolerance, 0, 1))
    else:  # lower_is_better
        return float(np.clip(((1 + tolerance) - ratio) / tolerance, 0, 1))


def score_rep(features: RepFeatures,
              calib: CalibrationResult,
              gate: Optional[IsolationForestGate] = None) -> RepScore:
    """
    Judge a single rep against calibration thresholds.

    Returns RepScore with:
      - passed (bool)
      - quality_score 0–1
      - list of fault names
      - per-dimension details
    """
    faults = []
    details = {}

    # Run rule engine
    for fault_name, test_fn, _ in _RULES:
        triggered = test_fn(features, calib)
        details[fault_name] = not triggered
        if triggered:
            faults.append(fault_name)

    # Isolation Forest gate — proportional weighted dimension (NOT a flat penalty)
    anomaly_norm = 1.0   # default = perfectly normal if gate inactive
    if gate is not None:
        if_score = gate.score(features)
        details['anomaly_score'] = round(if_score, 3)
        # Normalize decision_function to 0–1 where 1 = perfectly normal
        raw_range = IF_SCORE_MAX - IF_SCORE_MIN
        if raw_range > 0:
            anomaly_norm = float(np.clip((if_score - IF_SCORE_MIN) / raw_range, 0.0, 1.0))
        details['anomaly_norm'] = round(anomaly_norm, 3)
    else:
        details['anomaly_norm'] = 1.0

    # Composite quality score (weighted average of key dimensions)
    # Anomaly is now a proportional dimension, not a flat penalty
    scores = {
        "rom":       _dimension_score(features.rom, calib.target_rom,       "higher_is_better"),
        "smoothness":_dimension_score(-features.sparc, -calib.target_sparc, "higher_is_better"),
        "comp":      _dimension_score(features.comp_mean, calib.max_comp_mean, "lower_is_better"),
        "jerk":      _dimension_score(features.mean_jerk, calib.max_jerk,   "lower_is_better"),
        "swing":     _dimension_score(features.upper_rom_y, calib.max_upper_rom_y, "lower_is_better"),
        "anomaly":   anomaly_norm,   # proportional: higher = more normal
    }
    weights = {"rom": 0.30, "smoothness": 0.25, "comp": 0.15, "jerk": 0.15, "swing": 0.10, "anomaly": 0.10}
    quality = sum(scores[k] * weights[k] for k in weights)
    quality = float(np.clip(quality, 0.0, 1.0))

    details.update({f"score_{k}": round(v, 3) for k, v in scores.items()})
    details['quality_score'] = round(quality, 3)

    # 3-tier status
    has_faults = len(faults) > 0
    is_anomaly = gate is not None and gate.is_anomaly(features)

    if has_faults:
        status = "FAIL"
    elif is_anomaly:
        status = "WARN"
    else:
        status = "PASS"

    # Backward compatibility: passed=True for both PASS and WARN
    passed = not has_faults

    return RepScore(
        passed=passed,
        quality_score=quality,
        faults=faults,
        features=features,
        details=details,
        status=status,
    )


def get_feedback(score: RepScore) -> List[str]:
    """Return human-readable feedback strings for detected faults and warnings."""
    messages = []
    fault_map = {name: msg for name, _, msg in _RULES}
    for fault in score.faults:
        msg = fault_map.get(fault)
        if msg:
            messages.append(msg)
    # WARN tier: no rule faults, but anomaly gate flagged unusual pattern
    if score.status == "WARN":
        messages.append(
            "Movement pattern looks unusual — your rep met all measurable thresholds, "
            "but check your form and position."
        )
    return messages


# ──────────────────────────────────────────────
# Session-level summary
# ──────────────────────────────────────────────

def session_summary(scores: List[RepScore]) -> dict:
    if not scores:
        return {}
    qualities = [s.quality_score for s in scores]
    fault_counter: Dict[str, int] = {}
    for s in scores:
        for f in s.faults:
            fault_counter[f] = fault_counter.get(f, 0) + 1

    total = len(scores)
    n_pass = sum(1 for s in scores if s.status == "PASS")
    n_warn = sum(1 for s in scores if s.status == "WARN")
    n_fail = sum(1 for s in scores if s.status == "FAIL")

    return {
        "total_reps": total,
        "passed_reps": n_pass + n_warn,   # backward-compatible: completed reps
        "pass_rate": round((n_pass + n_warn) / total, 2) if total else 0.0,
        "warned_reps": n_warn,
        "warn_rate": round(n_warn / total, 2) if total else 0.0,
        "failed_reps": n_fail,
        "fail_rate": round(n_fail / total, 2) if total else 0.0,
        "mean_quality": round(float(np.mean(qualities)), 3),
        "min_quality":  round(float(np.min(qualities)), 3),
        "max_quality":  round(float(np.max(qualities)), 3),
        "fault_counts": fault_counter,
        "most_common_fault": max(fault_counter, key=fault_counter.get) if fault_counter else None,
    }
