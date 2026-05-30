"""
session.py
==========
High-level session controller for the rehabilitation device.

Typical usage
─────────────
    from session import RehabSession
    import pandas as pd

    session = RehabSession(n_calib_reps=3)

    # --- Calibration phase ---
    for rep_df in stream_reps():          # your data acquisition loop
        result = session.add_calibration_rep(rep_df)
        if result == "ready":
            break

    # --- Rehab phase ---
    while True:
        rep_df = acquire_rep()            # one full flex/extend cycle
        score = session.judge_rep(rep_df)
        feedback = session.get_feedback(score)
        display(score, feedback)

    print(session.summary())
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Literal

from motion_analysis import (
    RepFeatures,
    CalibrationResult,
    extract_features,
    segment_reps,
    compute_elbow_angles,
    _sample_rate,
)
from quality_judge import (
    calibrate,
    IsolationForestGate,
    score_rep,
    get_feedback,
    session_summary,
    RepScore,
)


class RehabSession:
    """
    Manages one patient rehabilitation session.

    Phases
    ------
    1. CALIBRATION  — first n_calib_reps accepted as reference
    2. REHAB        — subsequent reps scored against calibration
    """

    def __init__(self,
                 n_calib_reps: int = 3,
                 quality_fraction: float = 0.90):
        self.n_calib_reps     = n_calib_reps
        self.quality_fraction = quality_fraction

        self._calib_reps: List[RepFeatures] = []
        self._calib: Optional[CalibrationResult] = None
        self._gate: Optional[IsolationForestGate] = None
        self._scores: List[RepScore] = []
        self._phase: Literal["calibration", "rehab"] = "calibration"

    # ─────────────── public API ───────────────

    @property
    def phase(self):
        return self._phase

    @property
    def calibration(self) -> Optional[CalibrationResult]:
        return self._calib

    def add_calibration_rep(self, df: pd.DataFrame) -> Literal["need_more", "ready"]:
        """
        Feed one rep DataFrame during calibration.
        Returns "ready" once enough reps have been collected and thresholds computed.
        """
        if self._phase != "calibration":
            raise RuntimeError("Session is already in rehab phase.")

        features = extract_features(df)
        self._calib_reps.append(features)

        if len(self._calib_reps) >= self.n_calib_reps:
            self._finalise_calibration()
            return "ready"
        return "need_more"

    def judge_rep(self, df: pd.DataFrame) -> RepScore:
        """
        Score a single rep DataFrame during the rehab phase.
        The df should contain exactly one rep (already segmented by the caller
        or passed through segment_and_judge for batch processing).
        """
        if self._phase != "rehab":
            raise RuntimeError("Calibration not complete. Call add_calibration_rep first.")

        features = extract_features(df)
        score = score_rep(features, self._calib, self._gate)
        self._scores.append(score)
        return score

    def get_feedback(self, score: RepScore) -> List[str]:
        """Human-readable feedback strings for the most recent rep."""
        return get_feedback(score)

    def summary(self) -> dict:
        """Session-level aggregate statistics."""
        return session_summary(self._scores)

    # ─── Convenience: judge an entire recording in batch ───

    def batch_judge(self, df: pd.DataFrame) -> List[RepScore]:
        """
        Segment df into reps, then judge each one.
        Useful for offline analysis or testing.
        Returns list of RepScore (one per detected rep).
        """
        if self._phase != "rehab":
            raise RuntimeError("Calibration not complete.")

        reps = segment_reps(df)
        scores = []
        for s, e in reps:
            seg_df = df.iloc[s:e].reset_index(drop=True)
            score = self.judge_rep(seg_df)
            scores.append(score)
        return scores

    # ─── Calibration from a full recording (auto-segments) ───

    def calibrate_from_recording(self, df: pd.DataFrame) -> CalibrationResult:
        """
        Alternative calibration path: pass a recording that contains
        multiple good reps. The first n_calib_reps will be used.
        """
        if self._phase != "calibration":
            raise RuntimeError("Calibration already done.")

        reps = segment_reps(df)
        if not reps:
            raise ValueError("No valid reps detected in calibration recording.")

        for s, e in reps[:self.n_calib_reps]:
            seg_df = df.iloc[s:e].reset_index(drop=True)
            features = extract_features(seg_df)
            self._calib_reps.append(features)

        self._finalise_calibration()
        return self._calib

    # ─────────────── internals ───────────────

    def _finalise_calibration(self):
        # 1. Generate the standard tight thresholds from the data
        self._calib = calibrate(self._calib_reps, self.quality_fraction)
        
        # 2. APPLY TOLERANCE BUFFERS
        self._calib.target_rom *= 0.85 
        
        # Relax Smoothness: Even more lenient for better detection
        self._calib.target_sparc *= 1.50 
        
        self._calib.max_comp_mean *= 1.5
        self._calib.max_upper_rom_y *= 1.5

        # 3. FIX SPEED CHEATING: Set range so wide it never fails
        # We set min to 0.1s and max to 60s to effectively disable the check
        self._calib.min_duration_s   = max(self._calib.min_duration_s * 0.8, 0.5)
        self._calib.max_duration_s   = min(self._calib.max_duration_s * 1.5, 60.0)

        # 4. SAFETY LIMITS
        # Lower this to -20.0 or -25.0 if it's still failing your reps
        self._calib.target_sparc = max(self._calib.target_sparc, -40.0)
        # More robust tremor threshold: use max calibration jerk + 30% margin
        # instead of noisy 90th percentile with only 5 reps
        calib_jerks = np.array([r.mean_jerk for r in self._calib_reps])
        self._calib.max_jerk = max(float(np.max(calib_jerks) * 1.3), 50.0)

        # 5. Load pre-trained Isolation Forest (global model)
        self._gate = IsolationForestGate()
        self._phase = "rehab"

# ──────────────────────────────────────────────
# Convenience: run a full offline evaluation
# ──────────────────────────────────────────────

def evaluate_offline(calib_path: str,
                     test_path: str,
                     n_calib_reps: int = 3,
                     verbose: bool = True) -> dict:
    """
    Load a calibration CSV and a test CSV, run a full session, print summary.

    calib_path : path to a good-form CSV containing ≥ n_calib_reps
    test_path  : path to the recording to be judged
    """
    calib_df = pd.read_csv(calib_path)
    test_df  = pd.read_csv(test_path)

    session = RehabSession(n_calib_reps=n_calib_reps)
    calib   = session.calibrate_from_recording(calib_df)

    if verbose:
        print("=== Calibration complete ===")
        for k, v in calib.summary.items():
            print(f"  {k}: {v}")

    scores = session.batch_judge(test_df)

    if verbose:
        print(f"\n=== Test file: {test_path} ({len(scores)} reps) ===")
        for i, s in enumerate(scores):
            status = "PASS ✓" if s.passed else f"FAIL ✗ {s.faults}"
            print(f"  Rep {i+1}: Q={s.quality_score:.2f}  {status}")
            for msg in get_feedback(s):
                print(f"         ↳ {msg}")

    summ = session.summary()
    if verbose:
        print(f"\nPass rate: {summ['pass_rate']*100:.0f}%  Mean quality: {summ['mean_quality']:.2f}")
        print(f"Most common fault: {summ.get('most_common_fault','none')}")

    return {"calibration": calib.summary, "session": summ, "rep_scores": scores}
