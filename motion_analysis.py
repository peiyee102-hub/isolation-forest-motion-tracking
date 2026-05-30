"""
motion_analysis.py
==================
Core biomechanical feature extraction for elbow flexion/extension rehabilitation.
Input: dual-IMU quaternion data (qx1,qy1,qz1,qw1 = upper arm; qx2,qy2,qz2,qw2 = forearm)

Features extracted per rep:
  - Range of Motion (ROM) in degrees
  - SPARC smoothness (Spectral Arc Length) — more negative = smoother
  - Compensation score (upper arm movement proxy for trunk/shoulder sway)
  - Peak and mean angular velocity
  - Mean jerk (rate of velocity change)
  - Duration
  - Min / max angle (for start/end position assessment)
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.signal import find_peaks, butter, filtfilt
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class RepFeatures:
    rom: float                  # degrees
    min_angle: float            # degrees (extension limit)
    max_angle: float            # degrees (flexion limit)
    sparc: float                # spectral arc length (0 to -∞, less negative = smoother)
    comp_mean: float            # mean upper-arm angular motion (°/frame) — compensation
    comp_max: float             # max upper-arm angular motion — compensation spike
    upper_rom_x: float          # upper arm pitch ROM
    upper_rom_y: float          # upper arm yaw ROM  ← shoulder swing lives here
    upper_rom_z: float          # upper arm roll ROM ← trunk lean lives here
    peak_velocity: float        # °/s  — speed cheating shows high peak
    mean_velocity: float        # °/s
    mean_jerk: float            # °/s²  — tremor and speed cheating show high jerk
    duration_s: float           # rep duration
    n_samples: int              # raw sample count


@dataclass
class CalibrationResult:
    """Computed from patient's first N good reps."""
    target_rom: float
    target_sparc: float         # threshold (reps should be LESS negative than this)
    max_comp_mean: float        # compensation ceiling
    max_upper_rom_y: float      # shoulder swing ceiling
    max_jerk: float             # tremor / speed-cheat ceiling
    min_duration_s: float       # speed-cheat floor
    max_duration_s: float       # fatigue ceiling
    n_calib_reps: int
    raw_reps: List[RepFeatures] = field(default_factory=list)

    @property
    def summary(self) -> dict:
        return {
            "target_rom_deg": round(self.target_rom, 1),
            "target_sparc": round(self.target_sparc, 2),
            "max_comp_mean": round(self.max_comp_mean, 2),
            "max_upper_rom_y_deg": round(self.max_upper_rom_y, 1),
            "max_mean_jerk": round(self.max_jerk, 1),
            "duration_range_s": [round(self.min_duration_s, 1), round(self.max_duration_s, 1)],
            "calibrated_from_n_reps": self.n_calib_reps,
        }


@dataclass
class RepScore:
    """Outcome of judging a single rep."""
    passed: bool
    quality_score: float        # 0–1  (1 = perfect)
    faults: List[str]           # list of detected fault names
    features: RepFeatures
    details: dict               # per-dimension pass/fail + margin


# ──────────────────────────────────────────────
# Core geometry helpers
# ──────────────────────────────────────────────

def _quaternion_angle(row):
    """
    Calculates the elbow hinge angle using Euler projection.
    This replaces the arccos dot-product to reach the full 140-150° range.
    """
    try:
        # Define rotations (assuming [qx, qy, qz, qw] format)
        r1 = R.from_quat([row['qx1'], row['qy1'], row['qz1'], row['qw1']])
        r2 = R.from_quat([row['qx2'], row['qy2'], row['qz2'], row['qw2']])

        # Calculate relative rotation from upper arm to forearm
        rel_rot = r1.inv() * r2

        # 'zxy' isolates the primary flexion axis.
        euler = rel_rot.as_euler('zxy', degrees=True)

        return abs(euler[0])
    except Exception:
        return 0.0

def smooth_signal(x: np.ndarray) -> np.ndarray:
    if len(x) > 20:
        b, a = butter(2, 0.1)
        return filtfilt(b, a, x)
    return x

def compute_elbow_angles(df: pd.DataFrame) -> np.ndarray:
    """Return per-row elbow angle array (degrees)."""
    return np.array([_quaternion_angle(row) for _, row in df.iterrows()])


def compute_upper_arm_euler(df: pd.DataFrame) -> np.ndarray:
    """Return Nx3 array of upper-arm Euler angles (xyz, degrees) from IMU1."""
    eulers = []
    for _, row in df.iterrows():
        q1 = R.from_quat([row['qx1'], row['qy1'], row['qz1'], row['qw1']])
        eulers.append(q1.as_euler('xyz', degrees=True))
    return np.array(eulers)


def compute_upper_arm_angular_velocity(df: pd.DataFrame) -> np.ndarray:
    """
    Frame-to-frame angular velocity of upper arm (degrees) — gimbal-lock free.
    Uses quaternion geodesic distance instead of Euler differences.
    Returns N-1 values.
    """
    quats = df[['qx1','qy1','qz1','qw1']].values
    vel = []
    for i in range(1, len(quats)):
        q_prev = R.from_quat(quats[i-1])
        q_curr = R.from_quat(quats[i])
        q_diff = q_prev.inv() * q_curr
        angle = np.degrees(2.0 * np.arccos(np.clip(abs(q_diff.as_quat()[3]), 0, 1)))
        vel.append(angle)
    return np.array(vel)


def compute_upper_arm_total_rotation(df: pd.DataFrame) -> float:
    """
    Total rotation of upper arm from start to end of rep (degrees).
    Robust compensation measure — not affected by gimbal lock.
    """
    if len(df) < 2:
        return 0.0
    q_start = R.from_quat(df[['qx1','qy1','qz1','qw1']].iloc[0].values)
    q_end   = R.from_quat(df[['qx1','qy1','qz1','qw1']].iloc[-1].values)
    q_diff  = q_start.inv() * q_end
    return float(np.degrees(2.0 * np.arccos(np.clip(abs(q_diff.as_quat()[3]), 0, 1))))


# ──────────────────────────────────────────────
# Smoothness
# ──────────────────────────────────────────────

def sparc(movement: np.ndarray, sr: float,
          padlevel: int = 4, fc: float = 10.0, amp_th: float = 0.05) -> float:
    """
    Spectral Arc Length — measures movement smoothness.
    Returns 0 if signal too short or flat.
    More negative = more frequency content = less smooth.
    Typical good elbow flexion: ~ -20 to -80.
    """
    if len(movement) < 10:
        return 0.0
    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))
    Mow = np.abs(np.fft.fft(movement - movement.mean(), n=nfft) / sr)
    mx = Mow.max()
    if mx == 0:
        return 0.0
    Mow = Mow / mx
    fc_idx = int(fc * nfft / sr)
    Mow_sub = Mow[:fc_idx]
    mask = Mow_sub >= amp_th
    if mask.sum() < 2:
        return 0.0
    d = np.sqrt((1.0 / fc) ** 2 + np.diff(Mow_sub[mask]) ** 2)
    return float(-d.sum())


# ──────────────────────────────────────────────
# Rep segmentation
# ──────────────────────────────────────────────

def _sample_rate(df: pd.DataFrame) -> float:
    """Estimate sampling rate (Hz) from timestamp column (ms)."""
    dt_ms = df['Timestamp'].diff().median()
    return 1000.0 / dt_ms if dt_ms > 0 else 60.0


def segment_reps(df: pd.DataFrame,
                 prominence: float = 30.0,
                 min_distance_samples: int = 80,
                 min_samples: int = 100,
                 min_rom_deg: float = 50.0) -> List[Tuple[int, int]]:
    """
    Detect rep boundaries by finding valleys in the elbow angle signal.
    Returns list of (start, end) index pairs for each valid rep.

    Strategy:
      - Low-pass filter the angle to remove noise
      - Find valleys (arm near full extension = lower elbow angle)
      - Each valley-to-valley segment = one rep
    """
    angles = compute_elbow_angles(df)

    if len(angles) > 20:
        b, a = butter(2, 0.1)
        angles_smooth = filtfilt(b, a, angles)
    else:
        angles_smooth = angles

    valleys, _ = find_peaks(-angles_smooth,
                            prominence=prominence,
                            distance=min_distance_samples)

    boundaries = [0] + list(valleys) + [len(angles) - 1]

    reps = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        seg = angles[s:e]
        if len(seg) >= min_samples and (seg.max() - seg.min()) >= min_rom_deg:
            reps.append((s, e))

    return reps


# ──────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────

def extract_features(df: pd.DataFrame,
                     start: Optional[int] = None,
                     end: Optional[int] = None) -> RepFeatures:
    """
    Extract biomechanical features from one rep (or an entire recording).
    start/end are row indices into df; if None, uses entire df.
    """
    if start is not None and end is not None:
        seg_df = df.iloc[start:end].reset_index(drop=True)
    else:
        seg_df = df.reset_index(drop=True)

    sr = _sample_rate(seg_df)
    angles = compute_elbow_angles(seg_df)
    smooth_angles = smooth_signal(angles)

    upper_euler = compute_upper_arm_euler(seg_df)

    # Angular kinematics (forearm)
    # use smoothed signal only for derivatives
    ang_vel = np.abs(np.diff(smooth_angles)) * sr
    jerk = np.abs(np.diff(ang_vel)) * sr if len(ang_vel) > 1 else np.array([0.0])

    # Upper arm compensation — GIMBAL-LOCK-FREE quaternion-based metrics
    upper_angvel = compute_upper_arm_angular_velocity(seg_df)  # per-frame rotation
    upper_total  = compute_upper_arm_total_rotation(seg_df)    # start→end drift

    return RepFeatures(
        rom=float(angles.max() - angles.min()),
        min_angle=float(angles.min()),
        max_angle=float(angles.max()),
        sparc=sparc(angles, sr),
        comp_mean=float(upper_angvel.mean()) if len(upper_angvel) else 0.0,
        comp_max=float(upper_angvel.max()) if len(upper_angvel) else 0.0,
        upper_rom_x=float(np.ptp(upper_euler[:, 0])),
        upper_rom_y=upper_total,                         # total upper arm drift (compensation)
        upper_rom_z=float(np.ptp(upper_euler[:, 2])),   # kept for compatibility
        peak_velocity=float(ang_vel.max()) if len(ang_vel) else 0.0,
        mean_velocity=float(ang_vel.mean()) if len(ang_vel) else 0.0,
        mean_jerk=float(jerk.mean()),
        duration_s=float((seg_df['Timestamp'].iloc[-1] - seg_df['Timestamp'].iloc[0]) / 1000.0),
        n_samples=len(seg_df),
    )


def featurize_file(path: str, label: str = "unknown") -> List[dict]:
    """
    Load a CSV file, segment into reps, extract features.
    Returns list of dicts (label + all features) for downstream use.
    """
    df = pd.read_csv(path)
    reps = segment_reps(df)
    results = []
    for s, e in reps:
        f = extract_features(df, s, e)
        d = {"label": label, **f.__dict__}
        results.append(d)
    return results
