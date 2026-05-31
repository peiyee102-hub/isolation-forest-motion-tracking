"""
live_main.py  —  Real-time elbow rehab judge
=============================================

HOW TO RUN
----------
1. All files in the same folder:
       live_main.py, motion_analysis.py, quality_judge.py,
       session.py, data_logger.py

2. Install once:
       pip install pyserial pandas numpy scipy scikit-learn

3. Edit CONFIG below, then run:
       python live_main.py

SERIAL FORMAT (Arduino/ESP32 must output this):
    qx1,qy1,qz1,qw1,qx2,qy2,qz2,qw2
  IMU1 = UPPER ARM,  IMU2 = FOREARM  (scalar-last quaternion)

FIXES vs original
-----------------
BUG 1 — Duration mismatch (main false-positive source)
  Calibration used segment_from_window() which includes ~0.5s settle pads
  on each side of the movement, inflating duration_s.  The rehab detector
  (valley→valley) measured only the motion itself — same real rep, but
  ~2–3s shorter, instantly triggering speed_cheating on every good rep.
  FIX: calibration reps now go through the SAME valley→valley detector
  (segment_rehab_rep) that is used during live scoring, so thresholds and
  scores are measured on the same scale.

BUG 2 — Buffer too short for long reps
  WINDOW_SEC=10s means a 6-8s rep barely fits; buffer edges clip valleys,
  causing mis-detection or no detection.
  FIX: buffer extended to 20s and a committed-rep tracker prevents double-
  counting.

BUG 3 — Cooldown blocked back-to-back reps
  A 1.5s cooldown after any detection prevented scoring a new rep if the
  patient performed them continuously without a pause.
  FIX: cooldown is now sample-index based (tracks last rep's end index in
  the buffer), not wall-clock time.  A rep must start at least
  COOLDOWN_SAMPLES after the previous rep ended.
"""

import sys, os, time
import numpy as np
import pandas as pd
from collections import deque
from scipy.signal import find_peaks, butter, filtfilt
from scipy.spatial.transform import Rotation as R  # <--- ADD THIS LINE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_analysis import compute_elbow_angles, extract_features, _quaternion_angle
from session import RehabSession
from data_logger import DataLogger
from report_generation import generate_rehab_report

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
SERIAL_PORT      = "COM9"

BAUD_RATE        = 115200
SAMPLE_RATE      = 20.0   # <-- set to match your Arduino firmware (20 or 60)

PATIENT_ID       = "P001"
N_CALIB_REPS     = 5
CALIB_DURATION   = 12.0       # seconds per calibration window (give extra room)

MIN_ROM          = 30.0       # degrees — lower = more lenient for limited ROM patients
MIN_DURATION     = 1.5        # seconds — absolute floor (very impaired patients)

# ── Rolling buffer: large enough that even slow reps (10s+) fit comfortably ──
WINDOW_SEC       = 20.0
WINDOW_FRAMES    = int(WINDOW_SEC * SAMPLE_RATE)

# ── Rep detection ──
SMOOTH_CUTOFF    = 0.08
VALLEY_PROMINENCE = 5.0      # raised slightly to avoid noise valleys
VALLEY_DISTANCE  = 20

# ── Cooldown: ignore new reps that start within N samples of last rep end ──
COOLDOWN_SAMPLES = 5
DETECTION_INTERVAL = 3  # Runs rep-detection every 0.25s instead of every sample
BAR_WIDTH        = 30

# ══════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════
def time_bar(elapsed, total, width=BAR_WIDTH):
    filled = int(min(elapsed / total, 1.0) * width)
    return f"[{'█'*filled}{'░'*(width-filled)}] {max(total-elapsed,0):4.1f}s left"

def angle_bar(angle, min_a, max_a, width=BAR_WIDTH):
    span = max(max_a - min_a, 1.0)
    pos  = int(np.clip((angle - min_a) / span, 0, 1) * width)
    bar  = ['░'] * width
    for i in range(pos): bar[i] = '█'
    if 0 <= pos < width: bar[pos] = '▌'
    return f"[{''.join(bar)}] {angle:5.1f}°"

def quality_bar(q, width=BAR_WIDTH):
    filled = int(q * width)
    c = "\033[92m" if q >= 0.8 else ("\033[93m" if q >= 0.5 else "\033[91m")
    return f"{c}[{'█'*filled}{'░'*(width-filled)}] {q*100:.0f}%\033[0m"

# ══════════════════════════════════════════════════════════════
#  SERIAL HELPERS
# ══════════════════════════════════════════════════════════════
def read_one_raw(ser):
    """Read one line, return (vals_list, raw_angle) or None."""
    try:
        raw = ser.readline().decode("utf-8", errors="ignore").strip()
        if not raw: return None
        vals = [float(x) for x in raw.split(",")]
        if len(vals) != 8: return None
        row = {'qx1':vals[0],'qy1':vals[1],'qz1':vals[2],'qw1':vals[3],
               'qx2':vals[4],'qy2':vals[5],'qz2':vals[6],'qw2':vals[7]}
        return vals, _quaternion_angle(row)
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════
#  SHARED SIGNAL HELPER
# ══════════════════════════════════════════════════════════════
def smooth_signal(sig):
    if len(sig) < 12: return sig
    b, a = butter(2, SMOOTH_CUTOFF)
    return filtfilt(b, a, sig)

# ══════════════════════════════════════════════════════════════
#  REP SEGMENTATION — unified for calibration AND rehab
# ══════════════════════════════════════════════════════════════

def segment_rehab_rep(angles: np.ndarray):
    """
    Find complete valley→peak→valley rep segments in a signal.

    Returns list of (start_idx, end_idx) pairs.  Each pair is the
    tightest slice that contains one full flex/extend cycle.

    Used BOTH during calibration (so threshold durations are calibrated
    on the same measurement basis) and during live scoring.
    """
    if len(angles) < int(SAMPLE_RATE * 2):
        return []

    s = smooth_signal(angles)
    valleys, _ = find_peaks(-s, prominence=VALLEY_PROMINENCE,
                             distance=VALLEY_DISTANCE)
    peaks, _   = find_peaks( s, prominence=VALLEY_PROMINENCE,
                             distance=VALLEY_DISTANCE)

    if len(valleys) < 2 or len(peaks) < 1:
        return []

    reps = []
    for i in range(len(valleys) - 1):
        v0, v1 = valleys[i], valleys[i + 1]
        # there must be at least one peak between the two valleys
        mids = peaks[(peaks > v0) & (peaks < v1)]
        if not len(mids):
            continue
        seg = angles[v0:v1]
        duration = (v1 - v0) / SAMPLE_RATE
        if seg.max() - seg.min() >= MIN_ROM and duration >= MIN_DURATION:
            reps.append((v0, v1))

    return reps


def segment_single_rep_from_window(window_df: pd.DataFrame):
    """
    Extract the single rep from a timed calibration window.

    Strategy — use the SAME valley→valley detector so calibration durations
    match live-scoring durations exactly.  Falls back to deviation-from-rest
    only if the valley detector finds no rep (patient moved too slowly to
    create two clear valleys within the window).

    Returns (seg_df, method_name).
    """
    angles = compute_elbow_angles(window_df)

    # ── Primary: valley→valley (same as live scoring) ──────────────
    reps = segment_rehab_rep(angles)
    if reps:
        # take the rep with the largest ROM
        best = max(reps, key=lambda r: angles[r[0]:r[1]].max() - angles[r[0]:r[1]].min())
        s, e = best
        return window_df.iloc[s:e].reset_index(drop=True), "valley_detector"

    # ── Fallback: deviation-from-rest (single peak case) ───────────
    smooth = smooth_signal(angles)
    rest_n = max(1, int(0.5 * SAMPLE_RATE))
    rest_angle = float(np.mean(
        np.concatenate([smooth[:rest_n], smooth[-rest_n:]])
    ))
    deviations  = np.abs(smooth - rest_angle)
    extreme_idx = int(np.argmax(deviations))
    max_dev     = float(deviations[extreme_idx])

    if max_dev < MIN_ROM * 0.5:
        # No real movement detected — return full window, caller will check ROM
        return window_df.reset_index(drop=True), "full_window"

    threshold = max_dev * 0.25
    start = 0
    for i in range(extreme_idx, -1, -1):
        if deviations[i] < threshold:
            start = i
            break
    end = len(window_df) - 1
    for i in range(extreme_idx, len(deviations)):
        if deviations[i] < threshold:
            end = i
            break

    margin = int(0.3 * SAMPLE_RATE)
    start  = max(0, start - margin)
    end    = min(len(window_df) - 1, end + margin)

    if end - start < int(MIN_DURATION * SAMPLE_RATE):
        return window_df.reset_index(drop=True), "full_window"

    return window_df.iloc[start:end].reset_index(drop=True), "deviation_fallback"


# ══════════════════════════════════════════════════════════════
#  CALIBRATION PHASE
# ══════════════════════════════════════════════════════════════
def run_calibration(ser, session, logger):
    """
    Two-step calibration:
      Step 1 — Auto-zero: patient holds arm straight, measure resting angle.
      Step 2 — N timed calibration reps.

    IMPORTANT: Each timed window is segmented using the SAME valley→valley
    detector used during live scoring.  This guarantees that duration_s
    (and all other features) are measured on the same scale at both phases,
    eliminating false speed_cheating positives.
    """
    # ── Step 1: Auto-zero ──────────────────────────────────────
    print("\n" + "="*55)
    print("  STEP 1 — AUTO-ZERO BASELINE")
    print("="*55)
    print("  Hold your arm STRAIGHT DOWN and perfectly still.")
    input("  Press Enter when ready...")

    ser.reset_input_buffer()
    baseline_angles = []
    t_start = time.time()
    sys.stdout.write("  Measuring zero")
    while time.time() - t_start < 2.0:
        result = read_one_raw(ser)
        if result:
            _, angle = result
            baseline_angles.append(angle)
            if len(baseline_angles) % 10 == 0:
                sys.stdout.write("."); sys.stdout.flush()

    resting_offset = float(np.mean(baseline_angles)) if baseline_angles else 0.0
    print(f"\n  ✅ Zero set! Raw offset: {resting_offset:.1f}°\n")

    # ── Step 2: Timed calibration reps ────────────────────────
    print("="*55)
    print(f"  STEP 2 — CALIBRATION  ({N_CALIB_REPS} good reps)")
    print("="*55)
    print("  Perform full elbow flexion/extension — slow and controlled.")
    print("  Start and END with arm fully extended (hanging down).\n")

    calib_max_rom = 0.0
    accepted = 0

    while accepted < N_CALIB_REPS:
        rep_num = accepted + 1
        print(f"  ── REP {rep_num} of {N_CALIB_REPS} ─────────────────────────────")
        for i in range(3, 0, -1):
            sys.stdout.write(f"\r  Get ready... {i}"); sys.stdout.flush(); time.sleep(1)
        print(f"\r  GO! Curl your arm fully, then extend fully.   ")
        print(f"  Return your arm to full extension before the timer ends.\n")

        ser.reset_input_buffer()
        rep_data = []
        t_start  = time.time()
        live_max = 0.0

        while time.time() - t_start < CALIB_DURATION:
            elapsed = time.time() - t_start
            result  = read_one_raw(ser)
            if not result: continue
            vals, raw_angle = result
            angle = raw_angle - resting_offset
            rep_data.append([elapsed * 1000.0] + vals)
            if angle > live_max: live_max = angle
            # Compact single-line display to avoid terminal wrapping
            t_str = time_bar(elapsed, CALIB_DURATION, width=20)
            a_str = angle_bar(angle, 0, 130, width=20)
            sys.stdout.write(
                f"\r  {t_str}  {a_str}  Peak:{live_max:5.1f}°"
            )
            sys.stdout.flush()
        print()

        if not rep_data:
            print("  ⚠️  No data received. Check serial connection.\n")
            continue

        window_df = pd.DataFrame(rep_data, columns=[
            "Timestamp","qx1","qy1","qz1","qw1","qx2","qy2","qz2","qw2"])

        # ── KEY FIX: segment using the same detector as live scoring ──
        seg_df, method = segment_single_rep_from_window(window_df)
        features = extract_features(seg_df)

        print(f"  Segmentation: {method}  →  duration={features.duration_s:.1f}s  ROM={features.rom:.1f}°")

        if method == "full_window":
            print(f"  ⚠️  Could not auto-segment — thresholds may be less accurate.")
            print(f"     Ensure arm is fully extended at START and END of the window.\n")

        if features.rom < MIN_ROM:
            print(f"  ❌ ROM too small ({features.rom:.1f}°, need ≥{MIN_ROM}°). Try again.\n")
            continue

        calib_max_rom = max(calib_max_rom, features.rom)
        # Pass the segmented DataFrame so calibration sees the same features as live scoring
        session._calib_reps.append(features)
        if len(session._calib_reps) >= session.n_calib_reps:
            session._finalise_calibration()

        logger.log(seg_df, features, label="calibration", patient_id=PATIENT_ID)
        accepted += 1
        print(f"  ✅ Rep {rep_num} accepted!  ROM={features.rom:.1f}°  dur={features.duration_s:.1f}s\n")

        if session.phase == "rehab":
            calib = session.calibration
            print("  ╔══════════════════════════════════════════╗")
            print("  ║      CALIBRATION COMPLETE ✓              ║")
            print("  ╚══════════════════════════════════════════╝")
            print(f"  Target ROM      : {calib.target_rom:.1f}°")
            print(f"  SPARC target    : {calib.target_sparc:.2f}")
            print(f"  Max compensation: {calib.max_comp_mean:.3f}")
            print(f"  Max shoulder    : {calib.max_upper_rom_y:.2f}°")
            print(f"  Duration range  : {calib.min_duration_s:.1f}s – {calib.max_duration_s:.1f}s")
            print()

    return resting_offset, calib_max_rom


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    try:
        import serial
    except ImportError:
        sys.exit("ERROR: pyserial not installed. Run: pip install pyserial")

    session     = RehabSession(n_calib_reps=N_CALIB_REPS)
    logger      = DataLogger(patient_id=PATIENT_ID)
    all_rep_data = []  # <--- ADD THIS LINE TO STORE STITCHED DATA
    rep_counter = 0

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(1)
    except Exception as e:
        print(f"\nERROR: Cannot open {SERIAL_PORT}: {e}")
        print("Check: device plugged in, correct COM port, no other app using it.")
        sys.exit(1)

    print("="*55)
    print("  Elbow Rehab — Live Quality Judge")
    print("="*55)
    print(f"  Port: {SERIAL_PORT}  |  {SAMPLE_RATE:.0f} Hz  |  Patient: {PATIENT_ID}")

    try:
        with ser:
            rest_rel, calib_max = run_calibration(ser, session, logger)

            print("  🏋  START YOUR REHAB SESSION!")
            print("  (Ctrl+C to end and see summary)\n")

            buffer       = deque(maxlen=WINDOW_FRAMES)
            sample_idx   = 0
            # Track the buffer-relative index where the last rep ended
            # so we don't re-score the same rep from overlapping windows.
            last_rep_end_buf_idx = -COOLDOWN_SAMPLES
            ser.reset_input_buffer()

            while True:
                # 1. User Trigger
                print(f"\n  ── Rep {rep_counter + 1} ─────────────────────────────")
                input("  Press Enter and then START your elbow movement...")
                
                rep_data = []
                t_start = time.time()
                REP_WINDOW_SEC = 9.0  # Gives you 9 seconds to complete the rep
                
                # 2. Dedicated 9-second Recording Loop
                while time.time() - t_start < REP_WINDOW_SEC:
                    result = read_one_raw(ser)
                    if not result: continue
                    vals, raw_angle = result
                    angle = raw_angle - resting_offset
                    elapsed = time.time() - t_start
                    
                    rep_data.append([elapsed * 1000.0] + vals)
                    
                    # Compact single-line display to avoid terminal wrapping
                    t_str = time_bar(elapsed, REP_WINDOW_SEC, width=20)
                    a_str = angle_bar(angle, 0, calib_max, width=20)
                    sys.stdout.write(f"\r  {t_str}  {a_str}")
                    sys.stdout.flush()
                print() # New line after timer ends

                if not rep_data:
                    print("  ⚠️ No data captured. Check connection.")
                    continue

                # 3. Intelligent Analysis of the Window
                full_window_df = pd.DataFrame(rep_data, columns=[
                    "Timestamp","qx1","qy1","qz1","qw1","qx2","qy2","qz2","qw2"])

                # This trims the 9s down to the ACTUAL movement time
                seg_df, method = segment_single_rep_from_window(full_window_df)
                
                features = extract_features(seg_df)
                score    = session.judge_rep(seg_df)
                all_rep_data.append(seg_df) # Stitches only the movement parts
                rep_counter += 1

                # 4. Display Individual Results
                feedback = session.get_feedback(score)
                calib = session.calibration
                
                print(f"  ✅ Rep {rep_counter} Analyzed (Segmented via: {method})")
                print(f"  │  Movement Time : {features.duration_s:.1f}s  (Out of {REP_WINDOW_SEC:.1f}s window)")
                print(f"  │  ROM           : {features.rom:.1f}°")
                print(f"  │  Quality       : {quality_bar(score.quality_score)}")
                
                # Show Isolation Forest anomaly score if available
                anomaly_score = score.details.get('anomaly_score')
                if anomaly_score is not None:
                    status = "NORMAL" if anomaly_score >= 0.0197 else "ANOMALOUS"
                    print(f"  │  Anomaly Score : {anomaly_score:.3f}  ({status})")

                # Show SPARC (smoothness metric)
                print(f"  │  SPARC         : {features.sparc:.1f}  (less negative = smoother)")

                # 3-tier result display
                if score.status == "PASS":
                    print(f"  │  Result        : \033[92m✅ PASS\033[0m")
                elif score.status == "WARN":
                    print(f"  │  Result        : \033[93m⚠️  WARN\033[0m — pattern unusual")
                    for msg in feedback:
                        print(f"  │  → {msg}")
                else:
                    print(f"  │  Result        : \033[91m❌ FAIL\033[0m — {score.faults}")
                    for msg in feedback:
                        print(f"  │  → {msg}")
                print(f"  └────────────────────────────────────────────\n")

    except KeyboardInterrupt:
        print("\n\n  Session ended.")

        if all_rep_data and rep_counter > 0:
            # Use the judgment system's own summary — not recomputed raw metrics
            summary = session.summary()
            pass_rate    = summary.get('pass_rate', 0.0)
            most_common  = summary.get('most_common_fault', None)
            mean_quality = summary.get('mean_quality', 0.0)
            fault_counts = summary.get('fault_counts', {})

            print(f"\n  ══ SESSION SUMMARY ══════════════════════════")
            print(f"  Total reps    : {summary['total_reps']}")
            print(f"  Passed        : {summary.get('passed_reps',0)}  ({pass_rate*100:.0f}%)")
            print(f"  Warned        : {summary.get('warned_reps',0)}  ({summary.get('warn_rate',0)*100:.0f}%)")
            print(f"  Failed        : {summary.get('failed_reps',0)}  ({summary.get('fail_rate',0)*100:.0f}%)")
            print(f"  Mean quality  : {mean_quality:.2f}")
            if fault_counts:
                print(f"  Fault breakdown:")
                for fault, count in sorted(fault_counts.items(), key=lambda x: -x[1]):
                    print(f"    {fault:<25} : {count} reps")
            print(f"  ═════════════════════════════════════════════")

            # Compute session aggregates from scored reps
            avg_rom = 0.0
            overall_sparc = 0.0
            total_duration = 0.0
            if session._scores:
                avg_rom = float(np.mean([s.features.rom for s in session._scores]))
                overall_sparc = float(np.mean([s.features.sparc for s in session._scores]))
                total_duration = float(np.sum([s.features.duration_s for s in session._scores]))

            # Coach reads from actual judgment results, not raw recomputed metrics
            if pass_rate >= 0.80:
                coach_text = (f"Great session! {summary['passed_reps']} of "
                            f"{summary['total_reps']} reps passed. Keep it up.")
            elif most_common == 'insufficient_rom':
                coach_text = ("Focus on full range — try to straighten and bend "
                            "your arm completely on each rep.")
            elif most_common == 'shoulder_swing':
                coach_text = ("Your shoulder is taking over — keep your upper arm "
                            "pinned to your side throughout the movement.")
            elif most_common == 'trunk_lean':
                coach_text = ("You are rocking your body — sit upright and isolate "
                            "the movement to your elbow only.")
            elif most_common == 'compensation':
                coach_text = ("Upper arm is compensating — focus on moving only "
                            "from the elbow joint.")
            elif most_common == 'tremor':
                coach_text = ("Movement is unsteady — slow down and focus on "
                            "a smooth controlled arc.")
            elif most_common == 'speed_cheating':
                coach_text = ("Reps are too fast — slow down for better "
                            "muscle engagement.")
            elif most_common == 'unsmooth':
                coach_text = ("Movement is jerky — aim for one fluid arc "
                            "without stopping mid-rep.")
            elif most_common == 'statistical_anomaly':
                coach_text = ("Your movement pattern was inconsistent across reps "
                            "— check your starting position and try to repeat "
                            "the same motion each time.")
            else:
                coach_text = (f"Pass rate {pass_rate*100:.0f}% — "
                            "focus on consistency next session.")

            # summary_stats for report now comes from the judgment system
            summary_stats = {
                'total_reps'    : summary['total_reps'],
                'passed_reps'   : summary['passed_reps'],
                'pass_rate'     : pass_rate,
                'mean_quality'  : mean_quality,
                'avg_rom'       : avg_rom,
                'overall_sparc' : overall_sparc,
                'duration'      : total_duration,
                'most_common_fault': most_common or 'none',
                'fault_counts'  : fault_counts,
            }

            try:
                report_filename = generate_rehab_report(PATIENT_ID, summary_stats, coach_text)
                print(f"\n  ══ AI COACH ══════════════════════════════")
                print(f"  {coach_text}")
                print(f"  ══════════════════════════════════════════")
                print(f"\n  ✅ Report saved: {report_filename}")
            except Exception as e:
                print(f"  ⚠️ Could not generate report: {e}")
        else:
            print("  ⚠️ No reps completed.") 

if __name__ == "__main__":
    main()
