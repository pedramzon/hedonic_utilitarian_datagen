"""
generate_synthetic_dataset.py
==============================

Reproduces the synthetic multimodal neuromarketing dataset described in:

    "Hedonic and Utilitarian Values in Consumer Interactions with LLM-Based
    AI Agents: A Marketing Perspective" -- Zonoubi & Salehmohammadnia.

This script implements, deterministically and from documented first
principles, every generative step described in the paper's "Results"
section and Equation (4):

  1. 400 simulated shopping sessions x 30 five-second windows/session
     = 12,000 samples (as stated in the paper).
  2. Per-window latent response values h (hedonic) and u (utilitarian)
     drawn independently as h ~ Beta(2,2), u ~ Beta(2,2).
  3. Raw multimodal signal streams per window:
       - 64-channel EEG, sampled at 256 Hz (5 s window -> 1280 samples/ch).
         * A "frontal" subset of channels carries band-power that is a
           monotonically increasing function of h (additive Gaussian
           noise, sigma = 0.15), reflecting the paper's stated design
           ("frontal EEG band-power features were generated as a
           monotonically increasing function of h").
         * A "parietal" subset carries band-power that is a monotonically
           increasing function of u (same noise level), reflecting the
           paper's stated design for parietal activity / decision-related
           processing.
         * Remaining channels carry background 1/f-shaped noise only,
           uncoupled to (h,u), to emulate non-informative channels in a
           64-channel montage.
       - Eye-tracking, sampled at 500 Hz (5 s window -> 2500 samples),
         producing (x,y) gaze coordinates plus a derived fixation-duration
         feature that is a monotonically increasing function of u.
       - Galvanic skin response (GSR), sampled at 25 Hz, whose level is a
         function of max(h,u) (general arousal), per the paper.
       - An "LLM sentiment" scalar per window, a function of h, standing
         in for sentiment extracted from the LLM-mediated conversational
         log for that window.
  4. Per-window AGGREGATE FEATURES (the ones actually consumed by
     AS-GNN / baselines) are then computed from the raw streams:
       frontal_eeg_bandpower, parietal_eeg_bandpower, fixation_duration,
       gsr_level, llm_sentiment.
  5. Joint response r = lambda*h + (1-lambda)*u  (Eq. 4), lambda = 0.5.
  6. Discrete label: 1 = "hedonic-dominant" if r >= 0.5 else
     0 = "utilitarian-dominant" (thresholding rule stated in the
     "Hedonic-Utilitarian Response Modeling" subsection and used to
     derive ground truth in Results).

DESIGN CHOICE FLAGGED FOR THE READER
-------------------------------------
The paper specifies that each feature is "a monotonically increasing
function of h/u" with additive Gaussian noise (sigma = 0.15) but does not
name a specific nonlinearity. This script uses the identity mapping
(feature = latent + noise, clipped to [0,1]) as the simplest, fully
transparent, monotonically increasing function consistent with every
explicit statement in the paper. If a different nonlinearity was actually
used when the original results were produced, only the `_shape()` calls
below need to change -- the surrounding pipeline (sampling, noise level,
window/session structure, labeling rule) is unaffected.

Usage
-----
    python generate_synthetic_dataset.py --seed 0 --n_sessions 400 \
        --windows_per_session 30 --out dataset_seed0.csv

    # Representative public sample (smaller, for deposit alongside the paper):
    python generate_synthetic_dataset.py --seed 0 --n_sessions 20 \
        --windows_per_session 30 --out sample_dataset.csv --raw_sample 3
"""

import argparse
import json
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Fixed dataset-level constants (as stated in the paper's Results section)
# ----------------------------------------------------------------------
N_SESSIONS_FULL = 400          # "400 simulated shopping sessions"
WINDOWS_PER_SESSION = 30       # "30 windows per session"
WINDOW_SECONDS = 5             # "each sample spanning a 5-second window"

N_EEG_CHANNELS = 64            # "64-channel EEG recordings"
EEG_FS_HZ = 256                # sampling rate for the 64-ch EEG montage
N_FRONTAL_CH = 16              # subset of channels treated as "frontal"
N_PARIETAL_CH = 16             # subset of channels treated as "parietal"

EYE_FS_HZ = 500                # "500 Hz eye-tracking coordinates"
GSR_FS_HZ = 25                 # GSR is a slow physiological signal

FEATURE_NOISE_SIGMA = 0.15     # "additive Gaussian noise (sigma = 0.15)"
BETA_A, BETA_B = 2.0, 2.0      # "h ~ Beta(2,2) and u ~ Beta(2,2)"
LAMBDA_JOINT = 0.5             # lambda in r = lambda*h + (1-lambda)*u (Eq. 4)
LABEL_THRESHOLD = 0.5          # "thresholding the joint response r at r = 0.5"


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _shape(latent, rng, sigma=FEATURE_NOISE_SIGMA):
    """
    The monotonically-increasing-function-of-the-latent used throughout.
    See the "DESIGN CHOICE FLAGGED FOR THE READER" note in the module
    docstring: identity + Gaussian noise, clipped to [0,1].
    """
    return _clip01(latent + rng.normal(0.0, sigma, size=np.shape(latent)))


# ----------------------------------------------------------------------
# Raw multimodal stream simulation (per window)
# ----------------------------------------------------------------------
def simulate_eeg_window(h, u, rng, fs=EEG_FS_HZ, seconds=WINDOW_SECONDS,
                         n_channels=N_EEG_CHANNELS,
                         n_frontal=N_FRONTAL_CH, n_parietal=N_PARIETAL_CH):
    """
    Simulate one 5-second, 64-channel EEG window.

    Channel groups:
      0                      .. n_frontal-1                -> frontal
      n_frontal              .. n_frontal+n_parietal-1      -> parietal
      n_frontal+n_parietal   .. n_channels-1                -> other

    Frontal channel band-power is coupled to h; parietal channel
    band-power is coupled to u; both with sigma=0.15 additive noise.
    Other channels are uncoupled background activity.
    Returns the raw (n_channels, n_samples) time series plus the two
    aggregate band-power features actually used for classification.
    """
    n_samples = int(fs * seconds)
    t = np.arange(n_samples) / fs
    eeg = np.zeros((n_channels, n_samples), dtype=np.float32)

    # target band powers (the values the aggregate features will estimate)
    frontal_power = _shape(h, rng)     # increasing function of h
    parietal_power = _shape(u, rng)    # increasing function of u

    alpha_freq = 10.0  # Hz, alpha band carrier used as a stand-in oscillation

    for ch in range(n_channels):
        if ch < n_frontal:
            amp = frontal_power
        elif ch < n_frontal + n_parietal:
            amp = parietal_power
        else:
            amp = 0.3  # fixed baseline amplitude, uncoupled to (h,u)
        phase = rng.uniform(0, 2 * np.pi)
        signal = amp * np.sin(2 * np.pi * alpha_freq * t + phase)
        # 1/f-like background noise typical of scalp EEG
        pink_noise = np.cumsum(rng.normal(0, 1, n_samples))
        pink_noise = pink_noise / (np.max(np.abs(pink_noise)) + 1e-8)
        eeg[ch] = signal + 0.2 * pink_noise

    return eeg, float(frontal_power), float(parietal_power)


def simulate_eyetracking_window(u, rng, fs=EYE_FS_HZ, seconds=WINDOW_SECONDS):
    """
    Simulate one 5-second eye-tracking window (500 Hz gaze coordinates)
    and derive the fixation-duration feature, an increasing function of u.
    """
    n_samples = int(fs * seconds)
    fixation_duration = _shape(u, rng)  # increasing function of u, in [0,1]

    # Longer normalized fixation_duration -> gaze drifts less (more fixated)
    drift_scale = 0.05 * (1.0 - fixation_duration) + 0.01
    x = np.cumsum(rng.normal(0, drift_scale, n_samples))
    y = np.cumsum(rng.normal(0, drift_scale, n_samples))
    x = x - x.mean()
    y = y - y.mean()

    gaze = np.stack([x, y], axis=0).astype(np.float32)
    return gaze, float(fixation_duration)


def simulate_gsr_window(h, u, rng, fs=GSR_FS_HZ, seconds=WINDOW_SECONDS):
    """
    Simulate one 5-second GSR window. Level is a function of max(h,u)
    (general arousal), per the paper.
    """
    n_samples = int(fs * seconds)
    gsr_level = _shape(max(h, u), rng)  # function of max(h,u)
    baseline = gsr_level * np.ones(n_samples, dtype=np.float32)
    tonic_drift = np.cumsum(rng.normal(0, 0.01, n_samples))
    gsr = baseline + tonic_drift
    return gsr.astype(np.float32), float(gsr_level)


def simulate_llm_sentiment(h, rng):
    """
    LLM sentiment feature: a function of h, reflecting affective content
    in conversational logs.
    """
    return float(_shape(h, rng))


# ----------------------------------------------------------------------
# Full window / session / dataset generation
# ----------------------------------------------------------------------
def generate_window(rng):
    """Generate one 5-second window: latent (h,u), raw streams, features, label."""
    h = float(rng.beta(BETA_A, BETA_B))
    u = float(rng.beta(BETA_A, BETA_B))

    eeg_raw, frontal_bp, parietal_bp = simulate_eeg_window(h, u, rng)
    gaze_raw, fixation_duration = simulate_eyetracking_window(u, rng)
    gsr_raw, gsr_level = simulate_gsr_window(h, u, rng)
    llm_sentiment = simulate_llm_sentiment(h, rng)

    r = LAMBDA_JOINT * h + (1 - LAMBDA_JOINT) * u          # Eq. 4
    label = int(r >= LABEL_THRESHOLD)  # 1 = hedonic-dominant, 0 = utilitarian-dominant

    features = {
        "h_true": h,
        "u_true": u,
        "r_true": r,
        "label_hedonic_dominant": label,
        "frontal_eeg_bandpower": frontal_bp,
        "parietal_eeg_bandpower": parietal_bp,
        "fixation_duration": fixation_duration,
        "gsr_level": gsr_level,
        "llm_sentiment": llm_sentiment,
    }
    raw = {"eeg": eeg_raw, "gaze": gaze_raw, "gsr": gsr_raw}
    return features, raw


def generate_dataset(n_sessions=N_SESSIONS_FULL,
                      windows_per_session=WINDOWS_PER_SESSION,
                      seed=0, keep_raw_for_n_windows=0):
    """
    Generate the full tabular dataset (one row per window) plus,
    optionally, raw signal arrays for the first `keep_raw_for_n_windows`
    windows (useful for producing a small illustrative raw-signal sample
    without inflating the deposited file size).
    """
    master_rng = np.random.default_rng(seed)
    rows = []
    raw_samples = []
    window_counter = 0

    for session_id in range(n_sessions):
        # Each session gets its own child RNG stream, seeded deterministically
        # from the master RNG, so runs are reproducible and sessions are
        # statistically independent.
        session_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
        for window_idx in range(windows_per_session):
            features, raw = generate_window(session_rng)
            features["session_id"] = session_id
            features["window_idx"] = window_idx
            features["global_window_id"] = window_counter
            rows.append(features)

            if window_counter < keep_raw_for_n_windows:
                raw_samples.append({"global_window_id": window_counter, **raw})

            window_counter += 1

    df = pd.DataFrame(rows)
    cols = ["global_window_id", "session_id", "window_idx",
            "h_true", "u_true", "r_true", "label_hedonic_dominant",
            "frontal_eeg_bandpower", "parietal_eeg_bandpower",
            "fixation_duration", "gsr_level", "llm_sentiment"]
    df = df[cols]
    return df, raw_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0,
                         help="RNG seed (paper reports mean+-SD over seeds 0-4, N=5 runs)")
    parser.add_argument("--n_sessions", type=int, default=N_SESSIONS_FULL)
    parser.add_argument("--windows_per_session", type=int, default=WINDOWS_PER_SESSION)
    parser.add_argument("--out", type=str, default="dataset.csv",
                         help="Output CSV path for the per-window feature table")
    parser.add_argument("--raw_sample", type=int, default=0,
                         help="If >0, also save this many windows' raw signal "
                              "arrays (EEG/gaze/GSR) to a companion .npz file")
    args = parser.parse_args()

    df, raw_samples = generate_dataset(
        n_sessions=args.n_sessions,
        windows_per_session=args.windows_per_session,
        seed=args.seed,
        keep_raw_for_n_windows=args.raw_sample,
    )
    df.to_csv(args.out, index=False)

    class_balance = df["label_hedonic_dominant"].mean()
    print(f"Wrote {len(df)} rows to {args.out}")
    print(f"Hedonic-dominant fraction: {class_balance:.3f} "
          f"(paper reports 52% hedonic / 48% utilitarian for the full 12,000-sample set)")

    if raw_samples:
        npz_path = args.out.rsplit(".", 1)[0] + "_raw_sample.npz"
        save_dict = {}
        for entry in raw_samples:
            gid = entry["global_window_id"]
            save_dict[f"eeg_{gid}"] = entry["eeg"]
            save_dict[f"gaze_{gid}"] = entry["gaze"]
            save_dict[f"gsr_{gid}"] = entry["gsr"]
        np.savez_compressed(npz_path, **save_dict)
        print(f"Wrote raw-signal sample for {len(raw_samples)} windows to {npz_path}")

    meta = {
        "seed": args.seed,
        "n_sessions": args.n_sessions,
        "windows_per_session": args.windows_per_session,
        "window_seconds": WINDOW_SECONDS,
        "eeg_channels": N_EEG_CHANNELS,
        "eeg_fs_hz": EEG_FS_HZ,
        "eye_fs_hz": EYE_FS_HZ,
        "gsr_fs_hz": GSR_FS_HZ,
        "feature_noise_sigma": FEATURE_NOISE_SIGMA,
        "beta_params": [BETA_A, BETA_B],
        "lambda_joint": LAMBDA_JOINT,
        "label_threshold": LABEL_THRESHOLD,
    }
    meta_path = args.out.rsplit(".", 1)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote generation metadata to {meta_path}")


if __name__ == "__main__":
    main()
