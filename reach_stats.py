"""
ON vs OFF statistics for the reach-to-poke task, computed per patient on every poke found.

Run in the same folder as make_reach_figure.py and the DLC .h5 files:
    python reach_stats.py
Outputs: reach_metrics.csv (one row per reach), reach_stats.csv (tests), stats_figure.png/.pdf

Measures (index fingertip, top view, distances in "reach lengths" = median rest-to-ball
distance for that patient, because the cameras are not calibrated to mm):
  movement_time_s    onset -> arrival at the ball
  peak_speed         peak fingertip speed during the reach (reach lengths / s)
  path_ratio         path length / straight-line distance (1 = perfectly straight)
  max_lateral_dev    largest sideways deviation from the start->ball line
  smoothness_ldlj    log dimensionless jerk of the reach (more negative = less smooth);
                     used instead of in-reach tremor because reaches last only ~0.2-0.6 s,
                     too short to separate a 4-8 Hz tremor from the reach itself
  tremor_target      RMS of 3-12 Hz fingertip motion while held at the ball
                     (fingertip within 3% of its furthest point; up to 1 s)

Test: two-sided Mann-Whitney U per measure, Holm-corrected across the 6 measures within each
patient; effect size = Cliff's delta (OFF vs ON; positive = larger when OFF).
"""
import os, sys, glob
import numpy as np
from scipy import stats, signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_reach_figure as M

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else HERE
M.DATA_DIR = DATA_DIR
BAND = (3.0, 12.0)          # Hz, essential-tremor band
TARGET_MAX = 1.0            # s, cap on target window
SPEED_LP = 15.0             # Hz, low-pass used only for speed / path length (removes DLC jitter)

METRICS = [("movement_time_s", "Movement time (s)"),
           ("peak_speed", "Peak speed (reach lengths/s)"),
           ("path_ratio", "Path length / straight line"),
           ("max_lateral_dev", "Max lateral deviation (reach lengths)"),
           ("smoothness_ldlj", "Smoothness (log dimensionless jerk)"),
           ("tremor_target", "Tremor RMS at target (reach lengths)")]

b_bp, a_bp = signal.butter(2, BAND, btype="band", fs=M.FPS)
b_lp, a_lp = signal.butter(2, SPEED_LP, btype="low", fs=M.FPS)


def band_rms(xy):
    if len(xy) < 30:
        return np.nan
    f = signal.filtfilt(b_bp, a_bp, xy - xy.mean(0), axis=0, padlen=min(len(xy) - 1, 27))
    return float(np.sqrt(np.mean(np.sum(f ** 2, axis=1))))


def gather():
    """All pokes, per patient & condition, with the fingertip track in pixels (y up)."""
    files = [M.parse_name(f) for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))]
    out = []
    for pt in M.PATIENTS:
        pf = [f for f in files if f["pt"] == pt]
        cam = M.CAMERA
        if cam == "auto":
            score = {c: np.mean([np.nanmean(M.read_dlc(f["path"])[:, M.BODYPARTS.index(M.PLOT_MARKER), 2])
                                 for f in pf if f["cam"] == c]) for c in ("L", "R")}
            cam = max(score, key=score.get)
        for f in pf:
            if f["cam"] != cam:
                continue
            d = M.read_dlc(f["path"])
            seg_xy, _ = M.marker_xy(d, M.SEG_MARKER)
            tip, _ = M.marker_xy(d, M.PLOT_MARKER)
            tip = np.column_stack([tip[:, 0], -tip[:, 1]])
            # ball-ward progress (same definition used for poke detection)
            for k, (on, arr, ret) in enumerate(M.find_pokes(seg_xy)):
                out.append(dict(pt=pt, cond=f["cond"], cam=cam, trial=f["trial"], poke=k + 1,
                                on=on, arr=arr, ret=ret, tip=tip))
    return out


def compute(rows):
    recs = []
    for pt in M.PATIENTS:
        pr = [r for r in rows if r["pt"] == pt]
        # per-condition frame: rotate start->ball to +y; per-patient scale
        frames, lengths = {}, []
        for cond in ("OFF", "ON"):
            cr = [r for r in pr if r["cond"] == cond]
            v = np.mean([r["tip"][r["arr"]] - r["tip"][r["on"]] for r in cr], 0)
            ang = np.pi / 2 - np.arctan2(v[1], v[0])
            frames[cond] = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            lengths += [np.linalg.norm(r["tip"][r["arr"]] - r["tip"][r["on"]]) for r in cr]
        scale = np.median(lengths)
        for r in pr:
            R = frames[r["cond"]]
            tip = (r["tip"] - r["tip"][r["on"]]) @ R.T / scale
            reach = tip[r["on"]: r["arr"] + 1]
            # target window: from arrival until the hand starts back (or TARGET_MAX)
            fwd = tip[:, 1]
            seg = fwd[r["arr"]: r["ret"]]
            stay = r["arr"] + np.flatnonzero(seg >= seg.max() - 0.03)
            t0 = stay[0] if len(stay) else r["arr"]
            t1 = min(stay[-1] if len(stay) else r["arr"], t0 + int(TARGET_MAX * M.FPS))
            target = tip[t0: t1 + 1]
            smooth = signal.filtfilt(b_lp, a_lp, tip[max(r["on"] - 20, 0): r["arr"] + 21], axis=0)
            smooth = smooth[r["on"] - max(r["on"] - 20, 0):][: len(reach)]
            vel = np.diff(smooth, axis=0) * M.FPS
            path = np.sum(np.linalg.norm(np.diff(smooth, axis=0), axis=1))
            recs.append(dict(
                patient=pt, condition=r["cond"], camera=r["cam"], trial=r["trial"], poke=r["poke"],
                movement_time_s=(r["arr"] - r["on"]) / M.FPS,
                peak_speed=np.max(np.linalg.norm(vel, axis=1)),
                path_ratio=path / np.linalg.norm(smooth[-1] - smooth[0]),
                max_lateral_dev=np.max(np.abs(reach[:, 0])),
                smoothness_ldlj=ldlj(smooth, (r["arr"] - r["on"]) / M.FPS),
                tremor_target=band_rms(target),
                target_window_s=len(target) / M.FPS))
    return recs


def ldlj(pos, T):
    """Log dimensionless jerk (Balasubramanian et al. 2015) from position."""
    dt = 1 / M.FPS
    v = np.gradient(pos, dt, axis=0)
    j = np.gradient(np.gradient(v, dt, axis=0), dt, axis=0)
    vpeak = np.max(np.linalg.norm(v, axis=1))
    return float(-np.log(T ** 3 / vpeak ** 2 * np.sum(np.sum(j ** 2, axis=1)) * dt))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0
    for i, k in enumerate(order):
        run = max(run, min(1, (len(p) - i) * p[k]))
        adj[k] = run
    return adj


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def main():
    recs = compute(gather())
    keys = list(recs[0].keys())
    with open(os.path.join(DATA_DIR, "reach_metrics.csv"), "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in recs:
            fh.write(",".join(f"{r[k]:.5g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")

    results = []
    for pt in M.PATIENTS:
        rows = []
        for m, _ in METRICS:
            off = np.array([r[m] for r in recs if r["patient"] == pt and r["condition"] == "OFF"])
            on = np.array([r[m] for r in recs if r["patient"] == pt and r["condition"] == "ON"])
            off, on = off[~np.isnan(off)], on[~np.isnan(on)]
            U, p = stats.mannwhitneyu(off, on, alternative="two-sided")
            rows.append(dict(patient=pt, measure=m, n_off=len(off), n_on=len(on),
                             median_off=np.median(off), iqr_off=stats.iqr(off),
                             median_on=np.median(on), iqr_on=stats.iqr(on),
                             U=U, p=p, cliffs_delta=cliffs_delta(off, on)))
        for r, pa in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = pa
        results += rows

    cols = ["patient", "measure", "n_off", "n_on", "median_off", "iqr_off", "median_on",
            "iqr_on", "U", "p", "p_holm", "cliffs_delta"]
    with open(os.path.join(DATA_DIR, "reach_stats.csv"), "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in results:
            fh.write(",".join(f"{r[c]:.4g}" if isinstance(r[c], (float, np.floating)) else str(r[c])
                              for c in cols) + "\n")
    for r in results:
        star = "*" if r["p_holm"] < 0.05 else " "
        print(f"{r['patient']} {r['measure']:16s} OFF {r['median_off']:.4f} (n={r['n_off']:2d})  "
              f"ON {r['median_on']:.4f} (n={r['n_on']:2d})  p={r['p']:.4f}  p_holm={r['p_holm']:.4f}{star} "
              f"delta={r['cliffs_delta']:+.2f}")

    # ---- figure: one panel per measure, patients side by side, OFF vs ON dots ----
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.6))
    colors = {"OFF": "#2b2b2b", "ON": "#9a9a9a"}
    rng = np.random.default_rng(0)
    for ax, (m, label) in zip(axes.ravel(), METRICS):
        for i, pt in enumerate(M.PATIENTS):
            for j, cond in enumerate(("OFF", "ON")):
                v = np.array([r[m] for r in recs if r["patient"] == pt and r["condition"] == cond])
                x = i * 2.6 + j
                ax.scatter(x + rng.uniform(-0.15, 0.15, len(v)), v, s=10, color=colors[cond],
                           alpha=0.75, lw=0)
                ax.hlines(np.nanmedian(v), x - 0.3, x + 0.3, color="#c0392b" if False else "black", lw=1.6)
            res = next(r for r in results if r["patient"] == pt and r["measure"] == m)
            ptxt = "p<0.001" if res["p_holm"] < 0.001 else f"p={res['p_holm']:.3f}"
            ax.text(i * 2.6 + 0.5, 1.0, ptxt + (" *" if res["p_holm"] < 0.05 else ""),
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
        ax.set_xticks([0, 1, 2.6, 3.6])
        ax.set_xticklabels(["OFF", "ON", "OFF", "ON"], fontsize=8)
        ax.text(0.5, -0.2, "Patient 2", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color="#555")
        ax.text(3.1, -0.2, "Patient 3", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color="#555")
        ax.set_title(label, fontsize=9, pad=14)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.text(0.01, 0.005, "Dots = individual reaches; bar = median. p = Mann-Whitney U, Holm-corrected "
             "across 6 measures within each patient.", fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 1), h_pad=3.0)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(DATA_DIR, f"stats_figure.{ext}"), dpi=300)
    print("saved reach_metrics.csv, reach_stats.csv, stats_figure.png/.pdf")


if __name__ == "__main__":
    main()
