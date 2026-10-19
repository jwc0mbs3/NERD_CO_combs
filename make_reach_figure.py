"""
Reach-to-poke trajectory figure in the style of Groppa et al., Brain, Fig. 2
(STIM-OFF vs STIM-ON, all reaches overlaid, standing on a grey "table").

Input : DeepLabCut .h5 files, top-view cameras (2D), named like
        pt2_s9_lct_1_OFFDLC_...h5 / pt3_s10rct1_ONDLC_...h5
Output: reach_figure.png / reach_figure.pdf, reach_counts.csv

Pipeline
  1. load marker (x, y, likelihood); drop low-confidence points, fill short gaps
  2. find each poke: hand leaves the rest plate -> reaches the ball -> returns
  3. keep N middle pokes per condition (spread evenly across that condition's trials)
  4. cut each reach from movement onset to arrival at the ball + a short target window
  5. align: rest at origin, rotate so start->ball points the same way, scale per patient
  6. plot
No smoothing is applied, so tremor stays in the trajectories.
"""
import glob, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Rectangle

# ============================== SETTINGS =====================================
DATA_DIR    = os.path.dirname(os.path.abspath(__file__)) if len(sys.argv) < 2 else sys.argv[1]
FPS         = 180
BODYPARTS   = ["index_tip", "index_base", "back_hand", "wrist_top"]   # DLC order
SEG_MARKER  = "back_hand"     # marker used to find pokes (steady, high confidence)
PLOT_MARKER = "index_tip"     # marker drawn in the figure
CAMERA      = "auto"          # "auto" (best confidence per patient), "L" or "R"
N_POKES     = {"pt2": 15, "pt3": 15}  # max reaches drawn per panel (pt3 OFF has only 5)
TARGET_WIN  = 0.30            # s of trajectory kept after arriving at the ball
LIK_THRESH  = 0.6
PATIENTS    = ["pt2", "pt3"]  # row order (A, B, ...)
CONDITIONS  = [("OFF", "STIM-OFF"), ("ON", "STIM-ON")]
STYLE       = "2d"            # "2d" = plain top-view panels, "paper" = 3D-style table look
REACH_ANGLE = 90 if STYLE == "2d" else 55   # deg: direction start->ball points
# =============================================================================


# ---------- loading ----------------------------------------------------------
def _read_raw(path, n_bp):
    """Fallback reader for DLC's PyTables table when pandas/pytables is missing."""
    b = open(path, "rb").read()
    rec = np.dtype([("i", "<i8"), ("v", "<f4", (n_bp * 3,))])
    s = rec.itemsize // 8
    a = np.frombuffer(b[: len(b) // 8 * 8], dtype="<i8")
    hits = np.flatnonzero((a[:-s] >= 0) & (a[:-s] < 10_000_000) & (a[s:] == a[:-s] + 1))
    rows = {}
    for h in hits:
        for off in (h * 8, h * 8 + rec.itemsize):
            if off + rec.itemsize <= len(b):
                r = np.frombuffer(b, dtype=rec, count=1, offset=off)[0]
                rows.setdefault(int(r["i"]), r["v"])
    out = np.full((max(rows) + 1, n_bp * 3), np.nan, np.float32)
    for k, v in rows.items():
        out[k] = v
    return out.reshape(-1, n_bp, 3)


def read_dlc(path):
    """Returns array [frame, bodypart, (x, y, likelihood)] in BODYPARTS order."""
    try:
        import pandas as pd
        df = pd.read_hdf(path)
        scorer = df.columns.get_level_values(0)[0]
        return np.stack([df[scorer][bp][["x", "y", "likelihood"]].to_numpy()
                         for bp in BODYPARTS], axis=1)
    except Exception:
        return _read_raw(path, len(BODYPARTS))


def parse_name(f):
    f = os.path.basename(f)
    stem = re.split(r"_?(ON|OFF)DLC", f)[0]
    return dict(pt=re.match(r"(pt\d+)", f).group(1),
                cond="ON" if re.search(r"_?ONDLC", f) else "OFF",
                cam="L" if re.search(r"(lct|ltc)", stem) else "R",
                trial=stem, path=os.path.join(DATA_DIR, f))


def marker_xy(d, name):
    j = BODYPARTS.index(name)
    xy = d[:, j, :2].astype(float).copy()
    bad = ~(d[:, j, 2] >= LIK_THRESH)
    xy[bad] = np.nan
    t = np.arange(len(xy))
    for k in range(2):
        ok = ~np.isnan(xy[:, k])
        xy[:, k] = np.interp(t, t[ok], xy[ok, k])
    return xy, bad


# ---------- poke detection ---------------------------------------------------
def find_pokes(xy):
    """(onset, arrival, return) frames for every out-and-back poke."""
    # rest plate = the end of the path farthest from the ball; use extremes of
    # the main movement axis
    c = xy - np.median(xy, 0)
    u, s, vt = np.linalg.svd(c[~np.isnan(c).any(1)][::5], full_matrices=False)
    proj = c @ vt[0]
    lo, hi = np.percentile(proj, [2, 98])
    # rest plate is at the bottom of the top-view image (larger pixel y)
    if vt[0][1] > 0:
        proj, lo, hi = -proj, -hi, -lo
    p = (proj - lo) / (hi - lo)                 # 0 = rest, 1 = ball
    k = 9
    ps = np.convolve(p, np.ones(k) / k, mode="same")  # only for detection
    out = ps > 0.5
    ups = np.flatnonzero(~out[:-1] & out[1:]) + 1
    downs = np.flatnonzero(out[:-1] & ~out[1:]) + 1
    pokes = []
    for u_ in ups:
        nxt = downs[downs > u_]
        if not len(nxt):
            break
        r = nxt[0]
        on = u_
        while on > 0 and ps[on - 1] < ps[on] and ps[on] > 0.05:
            on -= 1                               # walk back to movement start
        seg = ps[u_:r]
        arrive = u_ + np.flatnonzero(seg >= 0.92 * seg.max())[0]
        if p[on] < 0.3 and seg.max() > 0.8:       # real rest->ball poke
            pokes.append((on, arrive, r))
    return pokes


def pick_middle(n_total, n_keep):
    if n_keep >= n_total:
        return list(range(n_total))
    start = (n_total - n_keep) // 2
    return list(range(start, start + n_keep))


# ---------- build reaches ----------------------------------------------------
def collect():
    files = [parse_name(f) for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))]
    report, reaches = [], {}
    for pt in PATIENTS:
        pf = [f for f in files if f["pt"] == pt]
        cam = CAMERA
        if cam == "auto":
            score = {c: np.mean([np.nanmean(read_dlc(f["path"])[:, BODYPARTS.index(PLOT_MARKER), 2])
                                 for f in pf if f["cam"] == c]) for c in ("L", "R")}
            cam = max(score, key=score.get)
        for cond, _ in CONDITIONS:
            trials = [f for f in pf if f["cam"] == cam and f["cond"] == cond]
            per_trial = []
            for f in trials:
                d = read_dlc(f["path"])
                seg_xy, _ = marker_xy(d, SEG_MARKER)
                plot_xy, _ = marker_xy(d, PLOT_MARKER)
                pk = find_pokes(seg_xy)
                per_trial.append((f, plot_xy, pk))
                report.append((pt, cond, cam, f["trial"], len(pk)))
            # spread N middle pokes across trials, proportional to what each has
            n_keep = N_POKES[pt]
            tot = sum(len(p) for _, _, p in per_trial)
            n_keep = min(n_keep, tot)
            alloc = [int(round(n_keep * len(p) / tot)) for _, _, p in per_trial]
            while sum(alloc) > n_keep: alloc[np.argmax(alloc)] -= 1
            while sum(alloc) < n_keep: alloc[np.argmin(alloc)] += 1
            out = []
            win = int(TARGET_WIN * FPS)
            for (f, xy, pk), a in zip(per_trial, alloc):
                for i in pick_middle(len(pk), a):
                    on, arr, ret = pk[i]
                    out.append(xy[on: min(arr + win, ret) + 1])
            reaches[(pt, cond)] = out
    return reaches, report


def align(reaches):
    """Rest at origin, start->ball rotated to REACH_ANGLE, y up, scaled per patient."""
    aligned = {}
    for pt in PATIENTS:
        lengths = []
        tmp = {}
        for cond, _ in CONDITIONS:
            trs = [np.column_stack([r[:, 0], -r[:, 1]]) for r in reaches[(pt, cond)]]  # y up
            starts = np.array([t[0] for t in trs]).mean(0)
            ends = np.array([t[-1] for t in trs]).mean(0)
            v = ends - starts
            ang = np.deg2rad(REACH_ANGLE) - np.arctan2(v[1], v[0])
            R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            trs = [(t - t[0]) @ R.T for t in trs]
            # mirror so the path bows up-left (like the paper) for every panel
            mid = np.mean([t[len(t) // 2] for t in trs], 0)
            u = np.array([np.cos(np.deg2rad(REACH_ANGLE)), np.sin(np.deg2rad(REACH_ANGLE))])
            if STYLE == "paper" and u[0] * mid[1] - u[1] * mid[0] < 0:
                M = np.outer(u, u) * 2 - np.eye(2)          # reflect across reach axis
                trs = [t @ M.T for t in trs]
            tmp[cond] = trs
            lengths += [np.linalg.norm(t[-1]) for t in trs]
        scale = np.median(lengths)
        for cond in tmp:
            aligned[(pt, cond)] = [t / scale for t in tmp[cond]]
    return aligned


# ---------- figure -----------------------------------------------------------
TABLE, PAD, INK, TXT, PAGE = "#dcdcdc", "#a8a8a8", "#2b2b2b", "#444444", "#f4f0e1"


def draw_row(ax, data, pt, letter, spacing=1.25):
    w = spacing * len(CONDITIONS)
    x0, y0, depth, skew = -0.45, -0.42, 0.62, 0.38
    ax.add_patch(Polygon([[x0, y0], [x0 + w, y0], [x0 + w + skew, y0 + depth],
                          [x0 + skew, y0 + depth]], fc=TABLE, ec="none", zorder=0))
    for i, (cond, label) in enumerate(CONDITIONS):
        ox = i * spacing
        ax.add_patch(Ellipse((ox, 0), 0.24, 0.085, fc=PAD, ec="none", zorder=1))
        for t in data[(pt, cond)]:
            ax.plot(t[:, 0] + ox, t[:, 1], color=INK, lw=0.55, alpha=0.8,
                    solid_joinstyle="round", zorder=2)
        ax.text(ox + 0.05, -0.24, label, ha="center", va="center", fontsize=10,
                color=TXT, zorder=3)
    ax.text(x0 - 0.05, 1.15, letter, fontsize=15, fontweight="bold", va="top")
    ax.set_xlim(x0 - 0.1, w + skew - 0.3)
    ax.set_ylim(y0 - 0.05, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_2d(data):
    nr, nc = len(PATIENTS), len(CONDITIONS)
    fig, axes = plt.subplots(nr, nc, figsize=(1.7 * nc + 0.9, 3.3 * nr),
                             sharex=True, sharey=True, squeeze=False)
    allpts = np.vstack([t for v in data.values() for t in v])
    half = max(0.25, np.abs(allpts[:, 0]).max() * 1.1)
    for r, pt in enumerate(PATIENTS):
        for c, (cond, label) in enumerate(CONDITIONS):
            ax = axes[r, c]
            for t in data[(pt, cond)]:
                ax.plot(t[:, 0], t[:, 1], color=INK, lw=0.7, alpha=0.85,
                        solid_joinstyle="round")
            ax.plot(0, 0, "o", ms=5, color=PAD, zorder=0)
            ax.text(0.97, 0.02, f"n = {len(data[(pt, cond)])}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color="#666")
            ax.set_aspect("equal")
            ax.set_xlim(-half, half)
            ax.set_ylim(-0.08, allpts[:, 1].max() * 1.05)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(labelsize=8, color="#888", labelcolor="#555")
            if r == 0:
                ax.set_title(label, fontsize=11, color=TXT)
            if c == 0:
                ax.set_ylabel("Forward (reach lengths)", fontsize=9, color=TXT)
                ax.text(-0.42, 1.02, "ABCDEFG"[r], transform=ax.transAxes,
                        fontsize=14, fontweight="bold", va="bottom")
            if r == nr - 1:
                ax.set_xlabel("Lateral (reach lengths)", fontsize=9, color=TXT)
    fig.tight_layout()
    return fig


def main():
    reaches, report = collect()
    with open(os.path.join(DATA_DIR, "reach_counts.csv"), "w") as fh:
        fh.write("patient,condition,camera,trial,pokes_found\n")
        for r in report:
            fh.write(",".join(map(str, r)) + "\n")
    for r in report:
        print(*r)
    data = align(reaches)
    if STYLE == "2d":
        fig = plot_2d(data)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(DATA_DIR, f"reach_figure.{ext}"), dpi=300, facecolor="white")
        print("saved reach_figure.png / .pdf")
        return
    fig, axes = plt.subplots(len(PATIENTS), 1, figsize=(5.2, 2.9 * len(PATIENTS)), facecolor=PAGE)
    axes = np.atleast_1d(axes)
    fig.patches.append(Rectangle((0.025, 0.02), 0.95, 0.96, transform=fig.transFigure,
                                 fc="white", ec="none", zorder=-1))
    for ax, pt, letter in zip(axes, PATIENTS, "ABCDEFG"):
        draw_row(ax, data, pt, letter)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.97, bottom=0.03, hspace=0.08)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(DATA_DIR, f"reach_figure.{ext}"), dpi=300,
                    facecolor=fig.get_facecolor())
    print("saved reach_figure.png / .pdf")


if __name__ == "__main__":
    main()
