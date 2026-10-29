#!/usr/bin/env python3
"""Plots and a delta-V budget summary from Ex_FormationFlying_SSO_FEEP.script.

GMAT writes two files per run into its output folder:
    FormationSSO_FEEP_Daily.txt      one row per control cycle (about 1.000 day)
    FormationSSO_FEEP_Summary.txt    delta-V budget and compliance counters

Usage
    python plot_formation_results.py RUN [RUN ...] [options]

RUN is a GMAT output folder, a Daily file, or a Daily/Summary file pair prefix
(NAME_daily.txt + NAME_sum.txt also works). With several RUNs the per-run
figures are written to OUTDIR/<label>/ and a comparison figure to OUTDIR.

    python plot_formation_results.py /path/to/GMAT/output
    python plot_formation_results.py nominal/ high_flux/ --labels Nominal "High flux"
    python plot_formation_results.py run_daily.txt --start-date 2027-01-01 --show

Figures (one PNG each)
    01_compliance     Is the formation inside the requirement? (along-track,
                      normal amplitude, 3-D separation)
    02_dv_budget      Where the delta-V goes and when (by category, over time,
                      per year)
    03_propulsion     Thruster duty, campaigns and propellant use
    04_orbit_health   Altitude, inclination, local time of the node, eccentricity
    05_controller     What the along-track and node controllers did
    06_comparison     Only with 2+ runs: budgets and trends side by side

It also prints the budget table and the compliance verdict, and writes them to
OUTDIR/<label>/budget_summary.md. The budget follows the ESA guidelines
EOP-FM/2024-07-177 (AD10): Table 1 with the beginning-of-life, nominal-operations
and end-of-life contributors, the margin policy of its Sec. 5.6, the residual
(1 %) and uncertainty (2 %) of the tank load of Sec. 5.8 and the iteration of
Sec. 6.5 on the propellant mass. Contributors that GMAT does not simulate
(launcher injection errors, collision avoidance, thruster attitude control) are
inputs: --lie-ip, --lie-oop, --n-cam, --acbt-dv; they are marked "not provided"
while they are zero. The end-of-life lifetime and the separation after release
are checked against ESSB-ST-U-007 (5.4.2.3, 5.3.3.2, 5.3.3.4).

Needs numpy and matplotlib only.
"""
import argparse
import datetime as dt
import math
import os
import sys

import numpy as np
import matplotlib

# ----------------------------------------------------------------------------
# Palette: categorical slots in fixed order, one entity = one colour everywhere.
# Satellites keep slot 1 / slot 2 in every figure; delta-V categories use other
# slots so a colour never means two things.
# ----------------------------------------------------------------------------
THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#8b8a86",
        grid="#e7e6e2", band="#eceae5",
        slot=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
              "#008300", "#6250d6", "#e34948"],
        good="#0ca30c", warn="#fab219", crit="#d03b3b"),
    "dark": dict(
        surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#8b8a86",
        grid="#2d2d2b", band="#262624",
        slot=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
              "#008300", "#9085e9", "#e66767"],
        good="#0ca30c", warn="#fab219", crit="#d03b3b"),
}
C = THEMES["light"]          # replaced in main() by --theme
SAT_COLOR = {1: 0, 2: 1}     # slot index of Sat1 / Sat2
CAT_COLOR = {"tang": 2, "norm": 6}   # delta-V categories: aqua, violet
G0 = 9.81                    # same value as the thrusters in the GMAT script
A_EQ = 6378.137              # km, for altitude = mean SMA - A_EQ


# ----------------------------------------------------------------------------
# Reading GMAT report files
# ----------------------------------------------------------------------------
def read_table(path):
    with open(path) as f:
        lines = [ln.split() for ln in f if ln.strip()]
    head = lines[0]
    cols = {k: [] for k in head}
    for row in lines[1:]:
        for k, v in zip(head, row):
            cols[k].append(float(v))
    return {k: np.array(v) for k, v in cols.items()}


def read_summary(path):
    """The summary is several (header line, value line) pairs."""
    with open(path) as f:
        lines = [ln.split() for ln in f if ln.strip()]
    out = {}
    for i in range(0, len(lines) - 1, 2):
        for k, v in zip(lines[i], lines[i + 1]):
            out[k] = float(v)
    return out


def find_files(arg):
    """Return (daily_path, summary_path_or_None, default_label)."""
    if os.path.isdir(arg):
        d = os.path.join(arg, "FormationSSO_FEEP_Daily.txt")
        s = os.path.join(arg, "FormationSSO_FEEP_Summary.txt")
        return d, (s if os.path.exists(s) else None), os.path.basename(
            os.path.normpath(arg)) or "run"
    base = os.path.basename(arg)
    low = base.lower()
    cands = []
    if "daily" in low:
        for old, new in (("Daily", "Summary"), ("daily", "summary"),
                         ("daily", "sum"), ("Daily", "Sum")):
            cands.append(os.path.join(os.path.dirname(arg), base.replace(old, new)))
    else:                                    # a prefix
        for suf in (("_daily.txt", "_sum.txt"), ("_Daily.txt", "_Summary.txt")):
            if os.path.exists(arg + suf[0]):
                return arg + suf[0], (arg + suf[1] if os.path.exists(arg + suf[1])
                                      else None), base
    summ = next((c for c in cands if os.path.exists(c) and c != arg), None)
    label = base.replace("_Daily.txt", "").replace("_daily.txt", "").replace(".txt", "")
    return arg, summ, label


# ----------------------------------------------------------------------------
# Derived quantities
# ----------------------------------------------------------------------------
class Run:
    def __init__(self, daily, summary, label, args):
        self.label = label
        self.d = d = read_table(daily)
        # The last row only measures and logs: its controller outputs were never
        # applied and its 15-orbit averages cover a partial cycle. Blank them.
        for k in ("aAvg1", "aAvg2", "iAvg1", "iAvg2", "ANamp", "dutyT1", "dutyT2",
                  "dutyN1", "dutyN2", "dACmd", "diCmd", "TdotM"):
            if k in d:
                d[k] = d[k].astype(float)
                d[k][-1] = np.nan
        self.s = read_summary(summary) if summary else {}
        self.args = args
        self.t = d["tDays"]
        self.start = dt.datetime.strptime(args.start_date, "%Y-%m-%d")
        self.dates = [self.start + dt.timedelta(days=float(x)) for x in self.t]
        self.years = self.t / 365.25
        self.controlled = (d["dvT1"][-1] + d["dvN1"][-1] + d["dvT2"][-1]
                           + d["dvN2"][-1]) > 0
        # delta-V (m/s), cumulative at each cycle start
        self.dv = {1: dict(tang=d["dvT1"], norm=d["dvN1"]),
                   2: dict(tang=d["dvT2"], norm=d["dvN2"])}
        self.dv_final = {s: {c: float(v[-1]) for c, v in self.dv[s].items()}
                         for s in (1, 2)}
        self.dv_total = sum(v for s in (1, 2) for v in self.dv_final[s].values())
        # Acquisition (both satellites start at the same position): rows with
        # acq = 1 are the phasing manoeuvre, compliance is judged after it.
        self.acq = d["acq"] if "acq" in d else np.zeros(len(self.t))
        after = np.where(self.acq < 0.5)[0]
        self.acq_done = bool(after.size)
        self.i_ops = int(after[0]) if after.size else len(self.t) - 1
        self.t_acq = float(self.t[self.i_ops]) if self.i_ops > 0 else 0.0
        self.dv_acq = {s: {c: float(v[self.i_ops]) if self.i_ops > 0 else 0.0
                           for c, v in self.dv[s].items()} for s in (1, 2)}
        self.dv_acq_total = sum(v for s in (1, 2) for v in self.dv_acq[s].values())
        self.prop_g = {1: (d["mass1"][0] - d["mass1"]) * 1000,
                       2: (d["mass2"][0] - d["mass2"]) * 1000}
        # relative geometry
        self.T, self.N, self.R, self.rng, self.AN = (
            d["Trel"], d["Nrel"], d["Rrel"], d["Rng"], d["ANamp"])
        # orbit health
        self.alt = {1: d["aAvg1"] - A_EQ, 2: d["aAvg2"] - A_EQ}
        self.alt_dev = {s: (self.alt[s] - self.alt[s][0]) * 1000 for s in (1, 2)}  # m
        self.inc = {1: d["iAvg1"], 2: d["iAvg2"]}
        self.ltan_dev = {1: self.ltan_minutes(d["raanM1"]),
                         2: self.ltan_minutes(d["raanM2"])}
        dr = d["raanM2"] - d["raanM1"]
        self.draan = (dr + 180) % 360 - 180
        # local time of the node of the centre of the pair (the acquisition moves
        # the two nodes apart symmetrically, so each satellite alone swings by half
        # of the node offset)
        self.ltan_centre = self.ltan_minutes(d["raanM1"] + self.draan / 2)

    def ltan_minutes(self, raan):
        """Local time of the ascending node relative to its start, minutes.
        Mean sun, so the equation of time does not show up as a false drift."""
        j2000 = dt.datetime(2000, 1, 1, 12)
        days = (self.start - j2000).total_seconds() / 86400 + self.t
        sun_ra = (280.460 + 0.9856474 * days) % 360
        ltan_h = 12 + ((raan - sun_ra + 180) % 360 - 180) / 15
        dev = (ltan_h - ltan_h[0] + 12) % 24 - 12
        return dev * 60

    def ltan_start(self):
        j2000 = dt.datetime(2000, 1, 1, 12)
        days = (self.start - j2000).total_seconds() / 86400
        sun_ra = (280.460 + 0.9856474 * days) % 360
        h = (12 + ((self.d["raanM1"][0] - sun_ra + 180) % 360 - 180) / 15) % 24
        return h

    def at_day(self, series, day):
        return float(np.interp(day, self.t, series))

    def per_year_dv(self):
        """Delta-V per mission year, [year][category] formation total (m/s)."""
        n_years = int(math.ceil(self.t[-1] / 365.25 - 1e-9))
        edges = [min(k * 365.25, self.t[-1]) for k in range(n_years + 1)]
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            row = {}
            for cat in ("tang", "norm"):
                tot = 0.0
                for s in (1, 2):
                    tot += self.at_day(self.dv[s][cat], b) - self.at_day(self.dv[s][cat], a)
                row[cat] = tot
            out.append(row)
        return out


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def style():
    matplotlib.rcParams.update({
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
        "savefig.facecolor": C["surface"], "text.color": C["ink"],
        "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"],
        "ytick.color": C["ink2"], "axes.edgecolor": C["grid"],
        "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": C["grid"],
        "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
        "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlecolor": C["ink"],
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.solid_capstyle": "round", "figure.dpi": 100,
    })


def col(i):
    return C["slot"][i]


def sat_col(s):
    return col(SAT_COLOR[s])


def shade_acq(ax, run):
    """Grey span over the acquisition (both satellites start at the same place)."""
    if run.i_ops > 0:
        ax.axvspan(run.dates[0], run.dates[run.i_ops], color=C["muted"], alpha=0.13,
                   lw=0, zorder=0)
        ax.text(run.dates[run.i_ops], 0.10, "acquisition ends ", ha="right", va="bottom",
                transform=ax.get_xaxis_transform(), fontsize=7.5, color=C["ink2"])


def year_axis(ax, run, show_labels=True):
    import matplotlib.dates as mdates
    shade_acq(ax, run)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(run.dates[0], run.dates[-1])
    if not show_labels:
        ax.tick_params(labelbottom=False)


def hline(ax, y, label=None, color=None, side="right", lw=0.9):
    color = color or C["muted"]
    ax.axhline(y, color=color, lw=lw, zorder=1)
    if label:
        x, ha = (0.995, "right") if side == "right" else (0.005, "left")
        ax.text(x, y, label, transform=ax.get_yaxis_transform(), ha=ha,
                va="bottom", fontsize=8, color=C["ink2"])


def band(ax, lo, hi, label=None):
    ax.axhspan(lo, hi, color=C["band"], zorder=0, lw=0)
    if label:
        ax.text(0.005, hi, label, transform=ax.get_yaxis_transform(),
                ha="left", va="top", fontsize=8, color=C["ink2"])


def legend_top(ax, **kw):
    """Legend above the axes, right aligned, so it never covers data."""
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4,
              borderaxespad=0.2, handlelength=1.6, **kw)


def robust_ylim(ax, series, skip=8, pad=1.5, floor=0.0, symmetric=False):
    """Axis limits from the settled data (ignoring the start-up transient and
    the blanked last row). Returns True if some points fall outside."""
    v = np.concatenate([np.asarray(x, float)[skip:] for x in series])
    v = v[~np.isnan(v)]
    if v.size == 0:
        return False
    lo, hi = float(v.min()), float(v.max())
    span = max(hi - lo, floor)
    mid = (hi + lo) / 2
    if symmetric:
        m = max(abs(lo), abs(hi), floor / 2) * pad
        lo, hi = -m, m
    else:
        lo, hi = mid - span / 2 * pad, mid + span / 2 * pad
    ax.set_ylim(lo, hi)
    full = np.concatenate([np.asarray(x, float) for x in series])
    full = full[~np.isnan(full)]
    return bool((full > hi).any() or (full < lo).any())


def clip_note(ax, text="start-up transient clipped"):
    ax.text(0.005, 0.03, text, transform=ax.transAxes, fontsize=8, color=C["muted"],
            va="bottom")


def end_label(ax, run, y, text, color):
    ax.annotate(text, (run.dates[-1], y), xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=C["ink"],
                bbox=dict(boxstyle="square,pad=0.0", fc="none", ec="none"))
    ax.plot([run.dates[-1]], [y], "o", ms=5, color=color,
            mec=C["surface"], mew=1.5, clip_on=False, zorder=5)


def status_chip(fig, x, y, ok, text):
    sym, colr = ("✓", C["good"]) if ok else ("✗", C["crit"])
    fig.text(x, y, sym, color=colr, fontsize=12, fontweight="bold",
             ha="left", va="center")
    fig.text(x + 0.016, y, text, color=C["ink"], fontsize=9.5, ha="left",
             va="center")


def finish(fig, path, show):
    fig.savefig(path, dpi=150)
    if show:
        return
    matplotlib.pyplot.close(fig)


# ----------------------------------------------------------------------------
# Compliance numbers
# ----------------------------------------------------------------------------
def compliance(run):
    a = run.args
    i0 = run.i_ops if run.acq_done else 0   # judged after the acquisition
    T = run.T[i0:]
    out = {}
    inside = (T >= a.t_min) & (T <= a.t_max)
    out["T_min"], out["T_max"] = float(np.nanmin(T)), float(np.nanmax(T))
    out["T_margin"] = float(min(out["T_min"] - a.t_min, a.t_max - out["T_max"]))
    out["T_in_frac"] = float(inside.mean())
    out["T_ok"] = bool(inside.all())
    lo, hi = a.n_amp * (1 - a.n_tol), a.n_amp * (1 + a.n_tol)
    AN = run.AN[i0:]
    ok = ~np.isnan(AN)
    n_in = (AN >= lo) & (AN <= hi)
    out["AN_min"], out["AN_max"] = float(np.nanmin(AN)), float(np.nanmax(AN))
    out["AN_in_frac"] = float(n_in[ok].mean())
    out["AN_ok"] = bool(n_in[ok].all())
    out["AN_lo"], out["AN_hi"] = lo, hi
    out["rng_min"] = float(run.rng[i0:].min())
    out["rng_ok"] = out["rng_min"] >= a.min_range
    out["ltan_max"] = float(np.abs(run.ltan_centre).max())
    out["inc_change"] = float(run.inc[1][-2] - run.inc[1][0])
    out["alt_change"] = float(run.alt[1][-2] - run.alt[1][0])
    out["ok"] = out["T_ok"] and out["AN_ok"] and out["rng_ok"]
    return out


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_compliance(run, comp, outdir, show):
    a = run.args
    fig, axs = plt().subplots(3, 1, figsize=(10, 9.4), sharex=True,
                              gridspec_kw=dict(hspace=0.16))
    fig.subplots_adjust(top=0.79, left=0.085, right=0.9, bottom=0.07)
    fig.text(0.085, 0.965, f"Formation compliance – {run.label}",
             fontsize=14, fontweight="bold", ha="left")
    fig.text(0.085, 0.94, "Sat2 relative to Sat1 in the RTN frame, sampled "
             "once per control cycle (15 orbits)", color=C["ink2"], fontsize=9.5)
    status_chip(fig, 0.085, 0.905, comp["T_ok"],
                f"Along-track {comp['T_min']:.0f}–{comp['T_max']:.0f} km, "
                f"{comp['T_margin']:.0f} km margin to the "
                f"{a.t_min:.0f}–{a.t_max:.0f} km limits")
    status_chip(fig, 0.085, 0.878, comp["AN_ok"],
                f"Normal amplitude {comp['AN_min']:.0f}–{comp['AN_max']:.0f} km "
                f"(requirement {a.n_amp:.0f} km ±{a.n_tol*100:.0f} %)")
    status_chip(fig, 0.085, 0.851, comp["rng_ok"],
                f"Closest approach {comp['rng_min']:.0f} km "
                f"(minimum allowed {a.min_range:.0f} km)")
    if run.i_ops > 0:
        fig.text(0.9, 0.851, f"judged after the acquisition (day {run.t_acq:.0f})",
                 color=C["ink2"], fontsize=8.5, ha="right", va="center")

    ax = axs[0]
    band(ax, a.t_min, a.t_max, "requirement")
    ax.plot(run.dates, run.T, color=col(0), lw=1.6)
    hline(ax, a.t_min, f"{a.t_min:.0f} km")
    hline(ax, a.t_max, f"{a.t_max:.0f} km")
    ax.set_ylabel("Along-track T (km)")
    ax.set_title("Along-track separation")
    ax.set_ylim(min(0, run.T.min() * 1.05), max(a.t_max * 1.08, run.T.max() * 1.05))
    shade_acq(ax, run)
    end_label(ax, run, run.T[-1], f"{run.T[-1]:.0f}", col(0))

    ax = axs[1]
    band(ax, comp["AN_lo"], comp["AN_hi"],
         f"{a.n_amp:.0f} km ±{a.n_tol*100:.0f} %")
    ax.plot(run.dates, run.AN, color=col(0), lw=1.6)
    hline(ax, a.n_amp)
    ax.set_ylabel("Normal amplitude (km)")
    ax.set_title("Normal separation amplitude  (a·√(Δi² + (sin i·ΔΩ)²))")
    span = max(comp["AN_max"] - a.n_amp, a.n_amp - comp["AN_min"], a.n_amp * a.n_tol)
    ax.set_ylim(a.n_amp - 1.5 * span, a.n_amp + 1.5 * span)
    if run.i_ops > 0:               # show the build-up from zero as well
        ax.set_ylim(0, a.n_amp * 1.15)
    shade_acq(ax, run)
    last = np.where(~np.isnan(run.AN))[0][-1]
    ax.plot([run.dates[last]], [run.AN[last]], "o", ms=5, color=col(0),
            mec=C["surface"], mew=1.5, clip_on=False, zorder=5)
    ax.annotate(f"{run.AN[last]:.1f}", (run.dates[last], run.AN[last]),
                xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.5)

    ax = axs[2]
    ax.plot(run.dates, run.rng, color=col(0), lw=1.6)
    hline(ax, a.min_range, f"minimum allowed {a.min_range:.0f} km", color=C["crit"])
    imin = run.i_ops + int(np.argmin(run.rng[run.i_ops:]))
    ax.plot([run.dates[imin]], [run.rng[imin]], "o", ms=6, color=col(0),
            mec=C["surface"], mew=1.5, zorder=5)
    late = imin > 0.7 * len(run.rng)
    ax.annotate(f"closest {run.rng[imin]:.0f} km", (run.dates[imin], run.rng[imin]),
                xytext=(-8, 12) if late else (8, -14), textcoords="offset points",
                ha="right" if late else "left", fontsize=8.5)
    ax.set_ylabel("3-D separation (km)")
    ax.set_title("Distance between the satellites")
    ax.set_ylim(0, max(run.rng.max() * 1.1, a.min_range * 2))
    year_axis(ax, run)
    if not run.controlled:
        fig.text(0.9, 0.965, "UNCONTROLLED (free drift)", color=C["crit"],
                 fontsize=10, fontweight="bold", ha="right")
    finish(fig, os.path.join(outdir, "01_compliance.png"), show)


def fig_dv(run, outdir, show):
    a = run.args
    fig = plt().figure(figsize=(11, 9.2))
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.93, top=0.86, bottom=0.075,
                          hspace=0.34, wspace=0.26)
    fig.text(0.08, 0.955, f"Delta-V budget – {run.label}", fontsize=14,
             fontweight="bold", ha="left")
    tot = run.dv_total
    marg = tot * (1 + a.margin)
    marg_txt = (f"  →  {marg:.1f} m/s with {a.margin*100:.0f} % margin" if a.margin else "")
    fig.text(0.08, 0.92, f"Formation total {tot:.1f} m/s{marg_txt}  "
             f"(Sat1 {sum(run.dv_final[1].values()):.1f}, "
             f"Sat2 {sum(run.dv_final[2].values()):.1f} m/s)",
             fontsize=10, color=C["ink"], ha="left")
    fig.text(0.08, 0.895, "tangential = altitude + along-track control;  normal = "
             "inclination / node control" + (
                 f";  acquisition (days 0–{run.t_acq:.0f}) = {run.dv_acq_total:.1f} m/s"
                 if run.i_ops > 0 else ""), fontsize=9, color=C["ink2"], ha="left")

    # (a) stacked bars per satellite + formation
    ax = fig.add_subplot(gs[0, 0])
    groups = ["Sat1", "Sat2", "Formation"]
    tang = [run.dv_final[1]["tang"], run.dv_final[2]["tang"],
            run.dv_final[1]["tang"] + run.dv_final[2]["tang"]]
    norm = [run.dv_final[1]["norm"], run.dv_final[2]["norm"],
            run.dv_final[1]["norm"] + run.dv_final[2]["norm"]]
    x = np.arange(3)
    w = 0.52
    gap = 0.012 * max(max(t + n for t, n in zip(tang, norm)), 1)
    ax.bar(x, tang, w, color=col(CAT_COLOR["tang"]), label="tangential")
    ax.bar(x, norm, w, bottom=[t + gap for t in tang], color=col(CAT_COLOR["norm"]),
           label="normal")
    ymax = max(t + n for t, n in zip(tang, norm))
    for xi, t, n in zip(x, tang, norm):
        ax.text(xi, t + n + gap + ymax * 0.02, f"{t + n:.1f}", ha="center",
                va="bottom", fontsize=10, fontweight="bold")
        if t > ymax * 0.07:
            ax.text(xi, t / 2, f"{t:.1f}", ha="center", va="center",
                    fontsize=8.5, color="#0b0b0b")
        if n > ymax * 0.07:
            ax.text(xi, t + gap + n / 2, f"{n:.1f}", ha="center", va="center",
                    fontsize=8.5, color="white")
    ax.set_xticks(x, groups)
    ax.set_ylabel("Δv (m/s)")
    ax.set_title("5-year total by category")
    ax.set_ylim(0, ymax * 1.18)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")

    # (b) cumulative over time
    ax = fig.add_subplot(gs[0, 1])
    tot1 = run.dv[1]["tang"] + run.dv[1]["norm"]
    tot2 = run.dv[2]["tang"] + run.dv[2]["norm"]
    ax.plot(run.dates, tot1 + tot2, color=C["ink"], lw=1.6, label="Formation")
    ax.plot(run.dates, tot1, color=sat_col(1), lw=1.6, label="Sat1")
    ax.plot(run.dates, tot2, color=sat_col(2), lw=1.6, label="Sat2")
    ax.set_ylabel("Cumulative Δv (m/s)")
    ax.set_title("Build-up over the mission")
    year_axis(ax, run)
    ax.legend(loc="upper left")

    # (c) per mission year
    ax = fig.add_subplot(gs[1, 0])
    yrs = run.per_year_dv()
    xs = np.arange(len(yrs))
    t_ = np.array([r["tang"] for r in yrs])
    n_ = np.array([r["norm"] for r in yrs])
    ymax = max((t_ + n_).max(), 1e-9)
    gap = 0.012 * ymax
    ax.bar(xs, t_, 0.55, color=col(CAT_COLOR["tang"]), label="tangential")
    ax.bar(xs, n_, 0.55, bottom=t_ + gap, color=col(CAT_COLOR["norm"]), label="normal")
    for xi, t, n in zip(xs, t_, n_):
        ax.text(xi, t + n + gap + ymax * 0.02, f"{t + n:.1f}", ha="center",
                va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(xs, [f"Year {i + 1}" for i in xs])
    ax.set_ylabel("Formation Δv (m/s)")
    ax.set_title("Per mission year")
    ax.set_ylim(0, ymax * 1.38)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=2)

    # (d) propellant
    ax = fig.add_subplot(gs[1, 1])
    m0 = {1: run.d["mass1"][0], 2: run.d["mass2"][0]}
    need = {s: m0[s] * (1 - math.exp(-sum(run.dv_final[s].values()) * (1 + a.margin)
                                      / (a.isp * G0))) * 1000 for s in (1, 2)}
    ax.plot(run.dates, run.prop_g[1], color=sat_col(1), lw=1.6, label="Sat1")
    ax.plot(run.dates, run.prop_g[2], color=sat_col(2), lw=1.6, label="Sat2")
    ax.set_ylabel("Propellant used (g)")
    ax.set_title(f"Propellant (Isp {a.isp:.0f} s)")
    year_axis(ax, run)
    ax.legend(loc="upper left")
    ax.text(0.99, 0.04, "Propellant to load, with %.0f %% margin:\nSat1 %.0f g  •  Sat2 %.0f g"
            % (a.margin * 100, need[1], need[2]), transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8.5, color=C["ink2"])
    finish(fig, os.path.join(outdir, "02_dv_budget.png"), show)


def fig_propulsion(run, outdir, show):
    a = run.args
    d = run.d
    fig, axs = plt().subplots(3, 1, figsize=(10, 9), sharex=True,
                              gridspec_kw=dict(hspace=0.18))
    fig.subplots_adjust(top=0.88, left=0.085, right=0.9, bottom=0.07)
    fig.text(0.085, 0.955, f"Thruster usage – {run.label}", fontsize=14,
             fontweight="bold", ha="left")
    fig.text(0.085, 0.925, f"Duty is the fraction of the {a.thrust*1000:.2f} mN "
             "maximum thrust; each cycle is one continuous burn setting.",
             color=C["ink2"], fontsize=9.5)

    ax = axs[0]
    ax.plot(run.dates, d["dutyT1"] * 100, color=sat_col(1), lw=1.2, label="Sat1")
    ax.plot(run.dates, d["dutyT2"] * 100, color=sat_col(2), lw=1.2, label="Sat2")
    ax.set_ylabel("Duty (%)")
    ax.set_title("Prograde thruster (drag make-up, along-track)")
    legend_top(ax)
    mx = max(np.nanmax(d["dutyT1"]), np.nanmax(d["dutyT2"])) * 100
    avg = (np.nanmean(d["dutyT1"]) + np.nanmean(d["dutyT2"])) / 2 * 100
    ax.text(0.01, 0.94, f"peak {mx:.2f} % • mean {avg:.2f} % of full thrust",
            transform=ax.transAxes, va="top", fontsize=9, color=C["ink2"])
    if robust_ylim(ax, [d["dutyT1"] * 100, d["dutyT2"] * 100], floor=0.05):
        clip_note(ax, "start-up transient (first guess of the drag) clipped")
    ax.set_ylim(bottom=0)

    ax = axs[1]
    for s, dn in ((1, d["dutyN1"]), (2, d["dutyN2"])):
        on = dn > 0
        ax.vlines([x for x, o in zip(run.dates, on) if o], 0,
                  [v * 100 for v, o in zip(dn, on) if o], color=sat_col(s),
                  lw=1.6, label=f"Sat{s}", alpha=0.95 if s == 1 else 0.8)
    n1 = int(np.nansum(d["dutyN1"] > 0))
    n2 = int(np.nansum(d["dutyN2"] > 0))
    ax.set_ylabel("Duty (%)")
    ax.set_title("Out-of-plane thrusters (inclination / node campaigns)")
    ax.set_ylim(0, 105)
    legend_top(ax)
    ax.text(0.01, 0.94, f"burn cycles: Sat1 {n1} • Sat2 {n2}",
            transform=ax.transAxes, va="top", fontsize=9, color=C["ink2"])

    ax = axs[2]
    # full-thrust equivalent hours from delta-V: dv / (F/m)
    for s, m in ((1, d["mass1"]), (2, d["mass2"])):
        dvt = run.dv[s]["tang"] + run.dv[s]["norm"]
        hours = np.cumsum(np.r_[0, np.diff(dvt) / (a.thrust / m[:-1])]) / 3600
        ax.plot(run.dates, hours, color=sat_col(s), lw=1.6, label=f"Sat{s}")
        end_label(ax, run, hours[-1], f"{hours[-1]:.0f} h", sat_col(s))
    ax.set_ylabel("Hours at full thrust")
    ax.set_title("Cumulative firing time (full-thrust equivalent)")
    year_axis(ax, run)
    legend_top(ax)
    finish(fig, os.path.join(outdir, "03_propulsion.png"), show)


def fig_orbit(run, outdir, show):
    fig, axs = plt().subplots(2, 2, figsize=(11, 8), sharex=True,
                              gridspec_kw=dict(hspace=0.22, wspace=0.26))
    fig.subplots_adjust(top=0.85, left=0.08, right=0.95, bottom=0.07)
    fig.text(0.08, 0.955, f"Orbit health – {run.label}", fontsize=14,
             fontweight="bold", ha="left")
    fig.text(0.08, 0.925, "Mean (Brouwer-Lyddane) elements, averaged over each "
             "15-orbit cycle, true-of-date frame", color=C["ink2"], fontsize=9.5)

    ax = axs[0, 0]
    for s in (1, 2):
        ax.plot(run.dates, run.alt_dev[s], color=sat_col(s), lw=1.4, label=f"Sat{s}")
    ax.set_ylabel("Mean altitude vs start (m)")
    ax.set_title("Altitude hold")
    legend_top(ax)
    hline(ax, 0)
    clipped = robust_ylim(ax, [run.alt_dev[1], run.alt_dev[2]], skip=30, floor=4,
                          symmetric=True, pad=1.6)
    settled = max(np.nanmax(np.abs(run.alt_dev[s][30:])) for s in (1, 2))
    ax.text(0.01, 0.04, f"after the first month: within ±{settled:.1f} m"
            + ("  (start-up transient clipped)" if clipped else ""),
            transform=ax.transAxes, fontsize=8.5, color=C["ink2"])

    ax = axs[0, 1]
    for s in (1, 2):
        ax.plot(run.dates, run.inc[s], color=sat_col(s), lw=1.6, label=f"Sat{s}")
    ax.set_ylabel("Inclination (deg)")
    ax.set_title("Mean inclination")
    legend_top(ax)
    ax.ticklabel_format(useOffset=False, axis="y")
    ax.text(0.01, 0.04, f"Sat1 change {run.inc[1][-2]-run.inc[1][0]:+.3f} deg over the mission",
            transform=ax.transAxes, fontsize=8.5, color=C["ink2"])

    ax = axs[1, 0]
    for s in (1, 2):
        ax.plot(run.dates, run.ltan_dev[s], color=sat_col(s), lw=1.6, label=f"Sat{s}")
    h = run.ltan_start()
    ax.set_ylabel("LTAN change (min)")
    ax.set_title(f"LTAN (start {int(h):02d}:{int(round((h % 1) * 60)):02d})")
    hline(ax, 0)
    legend_top(ax)
    ax.text(0.01, 0.04, "assumes the start date given with --start-date",
            transform=ax.transAxes, fontsize=8, color=C["muted"])

    ax = axs[1, 1]
    for s, k in ((1, "eM1"), (2, "eM2")):
        ax.plot(run.dates, run.d[k] * 1e3, color=sat_col(s), lw=1.6, label=f"Sat{s}")
    ax.set_ylabel("Eccentricity (×10⁻³)")
    ax.set_title("Mean eccentricity")
    legend_top(ax)
    for a_ in axs[1]:
        year_axis(a_, run)
    for a_ in axs[0]:
        year_axis(a_, run, show_labels=False)
    finish(fig, os.path.join(outdir, "04_orbit_health.png"), show)


def fig_controller(run, outdir, show):
    a = run.args
    d = run.d
    fig, axs = plt().subplots(4, 1, figsize=(10, 11), sharex=True,
                              gridspec_kw=dict(hspace=0.2))
    fig.subplots_adjust(top=0.91, left=0.09, right=0.93, bottom=0.05)
    fig.text(0.09, 0.965, f"What the controllers did – {run.label}",
             fontsize=14, fontweight="bold", ha="left")
    fig.text(0.09, 0.94, "Use this to tune the gains and limits in the "
             "Design parameters block of the GMAT script", color=C["ink2"],
             fontsize=9.5)

    ax = axs[0]
    ax.plot(run.dates, d["TdotM"], color=col(0), lw=1.2)
    hline(ax, 0)
    ax.set_ylabel("km/day")
    ax.set_title("Measured along-track drift rate (change of T per day)")
    if robust_ylim(ax, [d["TdotM"]], skip=8, floor=0.4, symmetric=True, pad=1.5):
        clip_note(ax)

    ax = axs[1]
    da = (d["aAvg2"] - d["aAvg1"]) * 1000
    ax.plot(run.dates, da, color=col(0), lw=1.4, label="Sat2 − Sat1 mean SMA")
    ax.plot(run.dates, d["dACmd"] * 1000, color=col(3), lw=1.4,
            label="commanded offset (drift reversal)")
    hline(ax, 0)
    ax.set_ylabel("metres")
    ax.set_title("Relative semi-major axis")
    legend_top(ax)
    if robust_ylim(ax, [da, d["dACmd"] * 1000], skip=8, floor=2, symmetric=True, pad=1.6):
        clip_note(ax)

    ax = axs[2]
    dr0 = run.draan[0]
    km = (run.draan - dr0) * math.pi / 180 * np.nan_to_num(d["aAvg1"], nan=6978.0) \
        * np.sin(np.radians(np.nan_to_num(d["iAvg1"], nan=97.8)))
    ax.plot(run.dates, km, color=col(0), lw=1.4)
    hline(ax, 0)
    ax.set_ylabel("km")
    ax.set_title("Drift of the RAAN offset, as normal amplitude (ΔΩ·a·sin i)")
    ax2 = ax
    diff = (d["iAvg2"] - d["iAvg1"]) * 1000
    ax2.text(0.01, 0.92, "steps are the differential-inclination trims", transform=ax.transAxes,
             va="top", fontsize=8.5, color=C["ink2"])

    ax = axs[3]
    ax.plot(run.dates, diff, color=col(0), lw=1.4, label="inclination Sat2 − Sat1")
    if "diCmd" in d:
        ax.plot(run.dates, d["diCmd"] * 1000, color=col(3), lw=1.4,
                label="commanded trim")
    hline(ax, 0)
    ax.set_ylabel("millidegrees")
    ax.set_title("Differential inclination")
    legend_top(ax)
    year_axis(ax, run)
    finish(fig, os.path.join(outdir, "05_controller.png"), show)


def fig_compare(runs, outdir, show):
    a = runs[0].args
    ctrl = [r for r in runs if r.controlled]
    fig = plt().figure(figsize=(11, 9.4))
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.94, top=0.83, bottom=0.07,
                          hspace=0.36, wspace=0.26)
    fig.text(0.08, 0.955, "Scenario comparison", fontsize=14, fontweight="bold",
             ha="left")
    fig.text(0.08, 0.93, "Colour = scenario, the same in every panel",
             color=C["ink2"], fontsize=9.5)
    # controlled runs take the categorical slots in order, the reference is grey
    color = {}
    k = 0
    for r in runs:
        if r.controlled:
            color[id(r)] = col(k)
            k += 1
        else:
            color[id(r)] = C["muted"]

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=color[id(r)], lw=2.0,
                      label=r.label + ("" if r.controlled else " (reference)"))
               for r in runs]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 0.915),
               ncol=min(len(runs), 4), frameon=False, fontsize=9.5,
               handlelength=1.8, columnspacing=1.6)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(ctrl))
    tang = np.array([sum(r.dv_final[s_]["tang"] for s_ in (1, 2)) for r in ctrl])
    norm = np.array([sum(r.dv_final[s_]["norm"] for s_ in (1, 2)) for r in ctrl])
    ymax = max((tang + norm).max(), 1e-9)
    gap = 0.012 * ymax
    ax.bar(x, tang, 0.5, color=col(CAT_COLOR["tang"]), label="tangential")
    ax.bar(x, norm, 0.5, bottom=tang + gap, color=col(CAT_COLOR["norm"]), label="normal")
    for xi, t, nn in zip(x, tang, norm):
        ax.text(xi, t + nn + gap + ymax * 0.02, f"{t + nn:.1f}", ha="center",
                va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x, [r.label for r in ctrl], rotation=15 if len(ctrl) > 3 else 0)
    ax.set_ylabel("Formation \u0394v (m/s)")
    ax.set_title("Total delta-V")
    ax.set_ylim(0, ymax * 1.2)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")

    ax = fig.add_subplot(gs[0, 1])
    for r in ctrl:
        tot = sum(r.dv[s_][k_] for s_ in (1, 2) for k_ in ("tang", "norm"))
        ax.plot(r.dates, tot, color=color[id(r)], lw=1.6, label=r.label)
    ax.set_ylabel("Cumulative \u0394v (m/s)")
    ax.set_title("Build-up over the mission")
    year_axis(ax, runs[0])

    ax = fig.add_subplot(gs[1, 0])
    band(ax, a.t_min, a.t_max, "requirement")
    for r in sorted(runs, key=lambda r: r.controlled):
        ax.plot(r.dates, r.T, color=color[id(r)], lw=1.5 if r.controlled else 1.0,
                label=r.label, zorder=3 if r.controlled else 2)
    top = max([a.t_max] + [np.nanmax(r.T) for r in ctrl]) * 1.1
    ax.set_ylim(0, top)
    ax.set_ylabel("Along-track T (km)")
    ax.set_title("Along-track separation")
    if any(not r.controlled for r in runs):
        ax.text(0.99, 0.97, "grey: uncontrolled, leaves the panel", ha="right",
                va="top", transform=ax.transAxes, fontsize=8, color=C["ink2"])
    year_axis(ax, runs[0])

    ax = fig.add_subplot(gs[1, 1])
    lo, hi = a.n_amp * (1 - a.n_tol), a.n_amp * (1 + a.n_tol)
    band(ax, lo, hi, f"{a.n_amp:.0f} km \u00b1{a.n_tol*100:.0f} %")
    for r in sorted(runs, key=lambda r: r.controlled):
        ax.plot(r.dates, r.AN, color=color[id(r)], lw=1.5 if r.controlled else 1.0,
                label=r.label, zorder=3 if r.controlled else 2)
    vmax = max([hi] + [np.nanmax(r.AN) for r in ctrl])
    vmin = min([lo] + [np.nanmin(r.AN) for r in ctrl])
    pad = 0.25 * (vmax - vmin)
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_ylabel("Normal amplitude (km)")
    ax.set_title("Normal separation amplitude")
    year_axis(ax, runs[0])
    finish(fig, os.path.join(outdir, "06_comparison.png"), show)


# ----------------------------------------------------------------------------
# AD10 (EOP-FM/2024-07-177) delta-V and propellant mass budget
# ----------------------------------------------------------------------------
def ad10_budget(run):
    """Table 1 of AD10 for both satellites. Returns (rows, totals, notes)."""
    a = run.args
    s = run.s
    g = lambda k: float(s.get(k, 0.0))
    n_mean = math.sqrt(398600.4415 / 6938.137 ** 3)             # rad/s, ~ operational orbit
    if "aAvg1" in run.d:
        n_mean = math.sqrt(398600.4415 / float(np.nanmean(run.d["aAvg1"])) ** 3)
    cam_each = n_mean * 100.0               # m/s: +100 m SMA then -100 m (AD10 6.2.4)
    acq = {1: g("dvAcqT1") + g("dvAcqN1"), 2: g("dvAcqT2") + g("dvAcqN2")}
    tang = {1: run.dv_final[1]["tang"] - g("dvAcqT1"), 2: run.dv_final[2]["tang"] - g("dvAcqT2")}
    oop = {1: g("dvNo1"), 2: g("dvNo2")}
    form = {1: g("dvNf1"), 2: g("dvNf2")}
    reacq = {k: a.n_reacq * acq[k] for k in (1, 2)}
    cam = {k: a.n_cam * 4 * cam_each for k in (1, 2)}      # factor 4 on the number of CAM
    clr = {1: g("dvClr1"), 2: g("dvClr2")}
    disp = {1: g("dvDisp1"), 2: g("dvDisp2")}
    # (phase, contributor, dv Sat1, dv Sat2, propellant factor, source)
    rows = [
        ("BOL", "Launcher injection errors, in-plane", a.lie_ip, a.lie_ip, 1, "input" if a.lie_ip else "not provided"),
        ("BOL", "Launcher injection errors, out-of-plane", a.lie_oop, a.lie_oop, 1, "input" if a.lie_oop else "not provided"),
        ("BOL", "Orbit acquisition (formation phasing)", acq[1], acq[2], 1, "GMAT"),
        ("OP", "Orbit transfers", 0.0, 0.0, 1, "n/a"),
        ("OP", "In-plane orbit control", tang[1], tang[2], 1, "GMAT"),
        ("OP", "Out-of-plane orbit control", oop[1], oop[2], 1, "GMAT"),
        ("OP", "Formation maintenance", form[1], form[2], 1, "GMAT"),
        ("OP", f"Formation re-acquisition ({a.n_reacq:g} x acquisition)", reacq[1], reacq[2], 1, "assumed"),
        ("OP", f"Collision avoidance ({a.n_cam:g} CAM x 4 margin)", cam[1], cam[2], 1,
         "input" if a.n_cam else "not provided"),
        ("OP", "Attitude control by thrusters (+100 % propellant)", a.acbt_dv, a.acbt_dv, 2,
         "input" if a.acbt_dv else "not provided"),
        ("EOL", "Clearance of the operational orbit", clr[1], clr[2], 1, "GMAT" if "dvClr1" in s else "missing"),
        ("EOL", "Disposal (uncontrolled re-entry)", disp[1], disp[2], 1, "GMAT" if "dvDisp1" in s else "missing"),
    ]
    ve = a.eta * a.isp * 9.80665
    totals = {}
    for sat in (1, 2):
        dry = run.d[f"mass{sat}"][0] - a.tank_kg
        mp = a.tank_kg                                   # initial guess: the tank load
        for _ in range(100):
            m = dry + mp
            props = []
            for ph, name, d1, d2, fac, src in rows:
                dv = d1 if sat == 1 else d2
                dm = fac * m * (1 - math.exp(-dv / ve))
                props.append(dm)
                m -= dm
            res, unc = 0.01 * a.tank_kg, 0.02 * a.tank_kg
            new = sum(props) + res + unc
            if abs(new - mp) < 1e-6:
                mp = new
                break
            mp = new
        totals[sat] = dict(props=props, res=res, unc=unc, mp=mp, dry=dry,
                           dv=sum((r[2] if sat == 1 else r[3]) for r in rows))
    notes = []
    for ph, name, d1, d2, fac, src in rows:
        if src == "not provided":
            notes.append(f"{name}: not provided (zero in the totals)")
    return rows, totals, notes


def ad10_text(run):
    a = run.args
    rows, tot, notes = ad10_budget(run)
    w = []
    w.append("## Delta-V and propellant mass budget (EOP-FM/2024-07-177, Table 1)")
    w.append("")
    w.append("Margin policy of AD10 5.6: none, except 4x on the number of collision-"
             "avoidance manoeuvres, +100 % on the propellant of thruster attitude "
             "control, and the residual (1 %) and uncertainty (2 %) of the tank load.")
    w.append("")
    w.append("| phase | contributor | dV Sat1 [m/s] | dV Sat2 [m/s] | m_p Sat1 [g] | m_p Sat2 [g] | source |")
    w.append("|---|---|---:|---:|---:|---:|---|")
    for i, (ph, name, d1, d2, fac, src) in enumerate(rows):
        w.append(f"| {ph} | {name} | {d1:.2f} | {d2:.2f} | "
                 f"{tot[1]['props'][i]*1000:.0f} | {tot[2]['props'][i]*1000:.0f} | {src} |")
    w.append(f"| | Propellant residual (1 % of {a.tank_kg:g} kg) | | | "
             f"{tot[1]['res']*1000:.0f} | {tot[2]['res']*1000:.0f} | AD10 5.8 |")
    w.append(f"| | Propellant uncertainty, gauging (2 % of {a.tank_kg:g} kg) | | | "
             f"{tot[1]['unc']*1000:.0f} | {tot[2]['unc']*1000:.0f} | AD10 5.8 |")
    w.append(f"| | **Total** | **{tot[1]['dv']:.2f}** | **{tot[2]['dv']:.2f}** | "
             f"**{tot[1]['mp']*1000:.0f}** | **{tot[2]['mp']*1000:.0f}** | |")
    w.append("")
    w.append(f"Formation total {tot[1]['dv']+tot[2]['dv']:.2f} m/s and "
             f"{(tot[1]['mp']+tot[2]['mp']):.3f} kg of propellant (Isp {a.isp:g} s, "
             f"efficiency eta {a.eta:g}, dry mass {tot[1]['dry']:.0f} kg, iterated as in "
             f"AD10 6.5).")
    fits = all(t["mp"] <= a.tank_kg for t in tot.values())
    w.append(f"Tank of {a.tank_kg:g} kg: {'PASS' if fits else 'FAIL'} "
             f"(largest need {max(t['mp'] for t in tot.values()):.3f} kg).")
    if a.margin:
        w.append(f"Additional programme margin requested: {a.margin*100:.0f} % on "
                 f"the total delta-V = {(tot[1]['dv']+tot[2]['dv'])*(1+a.margin):.2f} m/s.")
    if notes:
        w.append("")
        for n in notes:
            w.append(f"- INPUT NEEDED: {n}")
    w.append("")
    return w


# ----------------------------------------------------------------------------
# Text summary
# ----------------------------------------------------------------------------
def esa_checks(run):
    """Checks against ESSB-ST-U-007 that can be made from the run files."""
    s = run.s
    out = []
    t = run.t
    if run.i_ops > 0 or True:
        # 5.3.3.2.a / 5.3.3.4: separation of the two satellites after the release
        for day in (1, 3, 7):
            if t[-1] > day:
                out.append((f"Separation {day} d after release", float(np.interp(day, t, run.rng)), "km"))
    if "life1" in s:
        for sat in (1, 2):
            life = s[f"life{sat}"]
            out.append((f"Orbit lifetime after disposal, Sat{sat}",
                        life / 365.25, "years"))
        # the lifetime counts from the end of life: add the thrusting descent
        # (dv / (F/m)), estimated with the mean mass of the satellite
        args = run.args
        m_eol = 399.5
        desc = {k: (s[f"dvClr{k}"] + s[f"dvDisp{k}"]) / (args.thrust / m_eol * 86400)
                if "dvClr1" in s else 0.0 for k in (1, 2)}
        if "dvClr1" in s:
            for sat in (1, 2):
                out.append((f"Thrusting descent, Sat{sat} (estimate)", desc[sat], "days"))
                out.append((f"Total time from the end of life to re-entry, Sat{sat}",
                            (desc[sat] + s[f"life{sat}"]) / 365.25, "years"))
        ok = all(desc[k] + s[f"life{k}"] < 1826 and s.get(f"hEnd{k}", 0) <= 125 for k in (1, 2))
        out.append(("Natural decay < 5 years from the end of life (5.4.2.3.a.1)",
                    "PASS" if ok else "FAIL", ""))
        out.append(("Disposal orbit mean altitude", (s["aDispT"] - 6378.1363), "km"))
    return out


def summary_text(run, comp):
    a = run.args
    lines = []
    w = lines.append
    w(f"# Formation delta-V budget – {run.label}")
    w("")
    if not run.controlled:
        w("**Uncontrolled run (ControlOn = 0): no thrust, no delta-V.**")
        w("")
    w(f"Mission length {run.t[-1]:.0f} days ({run.t[-1]/365.25:.2f} years), "
      f"{len(run.t)-1} control cycles.")
    w("")
    w("## Delta-V (m/s)")
    w("")
    w("| | tangential | normal | total | with %d %% margin | propellant, g |"
      % round(a.margin * 100))
    w("|---|---:|---:|---:|---:|---:|")
    for s in (1, 2):
        t, n = run.dv_final[s]["tang"], run.dv_final[s]["norm"]
        m0 = run.d[f"mass{s}"][0]
        prop = m0 * (1 - math.exp(-(t + n) * (1 + a.margin) / (a.isp * G0))) * 1000
        w(f"| Sat{s} | {t:.2f} | {n:.2f} | {t+n:.2f} | {(t+n)*(1+a.margin):.2f} | {prop:.0f} |")
    tt = sum(run.dv_final[s]["tang"] for s in (1, 2))
    nn = sum(run.dv_final[s]["norm"] for s in (1, 2))
    w(f"| **Formation** | {tt:.2f} | {nn:.2f} | **{tt+nn:.2f}** | "
      f"**{(tt+nn)*(1+a.margin):.2f}** | |")
    w("")
    if run.i_ops > 0:
        w("")
        w(f"| phase | Sat1 | Sat2 | formation | with %d %% margin |" % round(a.margin * 100))
        w("|---|---:|---:|---:|---:|")
        a1 = sum(run.dv_acq[1].values())
        a2 = sum(run.dv_acq[2].values())
        o1 = sum(run.dv_final[1].values()) - a1
        o2 = sum(run.dv_final[2].values()) - a2
        for name, x1, x2 in ((f"acquisition, days 0–{run.t_acq:.0f}", a1, a2),
                             ("operations after it", o1, o2)):
            w(f"| {name} | {x1:.2f} | {x2:.2f} | {x1+x2:.2f} | "
              f"{(x1+x2)*(1+a.margin):.2f} |")
        w("")
        w("Both satellites start at the same position; the acquisition builds the "
          "along-track and normal separation and is part of the totals above.")
    w("")
    w("")
    for line in ad10_text(run):
        w(line)
    w("## ESSB-ST-U-007 checks that can be made from this run")
    w("")
    for name, val, unit in esa_checks(run):
        w(f"- {name}: {val:.1f} {unit}" if isinstance(val, float) else f"- {name}: {val}")
    w("")
    w("Not covered by the simulation (see the compliance note next to the script): "
      "probability of unintentional contact (5.3.3.4), probability of successful "
      "disposal and passivation (5.4.1.1, 5.3.2.2), collision probability until "
      "re-entry (5.4.2.3.a.2), re-entry casualty risk (5.5).")
    w("")
    w("Per mission year (formation, m/s): " + ", ".join(
        f"Y{i+1} {r['tang']+r['norm']:.1f}" for i, r in enumerate(run.per_year_dv())))
    w("")
    w("## Compliance")
    w("")
    if run.i_ops > 0:
        w(f"(Compliance is judged from day {run.t_acq:.0f}, after the acquisition.)" if run.acq_done
          else "(The acquisition was still running at the end of this run; compliance is judged over the whole run.)")
        w("")
    w(f"- Along-track {comp['T_min']:.0f}–{comp['T_max']:.0f} km "
      f"(limits {a.t_min:.0f}–{a.t_max:.0f}): "
      f"{'PASS' if comp['T_ok'] else 'FAIL'}, margin {comp['T_margin']:.0f} km, "
      f"{comp['T_in_frac']*100:.1f} % of cycles inside")
    w(f"- Normal amplitude {comp['AN_min']:.1f}–{comp['AN_max']:.1f} km "
      f"(requirement {a.n_amp:.0f} ±{a.n_tol*100:.0f} %): "
      f"{'PASS' if comp['AN_ok'] else 'FAIL'}, {comp['AN_in_frac']*100:.1f} % inside")
    w(f"- Closest approach {comp['rng_min']:.0f} km (minimum allowed "
      f"{a.min_range:.0f} km): {'PASS' if comp['rng_ok'] else 'FAIL'}")
    w(f"- Orbit plane: inclination {comp['inc_change']:+.3f} deg, LTAN of the pair centre up to "
      f"{comp['ltan_max']:.1f} min from the start, altitude {comp['alt_change']:+.2f} km")
    w("")
    w("## Checks on the run itself")
    w("")
    s = run.s
    problems = []
    if s:
        for k, sat in (("dvRkt1", 1), ("dvRkt2", 2)):
            if k in s:
                mine = sum(run.dv_final[sat].values())
                if mine > 0 and abs(s[k] - mine) > 0.01 * mine + 1e-3:
                    problems.append(f"Sat{sat}: rocket-equation delta-V {s[k]:.2f} "
                                    f"differs from integrated thrust {mine:.2f} m/s")
        if "epMax" in s and s["epMax"] > 1e-3:
            problems.append(f"satellites were up to {s['epMax']:.3g} s apart in epoch")
        if "nOutT" in s and s["nOutT"] > 0 and comp["T_ok"]:
            problems.append("summary reports along-track violations; the limits in "
                            "the script differ from --t-min/--t-max")
    if problems:
        for p in problems:
            w(f"- WARNING: {p}")
    else:
        w("- delta-V from the propellant (rocket equation) matches the integrated "
          "thrust" if s else "- no summary file found, delta-V taken from the daily log")
    w("")
    w("Daily samples are taken once per 15-orbit cycle at a fixed orbit phase, so "
      "short-period swings of a few km inside a cycle are not visible.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
def plt():
    import matplotlib.pyplot as p
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="GMAT output folder(s) or Daily file(s)")
    ap.add_argument("--labels", nargs="+", help="names for the runs, in order")
    ap.add_argument("--outdir", default="formation_plots")
    ap.add_argument("--start-date", default="2027-01-01",
                    help="mission start (YYYY-MM-DD), used for the time axis and "
                         "LTAN; must match the Epoch in the GMAT script")
    ap.add_argument("--t-min", type=float, default=50, help="along-track minimum, km")
    ap.add_argument("--t-max", type=float, default=1000, help="along-track maximum, km")
    ap.add_argument("--n-amp", type=float, default=200, help="normal amplitude, km")
    ap.add_argument("--n-tol", type=float, default=0.10,
                    help="allowed fractional error of the normal amplitude")
    ap.add_argument("--min-range", type=float, default=10,
                    help="smallest allowed 3-D separation, km")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="additional programme margin on the delta-V; AD10 5.6 "
                         "prescribes none (default 0)")
    ap.add_argument("--lie-ip", type=float, default=0.0,
                    help="launcher injection errors, in-plane delta-V per satellite, m/s (AD10 6.1.1)")
    ap.add_argument("--lie-oop", type=float, default=0.0,
                    help="launcher injection errors, out-of-plane delta-V per satellite, m/s")
    ap.add_argument("--n-reacq", type=float, default=1.0,
                    help="formation re-acquisitions after a contingency, each costing "
                         "as much as the acquisition (AD10 6.2.5)")
    ap.add_argument("--n-cam", type=float, default=0.0,
                    help="collision-avoidance manoeuvres per satellite from a DRAMA "
                         "run (AD10 6.2.4); 4x is applied, 0.109 m/s each at 560 km")
    ap.add_argument("--acbt-dv", type=float, default=0.0,
                    help="delta-V of thruster attitude control per satellite, m/s (100 %% propellant margin)")
    ap.add_argument("--tank-kg", type=float, default=5.0,
                    help="maximum propellant load of each tank, kg (AD10 5.8)")
    ap.add_argument("--eta", type=float, default=1.0, help="thruster efficiency (AD10 5.11)")
    ap.add_argument("--isp", type=float, default=2800, help="thruster Isp, s")
    ap.add_argument("--thrust", type=float, default=0.001,
                    help="thrust per thruster, N (for the firing-time estimate)")
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    ap.add_argument("--show", action="store_true", help="open the figures")
    args = ap.parse_args(argv)

    global C
    C = THEMES[args.theme]
    if not args.show:
        matplotlib.use("Agg")
    style()

    runs = []
    for i, r in enumerate(args.runs):
        daily, summ, label = find_files(r)
        if not os.path.exists(daily):
            sys.exit(f"cannot find {daily}")
        if args.labels and i < len(args.labels):
            label = args.labels[i]
        runs.append(Run(daily, summ, label, args))

    os.makedirs(args.outdir, exist_ok=True)
    for run in runs:
        out = args.outdir if len(runs) == 1 else os.path.join(args.outdir, run.label)
        os.makedirs(out, exist_ok=True)
        comp = compliance(run)
        fig_compliance(run, comp, out, args.show)
        fig_orbit(run, out, args.show)
        if run.controlled:
            fig_dv(run, out, args.show)
            fig_propulsion(run, out, args.show)
            fig_controller(run, out, args.show)
        else:
            print(f"{run.label}: uncontrolled run (no thrust), so the delta-V, "
                  "thruster and controller figures are skipped")
        text = summary_text(run, comp)
        with open(os.path.join(out, "budget_summary.md"), "w") as f:
            f.write(text + "\n")
        print(text)
        print(f"\nfigures written to {out}\n")
    if len(runs) > 1:
        fig_compare(runs, args.outdir, args.show)
        print(f"comparison written to {args.outdir}")
    if args.show:
        plt().show()


if __name__ == "__main__":
    main()
