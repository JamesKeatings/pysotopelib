#!/usr/bin/env python3
"""
check_calibration.py

Read a two-column ASCII spectrum (E in keV, counts), fit each peak using the
accepted gamma-ray energies from source_data.py (IAEA-TECDOC-619) as the
starting point, and report how well the energy calibration agrees with the
literature.

Steps
  1. np.loadtxt -> two NumPy arrays (energy, counts)
  2. take the accepted energies of the chosen nuclides; lines closer together
     than 4 x FWHM (e.g. 152Eu 1085.8/1089.8 keV) are fitted together as a
     doublet / multiplet with one shared width, everything else singly
  3. seed each fit at the accepted energies, shifted rigidly by up to
     +/- --tol keV to best match the data, then fit a window around them with
     bin-integrated Gaussian(s) (correct for ~1 keV channels) + quadratic
     background, Poisson weights. The FWHM used for the 4 x FWHM rule starts
     at --fwhm and is refined from the fitted peaks automatically
  4. compare fitted centroids with the accepted energies, then fit
     E_ref = c0 + c1*E_fit (and a quadratic) to see the calibration error

Usage:
    python3 check_calibration.py simulated_spectrum.txt
    python3 check_calibration.py spectrum.txt --nuclides 152Eu 60Co --plot cal.png
    python3 check_calibration.py simulated_spectrum.txt          (opens matplotlib window)
    python3 check_calibration.py simulated_spectrum.txt --no-show --plot cal.png
    python3 check_calibration.py simulated_spectrum.txt --no-plot
    python3 check_calibration.py simulated_spectrum.txt --order 2 --residuals
    python3 check_calibration.py spectrum.txt --report report.html --operator "J. Smith" \\
            --detector "GEM-30" --notes "Eu-152 + Ba-133, 25 cm" --tolerance 0.3
    python3 check_calibration.py spectrum.txt --report report.pdf     (LaTeX -> PDF)
    python3 check_calibration.py spectrum.txt --report report.tex     (LaTeX source only)
    python3 check_calibration.py simulated_spectrum.txt --skip-low   (ignore lines < 50 keV)
    python3 check_calibration.py simulated_spectrum.txt --emin 100   (any other cut)
"""

import argparse
import base64
import datetime
import hashlib
import html
import io
import platform
import re
import string
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

from source_data import GAMMA_DATA, combine_val_err

DEFAULT_NUCLIDES = ["241Am", "109Cd", "57Co", "139Ce", "51Cr", "113Sn",
                    "85Sr", "137Cs", "54Mn", "88Y", "65Zn", "60Co"]
SQRT2 = np.sqrt(2.0)
FWHM_FACTOR = 2.3548
LOW_ENERGY_CUT = 50.0   # keV, threshold used by --skip-low
MULTIPLET_FWHM = 4.0   # lines closer than this many FWHM are fitted together
WINDOW_FWHM = 3.0      # fit half-window beyond the outermost peak, in FWHM
MU_SLACK = 3.0      # keV the centroid may move from its seed during the fit


# ---------------------------------------------------------------- input
def read_spectrum(path):
    """Load a two-column ASCII spectrum -> (energy, counts) float arrays.
    Lines starting with '#' are ignored."""
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 2:
        sys.exit(f"{path}: expected two columns (E, counts)")
    return data[:, 0].astype(float), data[:, 1].astype(float)


def reference_lines(nuclides):
    """Library lines as a list of dicts: nuclide, E, dE, P (keV, keV, abs.)."""
    lines = []
    for nuc in nuclides:
        if nuc not in GAMMA_DATA:
            sys.exit(f"Nuclide {nuc} not in source_data.GAMMA_DATA")
        for e_s, e_err_s, p_s, p_err_s, ref, daughter in GAMMA_DATA[nuc]:
            e, de = combine_val_err(e_s, e_err_s)
            p, _ = combine_val_err(p_s, p_err_s)
            lines.append({"nuclide": nuc, "E": e, "dE": de or 0.0, "P": p})
    lines.sort(key=lambda l: l["E"])
    return lines


def group_lines(lines, gap):
    """Group energy-sorted lines so neighbours closer than `gap` keV are
    fitted together."""
    groups = []
    for line in lines:
        if groups and line["E"] - groups[-1][-1]["E"] <= gap:
            groups[-1].append(line)
        else:
            groups.append([line])
    return groups


# -------------------------------------------------------------- fitting
def multi_model(x, params, npk, xc):
    """Sum of bin-integrated Gaussians (shared sigma) + quadratic background.
    params = [area_1..n, mu_1..n, sigma, b0, b1, b2]."""
    areas = params[:npk]
    mus = params[npk:2 * npk]
    sigma = params[2 * npk]
    b0, b1, b2 = params[2 * npk + 1:2 * npk + 4]
    y = b0 + b1 * (x - xc) + b2 * (x - xc) ** 2
    for a, mu in zip(areas, mus):
        y = y + a * 0.5 * (erf((x + 0.5 - mu) / (SQRT2 * sigma))
                           - erf((x - 0.5 - mu) / (SQRT2 * sigma)))
    return y


def seed_shift(energy, smooth, e_ref, tol):
    """Rigid shift (keV, within +/- tol) of the accepted energies of one fit
    group that best matches the data. Using one shift for the whole group
    stops close neighbours (e.g. 152Eu 1085.8 / 1089.8 keV) from both
    seeding on the same maximum."""
    deltas = np.arange(-tol, tol + 1e-9, 0.1)
    score = [sum(np.interp(e + d, energy, smooth) for e in e_ref) for d in deltas]
    return deltas[int(np.argmax(score))]


def multiplet_name(n):
    return {1: "single", 2: "doublet", 3: "triplet"}.get(n, f"{n}-multiplet")


def fit_group(energy, counts, group, tol, half_window, fwhm_guess):
    """Fit all lines in `group` simultaneously (shared width). Returns a list
    of result dicts (one per line) or None if the fit fails."""
    npk = len(group)
    e_ref = [l["E"] for l in group]
    smooth = np.convolve(counts, np.ones(3) / 3.0, mode="same")
    shift = seed_shift(energy, smooth, e_ref, tol)
    mu0 = [e + shift for e in e_ref]
    # centroids may move by at most 3 keV, or 45% of the distance to the
    # nearest neighbour in the group so two lines can never swap places
    nearest = [min([abs(e - e2) for j, e2 in enumerate(e_ref) if j != k],
                   default=np.inf) for k, e in enumerate(e_ref)]
    slack_k = [min(MU_SLACK, 0.45 * d) for d in nearest]

    lo = np.searchsorted(energy, min(mu0) - half_window)
    hi = np.searchsorted(energy, max(mu0) + half_window, side="right")
    x, y = energy[lo:hi], counts[lo:hi]
    if len(x) < 2 * npk + 8:
        return None
    err = np.sqrt(np.maximum(y, 1.0))
    xc = x.mean()

    edge = max(3, len(x) // 8)
    bg = 0.5 * (y[:edge].mean() + y[-edge:].mean())
    sigma0 = fwhm_guess / FWHM_FACTOR
    area0 = []
    for m in mu0:
        height = np.interp(m, energy, smooth) - bg
        area0.append(max(height, 1.0) * sigma0 * np.sqrt(2 * np.pi))

    p0 = area0 + mu0 + [sigma0, bg, 0.0, 0.0]
    lower = [0.0] * npk + [m - w for m, w in zip(mu0, slack_k)] + [0.3, -np.inf, -np.inf, -np.inf]
    upper = [np.inf] * npk + [m + w for m, w in zip(mu0, slack_k)] + [max(4.0, 3 * sigma0), np.inf, np.inf, np.inf]
    try:
        popt, pcov = curve_fit(lambda xx, *p: multi_model(xx, p, npk, xc),
                               x, y, p0=p0, sigma=err, absolute_sigma=True,
                               bounds=(lower, upper), maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    perr = np.sqrt(np.diag(pcov))
    if not np.all(np.isfinite(perr)):
        return None

    chi2 = np.sum(((y - multi_model(x, popt, npk, xc)) / err) ** 2)
    chi2ndf = chi2 / max(len(x) - len(popt), 1)
    results = []
    for k, line in enumerate(group):
        results.append({**line,
                        "area": popt[k], "darea": perr[k],
                        "mu": popt[npk + k], "dmu": perr[npk + k],
                        "fwhm": FWHM_FACTOR * popt[2 * npk],
                        "chi2ndf": chi2ndf, "group_size": npk,
                        "dfwhm": FWHM_FACTOR * perr[2 * npk],
                        "group_key": (group[0]["nuclide"], group[0]["E"])})
    return results


def fit_calibration(matched, order):
    """Weighted polynomial fit E_ref = sum c_k * E_fit^k. Returns coeffs,
    errors, chi2/ndf and the per-peak residuals."""
    x = np.array([m["mu"] for m in matched])
    y = np.array([m["E"] for m in matched])
    s = np.array([np.hypot(m["dmu"], m["dE"]) for m in matched])
    s = np.maximum(s, 1e-6)
    ndf = len(x) - (order + 1)
    if ndf < 1:
        return None

    def poly(xx, *c):
        return sum(ck * xx ** k for k, ck in enumerate(c))

    popt, pcov = curve_fit(poly, x, y, p0=[0.0, 1.0] + [0.0] * (order - 1),
                           sigma=s, absolute_sigma=True)
    resid = y - poly(x, *popt)
    chi2 = np.sum((resid / s) ** 2)
    return {"coef": popt, "err": np.sqrt(np.diag(pcov)), "cov": pcov,
            "order": order, "chi2ndf": chi2 / ndf,
            "resid": resid, "sigma": s, "ndf": ndf}


def calibration_curve(cal, e):
    """Calibrated energy f(E_axis) and its 1-sigma uncertainty at the axis
    energies `e`, propagated from the fit covariance:

        sigma_f^2 = J C J^T,   J = (1, E, E^2, ...)

    If chi2/ndf > 1 the scatter of the points exceeds their quoted errors,
    so the band is scaled up by sqrt(chi2/ndf) (Birge ratio)."""
    e = np.atleast_1d(np.asarray(e, dtype=float))
    jac = np.vstack([e ** k for k in range(cal["order"] + 1)]).T
    f = jac @ cal["coef"]
    var = np.einsum("ij,jk,ik->i", jac, cal["cov"], jac)
    scale = np.sqrt(max(1.0, cal["chi2ndf"]))
    return f, scale * np.sqrt(np.maximum(var, 0.0))


def run_fits(energy, counts, library, args, fwhm):
    """Group the library lines (neighbours closer than MULTIPLET_FWHM x fwhm
    are fitted together), fit every group and sort the results."""
    gap = MULTIPLET_FWHM * fwhm
    half_window = args.window if args.window else WINDOW_FWHM * fwhm
    groups = group_lines(library, gap)
    matched, rejected = [], []
    for group in groups:
        res = fit_group(energy, counts, group, args.tol, half_window, fwhm)
        if res is None:
            rejected += [(l, "fit failed") for l in group]
            continue
        for r in res:
            if r["area"] / max(r["darea"], 1e-9) < args.min_sig:
                rejected.append((r, "not detected"))
            elif r["dmu"] > args.max_dmu:
                rejected.append((r, f"centroid error {r['dmu']:.2f} keV"))
            else:
                matched.append(r)
    return groups, matched, rejected


# ---------------------------------------------------------------- plots
def make_plot(path, energy, counts, matched, cal, show=True, relative=False,
              return_figs=False):
    """Top: spectrum (log y) with the fitted peaks marked and labelled.
    Bottom: E_fit - E_ref vs energy (points, with centroid + reference
    uncertainties) and the 1-sigma uncertainty of the fitted calibration as
    red upper/lower lines with a shaded band. relative=True instead plots
    the points after calibration (E_cal - E_ref) with the band centred on 0.
    Opens an interactive matplotlib window if show=True (zoom/pan with the
    toolbar) and also saves a PNG if `path` is given. With no display
    available (e.g. over ssh) it falls back to saving a PNG."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")      # no window, file output only
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    # Times New Roman-style fonts: first one installed on the machine wins,
    # STIX (bundled with matplotlib) is the maths/fallback Times-like face.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman",
                       "Liberation Serif", "TeX Gyre Termes", "STIXGeneral",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,          # embed real fonts in PDF output
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.semilogy(energy, np.maximum(counts, 0.5), lw=0.6, color="k")
    for m in matched:
        ax1.axvline(m["mu"], color="r", alpha=0.3, lw=0.8)
        y_top = counts[np.searchsorted(energy, m["mu"])]
        ax1.annotate(f"{m['nuclide']}\n{m['E']:.1f}", (m["mu"], max(y_top, 1)),
                     xytext=(0, 6), textcoords="offset points", ha="center",
                     fontsize=6, rotation=90, va="bottom")
    ax1.set_ylim(top=counts.max() * 12)
    ax1.set_ylabel("Counts / keV")
    ax1.set_title("Spectrum with fitted peaks")

    x = np.array([m["mu"] for m in matched])
    e_ref = np.array([m["E"] for m in matched])
    ddiff = np.array([np.hypot(m["dmu"], m["dE"]) for m in matched])

    # fitted calibration and its 1-sigma uncertainty over the whole axis
    e_grid = np.linspace(energy[0], energy[-1], 600)
    f_grid, sig_grid = calibration_curve(cal, e_grid)
    if relative:
        # points after calibration; the calibration itself is the zero line
        diff = calibration_curve(cal, x)[0] - e_ref
        centre = np.zeros_like(e_grid)
        ylabel = r"$E_{\mathrm{cal}} - E_{\mathrm{ref}}$ (keV)"
        pt_label = r"$E_{\mathrm{cal}} - E_{\mathrm{ref}}$"
    else:
        # raw axis energies; the fitted calibration is the curve E - f(E)
        diff = x - e_ref
        centre = e_grid - f_grid
        ylabel = r"$E_{\mathrm{fit}} - E_{\mathrm{ref}}$ (keV)"
        pt_label = r"$E_{\mathrm{fit}} - E_{\mathrm{ref}}$"

    ax2.fill_between(e_grid, centre - sig_grid, centre + sig_grid,
                     color="red", alpha=0.18, lw=0, zorder=1,
                     label=r"Calibration $\pm 1\sigma$")
    ax2.plot(e_grid, centre + sig_grid, color="red", lw=1.0, zorder=2)
    ax2.plot(e_grid, centre - sig_grid, color="red", lw=1.0, zorder=2)
    ax2.errorbar(x, diff, ddiff, fmt="o", ms=4, capsize=2, zorder=3,
                 label=pt_label)
    ax2.axhline(0, color="gray", lw=0.8, zorder=0)
    ax2.set_xlabel("Energy (keV)")
    ax2.set_ylabel(ylabel)
    ax2.legend(loc="best")

    # x axis (shared): major ticks every 250 keV, minor every 50 keV
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(MultipleLocator(250))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.tick_params(which="major", length=6)
        ax.tick_params(which="minor", length=3)
    ax2.set_xlim(left=energy[0] - 25, right=energy[-1] + 25)
    fig.tight_layout()
    figs = None
    if return_figs:                     # keep copies for the report
        fig.set_size_inches(8.0, 6.4)   # sized for the page width, text stays legible
        fig.tight_layout()
        figs = {}
        for fmt in ("png", "pdf"):
            buf = io.BytesIO()
            fig.savefig(buf, format=fmt, dpi=150)
            figs[fmt] = buf.getvalue()
        fig.set_size_inches(11, 7)      # back to the on-screen size
        fig.tight_layout()
    headless = matplotlib.get_backend().lower() == "agg"
    if show and headless and not path:
        path = "calibration_check.png"
        print("No display available for an interactive window; saving instead.")
    if path:
        fig.savefig(path, dpi=130)
        print(f"Plot written to {path}")
    if show and not headless:
        plt.show()
    plt.close(fig)
    return figs


# --------------------------------------------------------------- report
def fit_resolution(matched):
    """Weighted fit FWHM^2 = A + B*E, one point per fit group (lines in a
    multiplet share one width, so they must not be counted twice)."""
    seen, pts = set(), []
    for m in matched:
        if m["group_key"] in seen:
            continue
        seen.add(m["group_key"])
        pts.append((m["mu"], m["fwhm"], max(m["dfwhm"], 1e-3)))
    if len(pts) < 4:
        return None
    e, f, df = (np.array(v) for v in zip(*pts))
    try:
        popt, pcov = curve_fit(lambda x, a, b: a + b * x, e, f ** 2,
                               sigma=2 * f * df, absolute_sigma=False)
    except (RuntimeError, ValueError):
        return None
    return {"coef": popt, "cov": pcov, "n": len(pts), "emin": e.min(), "emax": e.max()}


def resolution_at(res, e):
    """FWHM (keV) and its 1-sigma uncertainty from the resolution fit."""
    y = res["coef"][0] + res["coef"][1] * e
    fwhm = np.sqrt(max(y, 0.0))
    jac = np.array([1.0, e])
    sy = np.sqrt(max(jac @ res["cov"] @ jac, 0.0))
    return fwhm, (sy / (2 * fwhm) if fwhm > 0 else float("nan"))


def html_table(header, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def kv_table(pairs):
    body = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in pairs)
    return f'<table class="kv"><tbody>{body}</tbody></table>'


def collect_flags(ctx):
    """Automatic warnings as plain-text strings (shared by all report formats)."""
    matched, rejected, cal, energy = ctx["matched"], ctx["rejected"], ctx["cal"], ctx["energy"]
    flags = []
    if rejected:
        flags.append(f"{len(rejected)} library line(s) could not be used (see sources section).")
    if cal["chi2ndf"] > 2:
        flags.append(f"Calibration chi2/ndf = {cal['chi2ndf']:.2f}: scatter exceeds the quoted errors.")
    for m, d, dd in zip(matched, ctx["diffs"], ctx["dds"]):
        if abs(d / dd) > 3:
            flags.append(f"{m['nuclide']} {m['E']:.2f} keV deviates by {d / dd:+.1f} sigma.")
        if m["chi2ndf"] > 3:
            flags.append(f"{m['nuclide']} {m['E']:.2f} keV: poor peak fit (chi2/ndf = {m['chi2ndf']:.1f}).")
    if len(matched) < 5:
        flags.append("Fewer than 5 peaks fitted: calibration is weakly constrained.")
    span = np.array([m["E"] for m in matched])
    if span.max() - span.min() < 0.5 * (energy[-1] - energy[0]):
        flags.append("Fitted peaks cover less than half of the energy range: "
                     "calibration is extrapolated at one end.")
    return flags


def write_report_html(args, ctx, figs):
    """Fill report_template.html (or --template) with the results and write
    a self-contained HTML file (figure embedded) to args.report."""
    template_path = Path(args.template) if args.template else \
        Path(__file__).with_name("report_template.html")
    if not template_path.exists():
        sys.exit(f"Report template not found: {template_path}")
    esc = html.escape
    pm = "&plusmn;"

    matched, rejected, library = ctx["matched"], ctx["rejected"], ctx["library"]
    cal, energy, counts = ctx["cal"], ctx["energy"], ctx["counts"]
    diffs, dds, fwhms = ctx["diffs"], ctx["dds"], ctx["fwhms"]
    e_cal_diff = -cal["resid"]                      # E_cal - E_ref
    now = datetime.datetime.now().astimezone()
    spec_path = Path(args.spectrum)
    sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()[:16]

    # --- verdict -----------------------------------------------------
    worst = ctx["worst"]
    max_diff = float(np.abs(diffs).max())
    ok = max_diff <= args.tolerance
    verdict_html = (
        f'<p class="verdict">Result: <span class="{"pass" if ok else "fail"}">'
        f'{"PASS" if ok else "FAIL"}</span> &ndash; largest |E<sub>fit</sub> &minus; '
        f'E<sub>ref</sub>| is {max_diff:.3f} keV ({esc(worst["nuclide"])} '
        f'{worst["E"]:.2f} keV) against a tolerance of {args.tolerance:g} keV.</p>')

    meta_table = kv_table([
        ("Date", now.strftime("%d %B %Y, %H:%M %Z")),
        ("Operator", esc(args.operator or "&mdash;")),
        ("Detector", esc(args.detector or "&mdash;")),
        ("Spectrum file", esc(spec_path.name)),
        ("File checksum (SHA-256, first 16)", sha),
        ("Notes", esc(args.notes or "&mdash;")),
    ])
    spectrum_table = kv_table([
        ("Channels", f"{len(energy)} ({energy[0]:g} &ndash; {energy[-1]:g} keV, "
                     f"{energy[1] - energy[0]:g} keV per channel)"),
        ("Total counts", f"{counts.sum():,.0f}"),
        ("Highest channel", f"{counts.max():,.0f} counts at {energy[int(np.argmax(counts))]:g} keV"),
    ])

    # --- sources -----------------------------------------------------
    fitted_by_nuc = {}
    for m in matched:
        fitted_by_nuc.setdefault(m["nuclide"], []).append(m)
    unused_by_nuc = {}
    for line, why in rejected:
        unused_by_nuc.setdefault(line["nuclide"], []).append((line, why))
    src_rows = []
    for nuc in args.nuclides:
        n_lib = sum(1 for l in library if l["nuclide"] == nuc)
        fl = fitted_by_nuc.get(nuc, [])
        src_rows.append([esc(nuc), n_lib, len(fl),
                         ", ".join(f"{m['E']:.1f}" for m in fl) or "&mdash;"])
    sources_table = html_table(["Nuclide", "Library lines in range", "Lines fitted",
                                "Energies fitted (keV)"], src_rows)
    if rejected:
        items = "".join(f"<li>{esc(l['nuclide'])} {l['E']:.2f} keV &ndash; {esc(why)}</li>"
                        for l, why in rejected)
        not_used = f"<p>Library lines not used in the calibration:</p><ul>{items}</ul>"
    else:
        not_used = "<p class=\"small\">All library lines in range were fitted.</p>"

    # --- settings ----------------------------------------------------
    settings_table = kv_table([
        ("Peak model", "bin-integrated Gaussian(s) + quadratic background"),
        ("Peak-fit weights", "Poisson, 1/&radic;N per channel (N &ge; 1)"),
        ("Seed search", f"accepted energies shifted by up to {pm}{args.tol:g} keV"),
        ("Multiplet rule", f"lines closer than {MULTIPLET_FWHM:g} &times; FWHM = "
                           f"{MULTIPLET_FWHM * ctx['fwhm_used']:.1f} keV fitted together "
                           f"(shared width)"),
        ("FWHM: start / used", f"{args.fwhm:g} keV / {ctx['fwhm_used']:.2f} keV"),
        ("Fit half-window", f"{ctx['half_window']:.1f} keV beyond the outermost peak"),
        ("Lowest line fitted", f"{args.emin:g} keV"),
        ("Minimum significance (area/error)", f"{args.min_sig:g}"),
        ("Maximum centroid uncertainty", f"{args.max_dmu:g} keV"),
        ("Calibration polynomial", f"order {cal['order']} (E<sub>true</sub> = f(E<sub>axis</sub>))"),
        ("Calibration weights", "1/(&sigma;<sub>centroid</sub>&sup2; + &sigma;<sub>ref</sub>&sup2;)"),
        ("Pass/fail tolerance", f"{args.tolerance:g} keV on |E<sub>fit</sub> &minus; E<sub>ref</sub>|"),
    ])

    # --- figure ------------------------------------------------------
    b64 = base64.b64encode(figs["png"]).decode("ascii")
    figure = f'<img alt="Spectrum and calibration residuals" src="data:image/png;base64,{b64}">'

    # --- peak table --------------------------------------------------
    peak_rows = []
    for m, d, dd, ec in zip(matched, diffs, dds, e_cal_diff):
        ref = f"{m['E']:.3f}" + (f" {pm} {m['dE']:.4f}" if m["dE"] > 0 else "")
        peak_rows.append([esc(m["nuclide"]), ref, f"{m['mu']:.3f} {pm} {m['dmu']:.3f}",
                          f"{d:+.3f}", f"{d / dd:+.1f}", f"{m['fwhm']:.2f}",
                          f"{m['area']:,.0f}", f"{m['chi2ndf']:.2f}",
                          multiplet_name(m["group_size"]), f"{ec:+.3f}"])
    peak_table = html_table(
        ["Nuclide", "E<sub>ref</sub> (keV)", "E<sub>fit</sub> (keV)",
         "E<sub>fit</sub> &minus; E<sub>ref</sub> (keV)", "Pull", "FWHM (keV)", "Area",
         "&chi;&sup2;/ndf", "Fit", "E<sub>cal</sub> &minus; E<sub>ref</sub> (keV)"], peak_rows)

    agreement_table = kv_table([
        ("Peaks fitted", f"{len(matched)} of {len(library)} library lines"),
        ("Mean difference", f"{diffs.mean():+.4f} keV"),
        ("RMS difference", f"{np.sqrt(np.mean(diffs ** 2)):.4f} keV"),
        ("Largest |difference|", f"{max_diff:.4f} keV ({esc(worst['nuclide'])} {worst['E']:.2f} keV)"),
        ("&chi;&sup2;/ndf of differences about zero", f"{np.sum((diffs / dds) ** 2) / len(diffs):.2f}"),
    ])

    # --- resolution --------------------------------------------------
    res = fit_resolution(matched)
    res_html = kv_table([
        ("Mean FWHM", f"{fwhms.mean():.2f} keV"),
        ("Median FWHM", f"{np.median(fwhms):.2f} keV"),
        ("Range", f"{fwhms.min():.2f} &ndash; {fwhms.max():.2f} keV"),
    ])
    if res:
        a, b = res["coef"]
        rows = []
        for e in (122, 662, 1332):
            if res["emin"] <= e <= res["emax"]:
                f, sf = resolution_at(res, e)
                rows.append([f"{e}", f"{f:.2f} {pm} {sf:.2f}"])
        res_html += (f"<p>Resolution curve fitted to {res['n']} independent widths: "
                     f"FWHM&sup2; = A + B&middot;E with A = {a:.3f} keV&sup2;, "
                     f"B = {b:.2e} keV.</p>")
        if rows:
            res_html += html_table(["E (keV)", "FWHM (keV)"], rows)
    else:
        res_html += '<p class="small">Too few independent peaks for a resolution-curve fit.</p>'

    # --- calibration -------------------------------------------------
    sd = np.sqrt(np.diag(cal["cov"]))
    p = cal["order"] + 1
    coef_rows = [[f"c<sub>{k}</sub>", f"{cal['coef'][k]:+.6g}", f"{sd[k]:.2g}"] for k in range(p)]
    cal_html = html_table(["Coefficient", "Value", "Uncertainty"], coef_rows)
    extra = [("Calibration form", "E<sub>true</sub> = " + " + ".join(
                 ["c<sub>0</sub>"] + [f"c<sub>{k}</sub>E<sup>{k}</sup>" if k > 1 else "c<sub>1</sub>E"
                                      for k in range(1, p)]) + " (E = axis energy)"),
             ("&chi;&sup2;/ndf", f"{cal['chi2ndf']:.2f} (ndf = {cal['ndf']})"),
             ("Error scaling", f"&times; {ctx['scale']:.2f}" + (" (&chi;&sup2;/ndf > 1)" if ctx["scale"] > 1 else " (none needed)")),
             ("RMS scatter about the fit", f"{np.sqrt(np.mean(cal['resid'] ** 2)):.4f} keV")]
    if p >= 2:
        extra.insert(1, ("Correlation c<sub>0</sub>&ndash;c<sub>1</sub>",
                         f"{cal['cov'][0, 1] / (sd[0] * sd[1]):+.3f}"))
    cal_html += kv_table(extra)
    cal_error_table = ("<p>1&sigma; uncertainty of the calibrated energy:</p>" + html_table(
        ["E<sub>axis</sub> (keV)", "f(E) &minus; E (keV)", "&sigma; (keV)"],
        [[f"{e:.0f}", f"{f - e:+.4f}", f"{sg:.4f}"] for e, f, sg in zip(*ctx["probe"])]))

    # --- flags -------------------------------------------------------
    flags = collect_flags(ctx)
    flags_html = ("<ul class=\"warn\">" + "".join(f"<li>{esc(f)}</li>" for f in flags) + "</ul>"
                  if flags else '<p>None.</p>')

    import matplotlib
    import scipy
    software = (f"Python {platform.python_version()}, NumPy {np.__version__}, "
                f"SciPy {scipy.__version__}, Matplotlib {matplotlib.__version__}")

    values = dict(
        title=esc(args.title), date=now.strftime("%d %B %Y, %H:%M %Z"),
        operator=esc(args.operator or "—"), detector=esc(args.detector or "—"),
        notes=esc(args.notes or "—"), spectrum_file=esc(spec_path.name),
        verdict_html=verdict_html, meta_table=meta_table,
        spectrum_table=spectrum_table, sources_table=sources_table,
        not_used=not_used, settings_table=settings_table, figure=figure,
        peak_table=peak_table, agreement_table=agreement_table,
        resolution_html=res_html, cal_html=cal_html,
        cal_error_table=cal_error_table, flags_html=flags_html, software=software)
    text = string.Template(template_path.read_text(encoding="utf-8")).safe_substitute(values)
    Path(args.report).write_text(text, encoding="utf-8")
    print(f"Report written to {args.report} (template: {template_path.name})")


# ------------------------------------------------------- LaTeX / PDF report
_TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex_escape(text):
    return "".join(_TEX_SPECIAL.get(ch, ch) for ch in str(text))


def tn(text):
    """Typeset the minus signs in an already-formatted number."""
    return str(text).replace("-", "$-$")


def tex_sci(x, nd=2):
    mant, exp = f"{x:.{nd}e}".split("e")
    return rf"${mant}\times10^{{{int(exp)}}}$"


def tex_num(text):
    """Formatted number (possibly in exponent form) -> LaTeX."""
    text = str(text)
    if "e" in text:
        mant, exp = text.split("e")
        return rf"${mant}\times10^{{{int(exp)}}}$"
    return tn(text)


def tex_kv(pairs):
    rows = " \\\\\n".join(f"{k} & {v}" for k, v in pairs)
    col = r">{\raggedright\arraybackslash}p{%s\textwidth}"
    return ("\\begin{tabular}{@{}" + col % "0.38" + col % "0.56" + "@{}}\n\\toprule\n"
            + rows + " \\\\\n\\bottomrule\n\\end{tabular}")


def tex_table(header, rows, colspec, long=False):
    head = " & ".join(header) + " \\\\"
    body = " \\\\\n".join(" & ".join(str(c) for c in r) for r in rows) + " \\\\"
    if long:
        return ("{\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\n"
                f"\\begin{{longtable}}{{{colspec}}}\n\\toprule\n{head}\n\\midrule\n"
                f"\\endfirsthead\n\\toprule\n{head}\n\\midrule\n\\endhead\n"
                f"\\bottomrule\n\\endlastfoot\n{body}\n\\end{{longtable}}}}")
    return (f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{head}\n\\midrule\n{body}\n"
            "\\bottomrule\n\\end{tabular}")


def build_tex(args, ctx, figure_name):
    """Fill report_template.tex (or --template) and return the LaTeX source."""
    template_path = Path(args.template) if args.template else \
        Path(__file__).with_name("report_template.tex")
    if not template_path.exists():
        sys.exit(f"Report template not found: {template_path}")
    tx, pm = tex_escape, r"$\pm$"

    matched, rejected, library = ctx["matched"], ctx["rejected"], ctx["library"]
    cal, energy, counts = ctx["cal"], ctx["energy"], ctx["counts"]
    diffs, dds, fwhms = ctx["diffs"], ctx["dds"], ctx["fwhms"]
    e_cal_diff = -cal["resid"]                      # E_cal - E_ref
    now = datetime.datetime.now().astimezone()
    spec_path = Path(args.spectrum)
    sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()[:16]
    worst = ctx["worst"]
    max_diff = float(np.abs(diffs).max())
    ok = max_diff <= args.tolerance
    date_str = now.strftime("%d %B %Y, %H:%M %Z")

    verdict = (r"\noindent\textbf{Result:} "
               + (r"\textcolor{passgreen}{\textbf{PASS}}" if ok
                  else r"\textcolor{failred}{\textbf{FAIL}}")
               + rf" --- the largest $|E_\mathrm{{fit}}-E_\mathrm{{ref}}|$ is "
                 rf"{max_diff:.3f}~keV ({tx(worst['nuclide'])} {worst['E']:.2f}~keV) "
                 rf"against a tolerance of {args.tolerance:g}~keV.")

    meta_table = tex_kv([
        ("Date", tx(date_str)),
        ("Operator", tx(args.operator) or "---"),
        ("Detector", tx(args.detector) or "---"),
        ("Spectrum file", rf"\texttt{{{tx(spec_path.name)}}}"),
        ("File checksum (SHA-256, first 16)", rf"\texttt{{{sha}}}"),
        ("Notes", tx(args.notes) or "---"),
    ])
    spectrum_table = tex_kv([
        ("Channels", f"{len(energy)} ({energy[0]:g}--{energy[-1]:g}~keV, "
                     f"{energy[1] - energy[0]:g}~keV per channel)"),
        ("Total counts", f"{counts.sum():,.0f}".replace(",", r"\,")),
        ("Highest channel", f"{counts.max():,.0f}".replace(",", r"\,")
         + f" counts at {energy[int(np.argmax(counts))]:g}~keV"),
    ])

    # sources
    fitted_by_nuc = {}
    for m in matched:
        fitted_by_nuc.setdefault(m["nuclide"], []).append(m)
    src_rows = []
    for nuc in args.nuclides:
        n_lib = sum(1 for l in library if l["nuclide"] == nuc)
        fl = fitted_by_nuc.get(nuc, [])
        src_rows.append([tx(nuc), n_lib, len(fl),
                         ", ".join(f"{m['E']:.1f}" for m in fl) or "---"])
    sources_table = tex_table(
        ["Nuclide", "Library lines in range", "Lines fitted", "Energies fitted (keV)"],
        src_rows, r"lrr>{\raggedright\arraybackslash}p{8cm}")
    if rejected:
        items = "\n".join(rf"  \item {tx(l['nuclide'])} {l['E']:.2f}~keV --- {tx(why)}"
                          for l, why in rejected)
        not_used = ("Library lines not used in the calibration:\n"
                    f"\\begin{{itemize}}\n{items}\n\\end{{itemize}}")
    else:
        not_used = r"{\small All library lines in range were fitted.}"

    # settings
    p = cal["order"] + 1
    form = " + ".join(["c_0"] + [f"c_{k}E" if k == 1 else f"c_{k}E^{k}" for k in range(1, p)])
    settings_table = tex_kv([
        ("Peak model", "bin-integrated Gaussian(s) + quadratic background"),
        ("Peak-fit weights", r"Poisson, $1/\sqrt{N}$ per channel ($N\geq 1$)"),
        ("Seed search", rf"accepted energies shifted by up to $\pm${args.tol:g}~keV"),
        ("Multiplet rule", rf"lines closer than {MULTIPLET_FWHM:g}$\times$FWHM = "
                           rf"{MULTIPLET_FWHM * ctx['fwhm_used']:.1f}~keV fitted together "
                           "(shared width)"),
        ("FWHM: start / used", f"{args.fwhm:g}~keV / {ctx['fwhm_used']:.2f}~keV"),
        ("Fit half-window", f"{ctx['half_window']:.1f}~keV beyond the outermost peak"),
        ("Lowest line fitted", f"{args.emin:g}~keV"),
        ("Minimum significance (area/error)", f"{args.min_sig:g}"),
        ("Maximum centroid uncertainty", f"{args.max_dmu:g}~keV"),
        ("Calibration polynomial", rf"order {cal['order']} "
                                   r"($E_\mathrm{true} = f(E_\mathrm{axis})$)"),
        ("Calibration weights", r"$1/(\sigma_\mathrm{centroid}^2+\sigma_\mathrm{ref}^2)$"),
        ("Pass/fail tolerance", rf"{args.tolerance:g}~keV on "
                                r"$|E_\mathrm{fit}-E_\mathrm{ref}|$"),
    ])

    # peak table
    peak_rows, any_multi = [], False
    for m, d, dd, ec in zip(matched, diffs, dds, e_cal_diff):
        multi = m["group_size"] > 1
        any_multi = any_multi or multi
        ref = f"{m['E']:.3f}" + (rf" {pm} {m['dE']:.4f}" if m["dE"] > 0 else "")
        peak_rows.append([tx(m["nuclide"]) + (r"$^\dagger$" if multi else ""),
                          ref, rf"{m['mu']:.3f} {pm} {m['dmu']:.3f}",
                          tn(f"{d:+.3f}"), tn(f"{d / dd:+.1f}"), f"{m['fwhm']:.2f}",
                          f"{m['area']:,.0f}".replace(",", r"\,"), f"{m['chi2ndf']:.2f}",
                          tn(f"{ec:+.3f}")])
    peak_table = ("Energies, differences and FWHM in keV.\\par\n" + tex_table(
        ["Nuclide", r"$E_\mathrm{ref}$", r"$E_\mathrm{fit}$",
         r"$E_\mathrm{fit}-E_\mathrm{ref}$", "Pull", "FWHM", "Area",
         r"$\chi^2/\mathrm{ndf}$", r"$E_\mathrm{cal}-E_\mathrm{ref}$"],
        peak_rows, "lrrrrrrrr", long=True))
    if any_multi:
        peak_table += ("\n\\par{\\footnotesize $^\\dagger$ fitted together with a neighbouring "
                       "line (multiplet, shared width).}")

    agreement_table = tex_kv([
        ("Peaks fitted", f"{len(matched)} of {len(library)} library lines"),
        ("Mean difference", tn(f"{diffs.mean():+.4f}") + "~keV"),
        ("RMS difference", f"{np.sqrt(np.mean(diffs ** 2)):.4f}~keV"),
        ("Largest $|$difference$|$", f"{max_diff:.4f}~keV ({tx(worst['nuclide'])} "
                                     f"{worst['E']:.2f}~keV)"),
        (r"$\chi^2$/ndf of differences about zero",
         f"{np.sum((diffs / dds) ** 2) / len(diffs):.2f}"),
    ])

    # resolution
    res = fit_resolution(matched)
    resolution = tex_kv([
        ("Mean FWHM", f"{fwhms.mean():.2f}~keV"),
        ("Median FWHM", f"{np.median(fwhms):.2f}~keV"),
        ("Range", f"{fwhms.min():.2f}--{fwhms.max():.2f}~keV"),
    ])
    if res:
        a, b = res["coef"]
        resolution += ("\n\\par\n" + rf"Resolution curve fitted to {res['n']} independent widths: "
                       rf"FWHM$^2 = A + B\,E$ with $A = {tn(f'{a:.3f}')}$~keV$^2$ and "
                       rf"$B =$ {tex_sci(b)}~keV.")
        rows = []
        for e in (122, 662, 1332):
            if res["emin"] <= e <= res["emax"]:
                f, sf = resolution_at(res, e)
                rows.append([str(e), rf"{f:.2f} {pm} {sf:.2f}"])
        if rows:
            resolution += "\n\\par\n" + tex_table(["$E$ (keV)", "FWHM (keV)"], rows, "rr")
    else:
        resolution += "\n\\par{\\small Too few independent peaks for a resolution-curve fit.}"

    # calibration
    sd = np.sqrt(np.diag(cal["cov"]))
    cal_table = tex_table(
        ["Coefficient", "Value", "Uncertainty"],
        [[f"$c_{k}$", tex_num(f"{cal['coef'][k]:+.6g}"), tex_num(f"{sd[k]:.2g}")]
         for k in range(p)], "lrr")
    extra = [("Calibration form", "$E_\\mathrm{true} = " + form + "$ ($E$ = axis energy)"),
             (r"$\chi^2$/ndf", f"{cal['chi2ndf']:.2f} (ndf = {cal['ndf']})"),
             ("Error scaling", rf"$\times$ {ctx['scale']:.2f}"
              + (r" ($\chi^2$/ndf $>1$)" if ctx["scale"] > 1 else " (none needed)")),
             ("RMS scatter about the fit", f"{np.sqrt(np.mean(cal['resid'] ** 2)):.4f}~keV")]
    if p >= 2:
        extra.insert(1, ("Correlation $c_0$--$c_1$",
                         tn(f"{cal['cov'][0, 1] / (sd[0] * sd[1]):+.3f}")))
    cal_table += "\n\n\\vspace{0.4em}\n" + tex_kv(extra)
    cal_error_table = ("\\noindent\\begin{minipage}{\\textwidth}\n"   # keep caption + table together
                       "$1\\sigma$ uncertainty of the calibrated energy:\\par\n" + tex_table(
        [r"$E_\mathrm{axis}$ (keV)", r"$f(E)-E$ (keV)", r"$\sigma$ (keV)"],
        [[f"{e:.0f}", tn(f"{f - e:+.4f}"), f"{sg:.4f}"] for e, f, sg in zip(*ctx["probe"])],
        "rrr") + "\n\\end{minipage}")

    flags = collect_flags(ctx)
    flags_tex = ("\\begin{itemize}\n" + "\n".join(f"  \\item {tx(f)}" for f in flags)
                 + "\n\\end{itemize}") if flags else "None."

    import matplotlib
    import scipy
    software = (f"Python {platform.python_version()}, NumPy {np.__version__}, "
                f"SciPy {scipy.__version__}, Matplotlib {matplotlib.__version__}")

    values = dict(
        title=tx(args.title), date=tx(date_str), detector=tx(args.detector) or "---",
        meta_table=meta_table, verdict=verdict, spectrum_table=spectrum_table,
        sources_table=sources_table, not_used=not_used, settings_table=settings_table,
        figure_file=figure_name, peak_table=peak_table, agreement_table=agreement_table,
        resolution=resolution, cal_table=cal_table, cal_error_table=cal_error_table,
        flags=flags_tex, software=tx(software))
    template = template_path.read_text(encoding="utf-8")
    return re.sub(r"<<\s*(\w+)\s*>>", lambda m: str(values.get(m.group(1), m.group(0))), template)


def write_tex_files(args, ctx, figs, out):
    """Write <out>.tex plus the figure PDF it includes next to it."""
    fig_name = f"{out.stem}_figure.pdf"
    out.with_name(fig_name).write_bytes(figs["pdf"])
    out.write_text(build_tex(args, ctx, fig_name), encoding="utf-8")
    return fig_name


def write_report_pdf(args, ctx, figs, out):
    """Build the LaTeX report and compile it with pdflatex (two passes, so
    the 'page x of y' footer is right)."""
    import shutil
    import subprocess
    import tempfile

    engine = shutil.which("pdflatex")
    if engine is None:
        tex_out = out.with_suffix(".tex")
        write_tex_files(args, ctx, figs, tex_out)
        print(f"pdflatex not found: wrote {tex_out} instead. Compile it with "
              f"`pdflatex {tex_out.name}` on a machine with LaTeX installed.")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "figure.pdf").write_bytes(figs["pdf"])
        (tmp / "report.tex").write_text(build_tex(args, ctx, "figure.pdf"), encoding="utf-8")
        for _ in range(2):
            run = subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error",
                                  "report.tex"], cwd=tmp, capture_output=True, text=True)
            if run.returncode != 0:
                tex_out = out.with_suffix(".tex")
                write_tex_files(args, ctx, figs, tex_out)
                sys.exit(f"pdflatex failed; LaTeX source kept as {tex_out}. "
                         f"End of the log:\n{run.stdout[-1500:]}")
        shutil.copyfile(tmp / "report.pdf", out)
    print(f"PDF report written to {out}")
    if args.keep_tex:
        tex_out = out.with_suffix(".tex")
        write_tex_files(args, ctx, figs, tex_out)
        print(f"LaTeX source kept as {tex_out}")


def write_report(args, ctx, figs):
    """Write the report in the format given by the file extension:
    .html (self-contained), .tex (+ figure PDF) or .pdf (built with pdflatex)."""
    out = Path(args.report)
    ext = out.suffix.lower()
    if ext in (".html", ".htm"):
        write_report_html(args, ctx, figs)
    elif ext == ".tex":
        fig_name = write_tex_files(args, ctx, figs, out)
        print(f"LaTeX report written to {out} (figure: {fig_name}); "
              f"compile with `pdflatex {out.name}`")
    elif ext == ".pdf":
        write_report_pdf(args, ctx, figs, out)
    else:
        sys.exit("--report file must end in .html, .tex or .pdf")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("spectrum", help="two-column ASCII file: E(keV) counts")
    ap.add_argument("--nuclides", nargs="+", default=DEFAULT_NUCLIDES,
                    help="nuclides whose accepted lines are fitted")
    ap.add_argument("--tol", type=float, default=5.0,
                    help="search +/- this many keV around each accepted energy")
    ap.add_argument("--fwhm", type=float, default=3.0,
                    help="expected peak FWHM in keV (starting value; refined "
                         "from the data). Lines closer than 4 x FWHM are fitted "
                         "together as a multiplet")
    ap.add_argument("--window", type=float, default=None,
                    help="fit half-window in keV beyond the outermost peak "
                         "(default: 3 x FWHM)")
    ap.add_argument("--min-sig", type=float, default=5.0,
                    help="minimum area/uncertainty for a peak to count as detected")
    ap.add_argument("--max-dmu", type=float, default=0.2,
                    help="drop fits whose centroid uncertainty exceeds this, keV")
    ap.add_argument("--skip-low", action="store_true",
                    help=f"ignore library lines below {LOW_ENERGY_CUT:g} keV "
                         "(shortcut for --emin 50)")
    ap.add_argument("--emin", type=float, default=10.0,
                    help="ignore library lines below this energy, keV")
    ap.add_argument("--order", type=int, choices=[1, 2], default=1,
                    help="polynomial order of the calibration used for the error "
                         "band and error analysis (default 1 = linear)")
    ap.add_argument("--residuals", action="store_true",
                    help="plot points relative to the fitted calibration "
                         "(E_cal - E_ref, band centred on zero)")
    ap.add_argument("--report", metavar="FILE",
                    help="write a report; format from the extension: .html "
                         "(figure embedded), .tex, or .pdf (built with LaTeX)")
    ap.add_argument("--template", metavar="FILE",
                    help="report template (default: report_template.html or "
                         "report_template.tex next to this script)")
    ap.add_argument("--keep-tex", action="store_true",
                    help="with a .pdf report, also keep the .tex source and figure")
    ap.add_argument("--tolerance", type=float, default=0.5,
                    help="PASS/FAIL limit on max |E_fit - E_ref| in the report, "
                         "keV (default 0.5; set to your own criterion)")
    ap.add_argument("--title", default="Energy calibration check",
                    help="report title")
    ap.add_argument("--operator", default="", help="report: operator name")
    ap.add_argument("--detector", default="", help="report: detector / system ID")
    ap.add_argument("--notes", default="", help="report: free-text notes")
    ap.add_argument("--plot", metavar="PNG",
                    help="also save the figure to this PNG file")
    ap.add_argument("--no-show", action="store_true",
                    help="don't open the matplotlib window (file output only)")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip drawing the plot entirely")
    args = ap.parse_args()
    if args.skip_low:
        args.emin = max(args.emin, LOW_ENERGY_CUT)

    energy, counts = read_spectrum(args.spectrum)
    print(f"Read {len(energy)} channels from {args.spectrum} "
          f"({energy[0]:g} - {energy[-1]:g} keV, {counts.sum():,.0f} counts)")

    library = [l for l in reference_lines(args.nuclides)
               if args.emin <= l["E"] <= energy[-1] - args.tol]
    fwhm = args.fwhm
    groups, matched, rejected = run_fits(energy, counts, library, args, fwhm)
    if len(matched) >= 3:
        measured = float(np.median([m["fwhm"] for m in matched]))
        if abs(measured - fwhm) > 0.05 * fwhm:
            print(f"FWHM guess {fwhm:.2f} keV, measured {measured:.2f} keV: "
                  f"refitting with the measured value")
            fwhm = measured
            groups, matched, rejected = run_fits(energy, counts, library, args, fwhm)

    print(f"{len(library)} library lines in {len(groups)} fit regions "
          f"(lines closer than {MULTIPLET_FWHM:g} x FWHM = "
          f"{MULTIPLET_FWHM * fwhm:.1f} keV are fitted together)")
    for g in groups:
        if len(g) > 1:
            print(f"  {multiplet_name(len(g))}: "
                  + ", ".join(f"{l['nuclide']} {l['E']:.2f}" for l in g))

    if len(matched) < 3:
        sys.exit("Fewer than 3 peaks fitted successfully; cannot assess calibration.")
    matched.sort(key=lambda m: m["E"])

    cal1 = fit_calibration(matched, 1)
    cal2 = fit_calibration(matched, 2) if len(matched) > 3 else None

    # ---- per-peak table
    print()
    print("Peak-by-peak comparison with literature (E_fit - E_ref):")
    hdr = (f"{'Nuclide':<7}{'E_ref(keV)':>13}{'E_fit(keV)':>15}{'diff(keV)':>13}"
           f"{'pull':>7}{'FWHM':>7}{'area':>10}{'chi2/ndf':>10}{'fit':>10}{'resid(lin)':>12}")
    print(hdr)
    print("-" * len(hdr))
    for m, r in zip(matched, cal1["resid"]):
        diff = m["mu"] - m["E"]
        dd = np.hypot(m["dmu"], m["dE"])
        print(f"{m['nuclide']:<7}{m['E']:>13.3f}{m['mu']:>10.3f}({m['dmu']*1000:3.0f}) "
              f"{diff:>+10.3f}{diff/dd:>+8.1f}{m['fwhm']:>7.2f}{m['area']:>10.0f}"
              f"{m['chi2ndf']:>10.2f}{multiplet_name(m['group_size']):>10}{r:>+12.3f}")

    # ---- summary
    diffs = np.array([m["mu"] - m["E"] for m in matched])
    dds = np.array([np.hypot(m["dmu"], m["dE"]) for m in matched])
    fwhms = np.array([m["fwhm"] for m in matched])
    worst = matched[int(np.argmax(np.abs(diffs)))]
    print()
    print(f"Fitted {len(matched)} of {len(library)} library lines.")
    for line, why in rejected:
        print(f"  not used: {line['nuclide']} {line['E']:.2f} keV ({why})")

    print()
    print("Agreement with literature energies (as supplied, no recalibration):")
    print(f"  mean difference   = {diffs.mean():+.4f} keV")
    print(f"  RMS difference    = {np.sqrt(np.mean(diffs**2)):.4f} keV")
    print(f"  max |difference|  = {np.abs(diffs).max():.4f} keV "
          f"({worst['nuclide']} {worst['E']:.2f})")
    print(f"  chi2/ndf (diff vs 0) = {np.sum((diffs/dds)**2)/len(diffs):.2f}")
    print(f"  fitted FWHM: mean {fwhms.mean():.2f} keV "
          f"(range {fwhms.min():.2f} - {fwhms.max():.2f})")

    c = cal1["coef"]; e = cal1["err"]
    print()
    print("Linear fit  E_true = c0 + c1 * E_axis :")
    print(f"  c0 = {c[0]:+.4f} +/- {e[0]:.4f} keV")
    print(f"  c1 = {c[1]:.6f} +/- {e[1]:.6f}   (chi2/ndf = {cal1['chi2ndf']:.2f}, "
          f"ndf = {cal1['ndf']})")
    print(f"  max residual after linear correction = {np.abs(cal1['resid']).max():.4f} keV")
    if cal2:
        c = cal2["coef"]; e = cal2["err"]
        print("Quadratic fit  E_true = c0 + c1*E + c2*E^2 :")
        print(f"  c0 = {c[0]:+.4f} +/- {e[0]:.4f} keV")
        print(f"  c1 = {c[1]:.6f} +/- {e[1]:.6f}")
        print(f"  c2 = {c[2]:+.3e} +/- {e[2]:.1e} keV^-1   "
              f"(chi2/ndf = {cal2['chi2ndf']:.2f}, ndf = {cal2['ndf']})")

    # ---- error analysis of the chosen calibration
    cal = cal2 if (args.order == 2 and cal2) else cal1
    if args.order == 2 and not cal2:
        print("\nNeed more than 3 fitted peaks for a quadratic calibration; using linear.")
    p = cal["order"] + 1
    cov = cal["cov"]
    sd = np.sqrt(np.diag(cov))
    scale = np.sqrt(max(1.0, cal["chi2ndf"]))
    print()
    print(f"Error analysis of the order-{cal['order']} calibration  E_true = f(E_axis):")
    print(f"  parameter uncertainties (quoted):  "
          + ", ".join(f"c{k} = {cal['coef'][k]:+.6g} +/- {sd[k]:.2g}" for k in range(p)))
    if p >= 2:
        print(f"  correlation c0-c1 = {cov[0, 1] / (sd[0] * sd[1]):+.3f}")
    print(f"  chi2/ndf = {cal['chi2ndf']:.2f} (ndf = {cal['ndf']}); "
          + (f"errors below are scaled by sqrt(chi2/ndf) = {scale:.2f}"
             if scale > 1 else "no error scaling needed (chi2/ndf <= 1)"))
    print(f"  RMS scatter of residuals about the fit = "
          f"{np.sqrt(np.mean(cal['resid'] ** 2)):.4f} keV")
    probe = [e for e in (20, 50, 100, 250, 500, 750, 1000, 1500, 2000)
             if energy[0] <= e <= energy[-1]]
    f_p, s_p = calibration_curve(cal, probe)
    print("  1-sigma uncertainty of the calibrated energy:")
    print(f"    {'E_axis(keV)':>12}{'f(E)-E (keV)':>15}{'sigma (keV)':>13}")
    for e, f, sg in zip(probe, f_p, s_p):
        print(f"    {e:>12.0f}{f - e:>+15.4f}{sg:>13.4f}")

    figs = None
    if not args.no_plot or args.report:
        show = not args.no_show and not args.no_plot
        path = args.plot
        if not args.no_plot and args.no_show and not path:
            path = "calibration_check.png"
        figs = make_plot(path, energy, counts, matched, cal, show=show,
                         relative=args.residuals, return_figs=bool(args.report))

    if args.report:
        ctx = dict(energy=energy, counts=counts, library=library, matched=matched,
                   rejected=rejected, cal=cal, diffs=diffs, dds=dds, fwhms=fwhms,
                   worst=worst, scale=scale, probe=(probe, f_p, s_p),
                   fwhm_used=fwhm,
                   half_window=args.window if args.window else WINDOW_FWHM * fwhm)
        write_report(args, ctx, figs)


if __name__ == "__main__":
    main()
