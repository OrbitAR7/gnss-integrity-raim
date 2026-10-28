#!/usr/bin/env python3
"""
Regenerate the README figures in docs/:

- stanford_diagram.png: Monte Carlo Stanford diagram (HPL vs. true horizontal
  error) after fault detection and exclusion, random geometries, 10% of epochs
  with a single-satellite fault
- protection_levels.png: HPL/VPL vs. number of satellites over random
  geometries, against LPV-like and LNAV alert limits

Also prints the availability table used in the README.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from gnss_raim import (RAIM, RAIMConfig, RAIMStatus, create_satellite_geometry,
                       enu_rotation, simulate_pseudoranges, stanford_region)

RX = np.array([-742000.0, -5462000.0, 3198000.0])
HAL_LPV, VAL_LPV, HAL_LNAV = 40.0, 50.0, 556.0
OUT = Path(__file__).resolve().parent / "docs"

COLORS = {
    'normal': '#4daf4a',
    'misleading': '#ff7f00',
    'hazardous': '#e41a1c',
    'unavailable': '#984ea3',
    'hpl': '#377eb8',
    'vpl': '#e41a1c',
}


def elevation_sigma(sats):
    """1-sigma pseudorange error growing toward the horizon."""
    return [1.0 / np.sin(np.radians(s.elevation)) for s in sats]


def stanford_diagram(n_epochs=20000, fault_rate=0.10, seed=2026):
    rng = np.random.default_rng(seed)
    raim = RAIM(RAIMConfig.lpv_like(pfa=1e-5, pmd=1e-3))
    R = enu_rotation(RX)

    hpl, hpe, n_alert, n_faulty = [], [], 0, 0
    for _ in range(n_epochs):
        geom = create_satellite_geometry(RX, int(rng.integers(6, 11)), min_elevation=10.0, rng=rng)
        faulty = rng.random() < fault_rate
        n_faulty += faulty
        sats = simulate_pseudoranges(
            RX, geom, noise_std=elevation_sigma(geom),
            fault_prn=int(rng.integers(1, len(geom) + 1)) if faulty else None,
            fault_magnitude=float(rng.uniform(0, 200)) if faulty else 0.0, rng=rng)
        res = raim.exclude_fault(sats, RX)
        if res.status in (RAIMStatus.FAULT_DETECTED, RAIMStatus.UNAVAILABLE):
            n_alert += 1          # alert raised, no position used
            continue
        hpl.append(res.hpl)
        hpe.append(np.linalg.norm((R @ (res.position - RX))[:2]))

    hpl, hpe = np.array(hpl), np.array(hpe)
    regions = [stanford_region(e, p, HAL_LPV) for e, p in zip(hpe, hpl)]
    counts = {k: regions.count(k) for k in
              ("normal", "misleading", "hazardous", "unavailable", "unavailable_mi")}

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    top = 120
    bins = np.linspace(0, top, 121)
    h = ax.hist2d(hpl, hpe, bins=[bins, bins], norm=LogNorm(), cmap="viridis", cmin=1)
    fig.colorbar(h[3], ax=ax, label="epochs per 1 m x 1 m bin")

    ax.plot([0, top], [0, top], color="k", lw=1)
    ax.axvline(HAL_LPV, color="k", ls="--", lw=1.2)
    ax.axhline(HAL_LPV, color="k", ls="--", lw=1.2, xmax=HAL_LPV / top)
    label = dict(fontsize=10, fontweight="600", ha="center", va="center")
    ax.text(34, 24, f"Normal\noperation\n{counts['normal']}", color=COLORS['normal'], **label)
    ax.text(12, 33, f"MI\n{counts['misleading']}", color=COLORS['misleading'], **label)
    ax.text(20, 80, f"HMI\n{counts['hazardous']}", color=COLORS['hazardous'], **label)
    ax.text(85, 25, f"Unavailable\n{counts['unavailable']}", color=COLORS['unavailable'], **label)
    ax.text(62, 105, f"Unavailable + MI\n{counts['unavailable_mi']}", color=COLORS['unavailable'], **label)
    ax.text(HAL_LPV + 1, top - 3, "HAL = 40 m", fontsize=9, va="top")

    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.set_xlabel("Horizontal protection level, HPL (m)")
    ax.set_ylabel("Horizontal position error, HPE (m)")
    alerts = (f"{n_alert} epochs left in alert (not plotted)" if n_alert
              else "every detected fault was excluded")
    ax.set_title(f"Stanford diagram: {n_epochs} epochs after FDE\n"
                 f"6-10 satellites, {int(fault_rate * 100)}% of epochs with a 0-200 m fault, "
                 f"{alerts}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "stanford_diagram.png", dpi=200, facecolor="white")
    plt.close(fig)

    print(f"Stanford diagram: {len(hpl)} plotted, {n_alert} alerts, {n_faulty} faulty epochs")
    print("  " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"  max HPE/HPL = {np.max(hpe / hpl):.2f}")


def protection_levels(n_geometries=500, seed=7):
    rng = np.random.default_rng(seed)
    raim = RAIM(RAIMConfig(pfa=1e-5, pmd=1e-3))
    ns = np.arange(5, 13)
    H = np.zeros((len(ns), n_geometries))
    V = np.zeros_like(H)
    for a, n in enumerate(ns):
        for b in range(n_geometries):
            geom = create_satellite_geometry(RX, int(n), min_elevation=10.0, rng=rng)
            sats = simulate_pseudoranges(RX, geom, noise_std=elevation_sigma(geom), rng=rng)
            r = raim.detect_fault(sats, RX)
            H[a, b], V[a, b] = r.hpl, r.vpl

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for arr, name, c in ((H, "HPL", COLORS['hpl']), (V, "VPL", COLORS['vpl'])):
        med = np.median(arr, axis=1)
        lo, hi = np.percentile(arr, [10, 90], axis=1)
        ax.fill_between(ns, lo, hi, color=c, alpha=0.15)
        ax.plot(ns, med, "o-", color=c, label=f"{name} (median, 10-90% band)")
    ax.axhline(HAL_LPV, color=COLORS['hpl'], ls="--", label="HAL 40 m (LPV-like)")
    ax.axhline(VAL_LPV, color=COLORS['vpl'], ls="--", label="VAL 50 m (LPV-like)")
    ax.axhline(HAL_LNAV, color="k", ls=":", label="HAL 556 m (LNAV / NPA)")
    ax.set_yscale("log")
    ax.set_ylim(10, 5000)
    ax.set_xticks(ns)
    ax.set_xlabel("Number of satellites")
    ax.set_ylabel("Protection level (m)")
    ax.set_title(f"Protection levels over {n_geometries} random geometries per point "
                 "(Pfa 1e-5, Pmd 1e-3)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(OUT / "protection_levels.png", dpi=200, facecolor="white")
    plt.close(fig)

    print("\n| Satellites | median HPL (m) | HPL <= 556 m (LNAV) | HPL <= 40 m | HPL <= 40 m and VPL <= 50 m |")
    print("|---|---|---|---|---|")
    for a, n in enumerate(ns):
        print(f"| {n} | {np.median(H[a]):.0f} | {np.mean(H[a] <= HAL_LNAV):.0%} | "
              f"{np.mean(H[a] <= HAL_LPV):.0%} | {np.mean((H[a] <= HAL_LPV) & (V[a] <= VAL_LPV)):.0%} |")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    stanford_diagram()
    protection_levels()
