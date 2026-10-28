"""Tests for gnss_raim. The statistical ones use fixed seeds and binomial bounds
of about 4 sigma, so they are deterministic and not flaky."""

import numpy as np
import pytest
from scipy import stats

from gnss_raim import (RAIM, RAIMConfig, RAIMStatus, Satellite, create_satellite_geometry,
                       ecef_to_geodetic, enu_rotation, simulate_pseudoranges, stanford_region)

RX = np.array([-742000.0, -5462000.0, 3198000.0])  # ECEF, ~30 N, 98 W


def binomial_ok(k, n, p, nsig=4.0):
    return abs(k / n - p) <= nsig * np.sqrt(p * (1 - p) / n)


def sats_from_enu(rx, dirs_enu, sigma=1.0):
    R = enu_rotation(rx)
    out = []
    for i, d in enumerate(dirs_enu, start=1):
        p = rx + 2.0e7 * (R.T @ d)
        out.append(Satellite(prn=i, x=p[0], y=p[1], z=p[2], pseudorange=0.0, sigma=sigma))
    return out


# --- frames -----------------------------------------------------------------

def test_enu_rotation_is_orthonormal_and_up_is_ellipsoid_normal():
    R = enu_rotation(RX)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    lat, lon, h = ecef_to_geodetic(RX)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    assert np.allclose(R[2], up, atol=1e-12)


def test_geodetic_round_trip():
    lat, lon, h = np.radians(40.0), np.radians(-105.0), 1650.0
    n = 6378137.0 / np.sqrt(1 - 6.69437999014e-3 * np.sin(lat) ** 2)
    p = np.array([(n + h) * np.cos(lat) * np.cos(lon),
                  (n + h) * np.cos(lat) * np.sin(lon),
                  (n * (1 - 6.69437999014e-3) + h) * np.sin(lat)])
    lat2, lon2, h2 = ecef_to_geodetic(p)
    assert lat2 == pytest.approx(lat, abs=1e-11)
    assert lon2 == pytest.approx(lon, abs=1e-12)
    assert h2 == pytest.approx(h, abs=1e-4)


# --- DOP --------------------------------------------------------------------

@pytest.mark.parametrize("n,el_deg", [(4, 30.0), (6, 15.0), (8, 45.0)])
def test_dop_matches_closed_form(n, el_deg):
    """One satellite at zenith plus n evenly spaced in azimuth at one elevation:
    HDOP = 2 / (cos(el) sqrt(n)),  VDOP = sqrt((n + 1) / n) / (1 - sin(el))."""
    el = np.radians(el_deg)
    dirs = [np.array([0.0, 0.0, 1.0])]
    for k in range(n):
        az = 2 * np.pi * k / n
        dirs.append(np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)]))
    raim = RAIM()
    H = raim.compute_geometry_matrix(sats_from_enu(RX, dirs), RX)
    dop = raim.compute_dop(H, RX)
    assert dop["hdop"] == pytest.approx(2 / (np.cos(el) * np.sqrt(n)), rel=1e-9)
    assert dop["vdop"] == pytest.approx(np.sqrt((n + 1) / n) / (1 - np.sin(el)), rel=1e-9)


# --- positioning ------------------------------------------------------------

def test_noise_free_solution_recovers_truth():
    raim = RAIM()
    sats = simulate_pseudoranges(RX, create_satellite_geometry(RX, 7, seed=3),
                                 true_clock_bias=1234.5, noise_std=1e-9, seed=4)
    pos, cb, res = raim.solve_position(sats)
    assert np.linalg.norm(pos - RX) < 1e-4
    assert cb == pytest.approx(1234.5, abs=1e-4)


# --- detection --------------------------------------------------------------

@pytest.mark.parametrize("weighted_sigmas", [False, True])
def test_false_alarm_rate_matches_pfa(weighted_sigmas):
    pfa, n_trials = 0.01, 4000
    raim = RAIM(RAIMConfig(pfa=pfa, pmd=1e-3))
    rng = np.random.default_rng(10)
    geom = create_satellite_geometry(RX, 8, seed=11)
    sig = np.linspace(0.5, 3.0, 8) if weighted_sigmas else 1.5
    alarms = sum(
        raim.detect_fault(simulate_pseudoranges(RX, geom, noise_std=sig, rng=rng), RX).status
        == RAIMStatus.FAULT_DETECTED
        for _ in range(n_trials)
    )
    assert binomial_ok(alarms, n_trials, pfa), f"{alarms}/{n_trials} false alarms"


def test_detects_a_50m_bias():
    raim = RAIM(RAIMConfig(pfa=1e-5, pmd=1e-3))
    rng = np.random.default_rng(20)
    geom = create_satellite_geometry(RX, 8, seed=21)
    detected = sum(
        raim.detect_fault(simulate_pseudoranges(RX, geom, noise_std=1.0, fault_prn=prn,
                                                 fault_magnitude=50.0, rng=rng), RX).status
        == RAIMStatus.FAULT_DETECTED
        for prn in range(1, 9) for _ in range(25)
    )
    assert detected == 200


def test_excludes_the_faulty_prn():
    raim = RAIM(RAIMConfig(pfa=1e-5, pmd=1e-3))
    rng = np.random.default_rng(30)
    correct, trials = 0, 0
    for g in range(20):
        geom = create_satellite_geometry(RX, 9, rng=rng)
        for prn in (1, 5, 9):
            sats = simulate_pseudoranges(RX, geom, noise_std=1.0, fault_prn=prn,
                                         fault_magnitude=60.0, rng=rng)
            res = raim.exclude_fault(sats, RX)
            trials += 1
            correct += (res.excluded_prns == [prn] and res.status == RAIMStatus.FAULT_EXCLUDED)
    assert correct / trials >= 0.95, f"{correct}/{trials}"


# --- protection levels ------------------------------------------------------

def test_p_bias_gives_exactly_pmd():
    cfg = RAIMConfig(pfa=1e-5, pmd=1e-3)
    raim = RAIM(cfg)
    sats = simulate_pseudoranges(RX, create_satellite_geometry(RX, 8, seed=40), seed=41)
    res = raim.detect_fault(sats, RX)
    dof = 4
    t2 = stats.chi2.ppf(1 - cfg.pfa, dof)
    assert stats.ncx2.cdf(t2, dof, res.p_bias**2) == pytest.approx(cfg.pmd, rel=1e-6)


def test_bias_at_the_protection_level_is_missed_at_rate_pmd():
    """Put the bias on the worst-slope satellite, sized so that its mean
    horizontal error equals HPL. It should go undetected with probability Pmd."""
    pmd, n_trials = 0.05, 3000
    raim = RAIM(RAIMConfig(pfa=1e-3, pmd=pmd))
    rng = np.random.default_rng(50)
    geom = create_satellite_geometry(RX, 8, seed=51)
    clean = simulate_pseudoranges(RX, geom, noise_std=1.0, seed=52)
    H = raim.compute_geometry_matrix(clean, RX)
    W = raim.compute_weight_matrix(clean)
    hpl, _, _, _, p_bias = raim.compute_protection_levels(H, W, len(clean) - 4, RX)
    hslope, _ = raim.compute_slopes(H, W, RX)
    i = int(np.argmax(hslope))
    A = np.linalg.solve(H.T @ W @ H, H.T @ W)
    S = np.eye(len(clean)) - H @ A
    bias = p_bias / np.sqrt(W[i, i] * S[i, i])

    R = enu_rotation(RX)
    assert np.linalg.norm((R @ A[:3, i])[:2]) * bias == pytest.approx(hpl, rel=1e-9)

    missed, hmi = 0, 0
    for _ in range(n_trials):
        sats = simulate_pseudoranges(RX, geom, noise_std=1.0, fault_prn=i + 1,
                                     fault_magnitude=bias, rng=rng)
        res = raim.detect_fault(sats, RX)
        if res.status != RAIMStatus.FAULT_DETECTED:
            missed += 1
            hpe = np.linalg.norm((R @ (res.position - RX))[:2])
            hmi += hpe > res.hpl
    assert binomial_ok(missed, n_trials, pmd), f"missed {missed}/{n_trials}"
    assert hmi <= missed


def test_hpl_does_not_depend_on_the_alert_limit():
    sats = simulate_pseudoranges(RX, create_satellite_geometry(RX, 8, seed=60), seed=61)
    a = RAIM(RAIMConfig.lnav()).detect_fault(sats, RX)
    b = RAIM(RAIMConfig.lpv_like()).detect_fault(sats, RX)
    assert a.hpl == pytest.approx(b.hpl)


def test_pl_above_al_is_unavailable_not_misleading():
    raim = RAIM(RAIMConfig(hal=1.0, val=1.0))
    sats = simulate_pseudoranges(RX, create_satellite_geometry(RX, 8, seed=70), seed=71)
    res = raim.detect_fault(sats, RX)
    assert res.hpl > 1.0
    assert raim.compute_integrity_risk(res)["availability"] == "unavailable"


@pytest.mark.parametrize("pe,pl,al,region", [
    (5, 10, 40, "normal"),
    (20, 10, 40, "misleading"),
    (50, 10, 40, "hazardous"),
    (30, 60, 40, "unavailable"),
    (80, 60, 40, "unavailable_mi"),
])
def test_stanford_regions(pe, pl, al, region):
    assert stanford_region(pe, pl, al) == region


def test_too_few_satellites_is_unavailable():
    raim = RAIM()
    sats = simulate_pseudoranges(RX, create_satellite_geometry(RX, 4, seed=80), seed=81)
    assert raim.detect_fault(sats).status == RAIMStatus.UNAVAILABLE
