"""
Snapshot RAIM (Receiver Autonomous Integrity Monitoring).

Weighted least-squares positioning, chi-square fault detection on the
normalized residuals, largest-residual fault exclusion, and slope-based
horizontal/vertical protection levels.

Notation (n satellites, 4 unknowns: ECEF position + clock):
    H   n x 4 geometry matrix
    W   diag(1/sigma_i^2), sigma_i = 1-sigma pseudorange error of satellite i
    A   = (H'WH)^-1 H'W, maps pseudorange errors to state errors
    S   = I - H A, maps pseudorange errors to residuals
    WSSE = r' W r ~ chi-square(n - 4) with no fault

Author: Hamoud Alshammari
"""

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import optimize, stats


WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
GPS_ORBIT_RADIUS = 26_560_000.0  # m


class RAIMStatus(Enum):
    """RAIM algorithm status codes"""
    AVAILABLE = "available"           # no fault detected
    FAULT_DETECTED = "fault_detected" # fault detected, not excluded
    FAULT_EXCLUDED = "fault_excluded" # fault detected and excluded
    UNAVAILABLE = "unavailable"       # too few satellites for RAIM
    GEOMETRY_POOR = "geometry_poor"   # GDOP > 10


@dataclass
class Satellite:
    """One pseudorange measurement"""
    prn: int
    x: float  # ECEF X position [m]
    y: float  # ECEF Y position [m]
    z: float  # ECEF Z position [m]
    pseudorange: float  # [m]
    cn0: float = 45.0  # [dB-Hz], informational
    elevation: float = 45.0  # [deg]
    sigma: float = 1.0  # 1-sigma pseudorange error [m]; sets the weight

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class RAIMResult:
    """RAIM algorithm results"""
    status: RAIMStatus
    position: np.ndarray          # estimated ECEF position [m]
    clock_bias: float             # receiver clock bias [m]
    test_statistic: float         # WSSE (chi-square with n-4 dof under no fault)
    threshold: float              # detection threshold on WSSE
    hpl: float                    # horizontal protection level [m]
    vpl: float                    # vertical protection level [m]
    excluded_prns: List[int]      # PRNs excluded as faulty
    residuals: np.ndarray         # post-fit residuals [m]
    dop: dict                     # GDOP/PDOP/HDOP/VDOP/TDOP (HDOP/VDOP in local ENU)
    num_satellites: int           # satellites used
    pfa: float                    # probability of false alarm
    pmd: float                    # probability of missed detection
    slope_max: float              # max horizontal slope [m]
    vslope_max: float = 0.0       # max vertical slope [m]
    p_bias: float = 0.0           # sqrt of the noncentrality that gives Pmd


@dataclass
class RAIMConfig:
    """RAIM configuration.

    The defaults (HAL 40 m, VAL 50 m) are LPV-like alert limits, used here
    because they make the geometry effects visible. Snapshot RAIM was designed
    for en-route, terminal and LNAV/NPA operations, where the limits are
    horizontal only and much larger (LNAV: 0.3 NM = 556 m). See lnav() and
    lpv_like().
    """
    pfa: float = 1e-5             # probability of false alarm
    pmd: float = 1e-3             # probability of missed detection
    hal: float = 40.0             # horizontal alert limit [m]
    val: Optional[float] = 50.0   # vertical alert limit [m]; None = horizontal only
    min_satellites: int = 5       # minimum satellites for detection
    max_iterations: int = 10      # max iterations (LS and exclusion)
    convergence_threshold: float = 1e-4  # LS convergence [m]
    use_weighted: bool = True     # False: equal weights using the largest sigma
    elevation_mask: float = 5.0   # [deg], informational

    @classmethod
    def lnav(cls, **kw) -> "RAIMConfig":
        """LNAV / non-precision approach: HAL = 0.3 NM (556 m), no vertical limit."""
        return cls(hal=556.0, val=None, **kw)

    @classmethod
    def lpv_like(cls, **kw) -> "RAIMConfig":
        """LPV alert limits (HAL 40 m, VAL 50 m). LPV is an SBAS service; these are
        used to stress the geometry, not because snapshot RAIM supports LPV."""
        return cls(hal=40.0, val=50.0, **kw)


def ecef_to_geodetic(pos: np.ndarray) -> Tuple[float, float, float]:
    """WGS-84 latitude, longitude [rad] and height [m] from ECEF [m]."""
    x, y, z = pos
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    for _ in range(6):
        n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - WGS84_E2 * n / (n + h)))
    return float(lat), float(lon), float(h)


def enu_rotation(pos: np.ndarray) -> np.ndarray:
    """3x3 matrix whose rows are the local East, North, Up unit vectors (in ECEF)."""
    lat, lon, _ = ecef_to_geodetic(pos)
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([
        [-so, co, 0.0],
        [-sl * co, -sl * so, cl],
        [cl * co, cl * so, sl],
    ])


@lru_cache(maxsize=256)
def _p_bias(dof: int, pfa: float, pmd: float) -> float:
    """sqrt(lambda) such that a noncentral chi-square(dof, lambda) stays below the
    detection threshold with probability pmd."""
    t2 = stats.chi2.ppf(1 - pfa, dof)
    f = lambda lam: stats.ncx2.cdf(t2, dof, lam) - pmd
    lam = optimize.brentq(f, 1e-9, 1e4, xtol=1e-10)
    return float(np.sqrt(lam))


def stanford_region(pe: float, pl: float, al: float) -> str:
    """Classify one epoch on the Stanford diagram.

    normal       PE < PL <= AL   bound holds, service available
    misleading   PL < PE <= AL   bound violated, error still inside AL (MI)
    hazardous    PL <= AL < PE   service available, error beyond AL (HMI)
    unavailable  PL > AL, PE <= PL
    unavailable_mi  PE > PL > AL  bound violated while unavailable
    """
    if pl > al:
        return "unavailable_mi" if pe > pl else "unavailable"
    if pe <= pl:
        return "normal"
    return "hazardous" if pe > al else "misleading"


class RAIM:
    """
    Receiver Autonomous Integrity Monitoring

    Snapshot RAIM with:
    - Weighted least-squares residual (WSSE) chi-square fault detection
    - Largest-residual sequential fault exclusion
    - Slope-based horizontal and vertical protection levels
    - DOP in the local ENU frame
    """

    def __init__(self, config: Optional[RAIMConfig] = None):
        self.config = config or RAIMConfig()

    def check_availability(self, satellites: List[Satellite]) -> Tuple[bool, str]:
        """
        Check if RAIM is available given current satellite count

        Args:
            satellites: List of visible satellites

        Returns:
            (available, reason): Availability flag and reason string
        """
        n = len(satellites)
        if n < self.config.min_satellites:
            return False, f"Insufficient satellites: {n} < {self.config.min_satellites}"
        if n < 6:
            return True, "FD only (no exclusion capability)"
        return True, "Full RAIM available (FD + FDE)"

    def compute_geometry_matrix(self, satellites: List[Satellite],
                                receiver_pos: np.ndarray) -> np.ndarray:
        """
        Geometry (design) matrix H, n x 4: minus the receiver-to-satellite unit
        vectors, and 1 for the clock.
        """
        H = np.ones((len(satellites), 4))
        for i, sat in enumerate(satellites):
            los = sat.position - receiver_pos
            H[i, :3] = -los / np.linalg.norm(los)
        return H

    def compute_weight_matrix(self, satellites: List[Satellite]) -> np.ndarray:
        """
        W = diag(1/sigma_i^2) from each satellite's sigma.

        Weights must match the real measurement noise, otherwise the WSSE is not
        chi-square distributed and the false-alarm rate is not Pfa. With
        use_weighted=False every satellite gets the largest sigma.
        """
        sig = np.array([s.sigma for s in satellites], dtype=float)
        if not self.config.use_weighted:
            sig = np.full_like(sig, sig.max())
        return np.diag(1.0 / sig**2)

    def solve_position(self, satellites: List[Satellite],
                       initial_pos: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Iterative weighted least squares for position and clock.

        Args:
            satellites: List of satellite data
            initial_pos: Initial position estimate (default: center of Earth)

        Returns:
            (position, clock_bias, residuals)
        """
        pos = np.zeros(3) if initial_pos is None else np.array(initial_pos, dtype=float)
        clock_bias = 0.0
        W = self.compute_weight_matrix(satellites)
        measured = np.array([sat.pseudorange for sat in satellites])

        for _ in range(self.config.max_iterations):
            H = self.compute_geometry_matrix(satellites, pos)
            predicted = np.array([np.linalg.norm(s.position - pos) for s in satellites]) + clock_bias
            dz = measured - predicted
            try:
                dx = np.linalg.solve(H.T @ W @ H, H.T @ W @ dz)
            except np.linalg.LinAlgError:
                break
            pos += dx[:3]
            clock_bias += dx[3]
            if np.linalg.norm(dx[:3]) < self.config.convergence_threshold:
                break

        predicted = np.array([np.linalg.norm(s.position - pos) for s in satellites]) + clock_bias
        return pos, clock_bias, measured - predicted

    def compute_test_statistic(self, residuals: np.ndarray, W: np.ndarray) -> float:
        """
        WSSE = r' W r. With correct weights and no fault it is chi-square with
        n - 4 degrees of freedom.
        """
        return float(residuals @ W @ residuals)

    def compute_threshold(self, dof: int) -> float:
        """Detection threshold on WSSE for the configured Pfa."""
        if dof <= 0:
            return float('inf')
        return float(stats.chi2.ppf(1 - self.config.pfa, dof))

    def compute_dop(self, H: np.ndarray, receiver_pos: np.ndarray) -> dict:
        """
        DOP values. HDOP and VDOP are taken in the local East-North-Up frame at
        receiver_pos.
        """
        try:
            Q = np.linalg.inv(H.T @ H)
        except np.linalg.LinAlgError:
            return {'gdop': 99.9, 'pdop': 99.9, 'hdop': 99.9, 'vdop': 99.9, 'tdop': 99.9}
        R = enu_rotation(receiver_pos)
        Q_enu = R @ Q[:3, :3] @ R.T
        return {
            'gdop': float(np.sqrt(np.trace(Q))),
            'pdop': float(np.sqrt(np.trace(Q[:3, :3]))),
            'hdop': float(np.sqrt(Q_enu[0, 0] + Q_enu[1, 1])),
            'vdop': float(np.sqrt(Q_enu[2, 2])),
            'tdop': float(np.sqrt(Q[3, 3])),
        }

    def compute_slopes(self, H: np.ndarray, W: np.ndarray,
                       receiver_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Horizontal and vertical slopes for each satellite.

        A bias b on satellite i moves the position by A[:3, i] b and the test
        statistic sqrt(WSSE) by |b| sqrt(w_i S_ii), so

            slope_H,i = |(A_E,i, A_N,i)| / sqrt(w_i S_ii)
            slope_V,i = |A_U,i| / sqrt(w_i S_ii)

        i.e. metres of position error per unit of sqrt(WSSE).
        """
        n = H.shape[0]
        try:
            A = np.linalg.solve(H.T @ W @ H, H.T @ W)
        except np.linalg.LinAlgError:
            return np.full(n, np.inf), np.full(n, np.inf)
        S = np.eye(n) - H @ A
        A_enu = enu_rotation(receiver_pos) @ A[:3, :]
        denom = np.sqrt(np.clip(np.diag(W) * np.diag(S), 1e-300, None))
        hslope = np.hypot(A_enu[0], A_enu[1]) / denom
        vslope = np.abs(A_enu[2]) / denom
        # a satellite with S_ii ~ 0 cannot be detected at all
        undetectable = np.diag(S) < 1e-12
        hslope[undetectable] = np.inf
        vslope[undetectable] = np.inf
        return hslope, vslope

    def compute_protection_levels(self, H: np.ndarray, W: np.ndarray, dof: int,
                                  receiver_pos: np.ndarray) -> Tuple[float, float, float, float, float]:
        """
        HPL = max_i slope_H,i * p_bias,  VPL = max_i slope_V,i * p_bias

        p_bias = sqrt(lambda), where lambda is the noncentrality of the WSSE
        that leaves exactly Pmd probability below the detection threshold.
        A single-satellite bias that big or bigger is detected with probability
        at least 1 - Pmd, and anything smaller moves the position by at most
        slope * p_bias along the worst satellite. Noise in the position
        solution itself is not added (classic Brown-style RAIM).

        Returns:
            (HPL, VPL, slope_max_h, slope_max_v, p_bias)
        """
        if dof <= 0:
            return float('inf'), float('inf'), 0.0, 0.0, float('inf')
        p_bias = _p_bias(dof, self.config.pfa, self.config.pmd)
        hslope, vslope = self.compute_slopes(H, W, receiver_pos)
        sh, sv = float(np.max(hslope)), float(np.max(vslope))
        return sh * p_bias, sv * p_bias, sh, sv, p_bias

    def detect_fault(self, satellites: List[Satellite],
                     initial_pos: Optional[np.ndarray] = None) -> RAIMResult:
        """
        RAIM fault detection on one epoch.

        Args:
            satellites: List of satellite measurements
            initial_pos: Initial position estimate

        Returns:
            RAIMResult with detection status and diagnostics
        """
        n = len(satellites)
        available, _ = self.check_availability(satellites)
        if not available:
            return RAIMResult(
                status=RAIMStatus.UNAVAILABLE, position=np.zeros(3), clock_bias=0.0,
                test_statistic=0.0, threshold=0.0, hpl=float('inf'), vpl=float('inf'),
                excluded_prns=[], residuals=np.array([]),
                dop={'gdop': 99.9, 'pdop': 99.9, 'hdop': 99.9, 'vdop': 99.9, 'tdop': 99.9},
                num_satellites=n, pfa=self.config.pfa, pmd=self.config.pmd, slope_max=0.0,
            )

        pos, clock_bias, residuals = self.solve_position(satellites, initial_pos)
        H = self.compute_geometry_matrix(satellites, pos)
        W = self.compute_weight_matrix(satellites)
        dof = n - 4

        test_stat = self.compute_test_statistic(residuals, W)
        threshold = self.compute_threshold(dof)
        dop = self.compute_dop(H, pos)
        hpl, vpl, sh, sv, p_bias = self.compute_protection_levels(H, W, dof, pos)

        if test_stat > threshold:
            status = RAIMStatus.FAULT_DETECTED
        elif dop['gdop'] > 10.0:
            status = RAIMStatus.GEOMETRY_POOR
        else:
            status = RAIMStatus.AVAILABLE

        return RAIMResult(
            status=status, position=pos, clock_bias=clock_bias,
            test_statistic=test_stat, threshold=threshold, hpl=hpl, vpl=vpl,
            excluded_prns=[], residuals=residuals, dop=dop, num_satellites=n,
            pfa=self.config.pfa, pmd=self.config.pmd,
            slope_max=sh, vslope_max=sv, p_bias=p_bias,
        )

    def exclude_fault(self, satellites: List[Satellite],
                      initial_pos: Optional[np.ndarray] = None) -> RAIMResult:
        """
        Fault detection and exclusion (FDE).

        Removes the satellite with the largest normalized residual
        |r_i| / sqrt(S_ii / w_i) and re-tests, until the test passes or too few
        satellites remain.

        Args:
            satellites: List of satellite measurements
            initial_pos: Initial position estimate

        Returns:
            RAIMResult with exclusion status
        """
        current_sats = list(satellites)
        excluded_prns = []

        for _ in range(self.config.max_iterations):
            result = self.detect_fault(current_sats, initial_pos)

            if result.status != RAIMStatus.FAULT_DETECTED:
                result.excluded_prns = excluded_prns
                if excluded_prns:
                    result.status = RAIMStatus.FAULT_EXCLUDED
                return result

            if len(current_sats) <= self.config.min_satellites:
                result.excluded_prns = excluded_prns
                return result

            # Normalized residuals: each residual divided by its own std. dev.
            H = self.compute_geometry_matrix(current_sats, result.position)
            W = self.compute_weight_matrix(current_sats)
            A = np.linalg.solve(H.T @ W @ H, H.T @ W)
            S = np.eye(len(current_sats)) - H @ A
            r_std = np.sqrt(np.clip(np.diag(S) / np.diag(W), 1e-300, None))
            worst = int(np.argmax(np.abs(result.residuals) / r_std))

            excluded_prns.append(current_sats[worst].prn)
            current_sats = [s for j, s in enumerate(current_sats) if j != worst]
            initial_pos = result.position

        result = self.detect_fault(current_sats, initial_pos)
        result.excluded_prns = excluded_prns
        return result

    def compute_integrity_risk(self, result: RAIMResult) -> dict:
        """
        Compare protection levels with the alert limits.

        Without the true position error only availability can be decided here;
        use stanford_region(pe, pl, al) when the truth is known.
        """
        hal, val = self.config.hal, self.config.val
        hpl_ratio = result.hpl / hal if hal > 0 else float('inf')
        vpl_ratio = (result.vpl / val) if val else 0.0
        vertical_ok = True if val is None else result.vpl <= val
        if result.status == RAIMStatus.UNAVAILABLE:
            availability = "unavailable"
        elif result.status == RAIMStatus.FAULT_DETECTED:
            availability = "alert"
        elif result.hpl <= hal and vertical_ok:
            availability = "available"
        else:
            availability = "unavailable"  # PL exceeds AL
        return {
            'hpl_ratio': hpl_ratio,
            'vpl_ratio': vpl_ratio,
            'availability': availability,
            'integrity_available': availability == "available",
            'pfa': result.pfa,
            'pmd': result.pmd,
        }


def _rng(seed, rng):
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def simulate_pseudoranges(true_pos: np.ndarray, satellites: List[Satellite],
                          true_clock_bias: float = 0.0,
                          noise_std: Union[float, Sequence[float]] = 1.0,
                          fault_prn: Optional[int] = None,
                          fault_magnitude: float = 0.0,
                          seed: Optional[int] = None,
                          rng: Optional[np.random.Generator] = None) -> List[Satellite]:
    """
    Simulate pseudoranges with Gaussian noise and an optional step fault.

    Args:
        true_pos: True receiver position [m]
        satellites: Satellites with positions
        true_clock_bias: True receiver clock bias [m]
        noise_std: 1-sigma noise [m], one value or one per satellite. It is also
            stored in each Satellite.sigma, so RAIM weights match the noise.
        fault_prn: PRN of satellite to inject fault (None = no fault)
        fault_magnitude: Magnitude of fault [m]
        seed / rng: Random seed or generator

    Returns:
        List of satellites with simulated pseudoranges
    """
    g = _rng(seed, rng)
    sig = np.broadcast_to(np.asarray(noise_std, dtype=float), (len(satellites),))
    out = []
    for sat, s in zip(satellites, sig):
        pr = np.linalg.norm(sat.position - true_pos) + true_clock_bias + g.normal(0.0, s)
        if fault_prn is not None and sat.prn == fault_prn:
            pr += fault_magnitude
        out.append(Satellite(prn=sat.prn, x=sat.x, y=sat.y, z=sat.z, pseudorange=pr,
                             cn0=sat.cn0, elevation=sat.elevation, sigma=float(s)))
    return out


def create_satellite_geometry(receiver_pos: np.ndarray,
                              num_satellites: int = 8,
                              seed: Optional[int] = None,
                              min_elevation: float = 15.0,
                              max_elevation: float = 85.0,
                              rng: Optional[np.random.Generator] = None) -> List[Satellite]:
    """
    Random sky geometry: azimuth uniform, elevation uniform in
    [min_elevation, max_elevation]. Each satellite is placed on a sphere of GPS
    orbit radius along that line of sight. This is a geometry generator, not an
    orbit model.

    Args:
        receiver_pos: Receiver ECEF position [m]
        num_satellites: Number of satellites to generate
        seed / rng: Random seed or generator

    Returns:
        List of satellites with positions (pseudoranges are zero)
    """
    g = _rng(seed, rng)
    R = enu_rotation(receiver_pos)  # rows E, N, U
    satellites = []
    for prn in range(1, num_satellites + 1):
        az = g.uniform(0, 2 * np.pi)
        el = g.uniform(min_elevation, max_elevation)
        el_rad = np.radians(el)
        d_enu = np.array([np.sin(az) * np.cos(el_rad), np.cos(az) * np.cos(el_rad), np.sin(el_rad)])
        d = R.T @ d_enu
        # distance along d to the GPS orbit sphere
        b = receiver_pos @ d
        rho = -b + np.sqrt(b**2 - receiver_pos @ receiver_pos + GPS_ORBIT_RADIUS**2)
        p = receiver_pos + rho * d
        satellites.append(Satellite(prn=prn, x=p[0], y=p[1], z=p[2], pseudorange=0.0,
                                    cn0=40 + g.uniform(0, 10), elevation=el))
    return satellites
