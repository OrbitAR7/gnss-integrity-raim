"""
GNSS RAIM (Receiver Autonomous Integrity Monitoring) Module

Implements fault detection and exclusion algorithms for GNSS integrity monitoring.
Supports both snapshot RAIM and weighted RAIM approaches.

Author: Hamoud Alshammari
Date: 2025
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy import stats
from enum import Enum


class RAIMStatus(Enum):
    """RAIM algorithm status codes"""
    AVAILABLE = "available"           # RAIM available, no fault detected
    FAULT_DETECTED = "fault_detected" # Fault detected but not excluded
    FAULT_EXCLUDED = "fault_excluded" # Fault detected and excluded
    UNAVAILABLE = "unavailable"       # Insufficient satellites for RAIM
    GEOMETRY_POOR = "geometry_poor"   # Poor geometry, RAIM unreliable


@dataclass
class Satellite:
    """Satellite measurement data"""
    prn: int
    x: float  # ECEF X position [m]
    y: float  # ECEF Y position [m]
    z: float  # ECEF Z position [m]
    pseudorange: float  # Measured pseudorange [m]
    cn0: float = 45.0  # Carrier-to-noise ratio [dB-Hz]
    elevation: float = 45.0  # Elevation angle [deg]
    
    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class RAIMResult:
    """RAIM algorithm results"""
    status: RAIMStatus
    position: np.ndarray          # Estimated position [m] (ECEF)
    clock_bias: float             # Receiver clock bias [m]
    test_statistic: float         # Chi-square test statistic
    threshold: float              # Detection threshold
    hpl: float                    # Horizontal Protection Level [m]
    vpl: float                    # Vertical Protection Level [m]
    excluded_prns: List[int]      # PRNs excluded due to faults
    residuals: np.ndarray         # Post-fit residuals [m]
    dop: dict                     # Dilution of Precision values
    num_satellites: int           # Number of satellites used
    pfa: float                    # Probability of false alarm
    pmd: float                    # Probability of missed detection
    slope_max: float              # Maximum slope (for protection level)


@dataclass 
class RAIMConfig:
    """RAIM algorithm configuration"""
    pfa: float = 1e-5             # Probability of false alarm
    pmd: float = 1e-3             # Probability of missed detection
    hal: float = 40.0             # Horizontal Alert Limit [m]
    val: float = 50.0             # Vertical Alert Limit [m]
    min_satellites: int = 5       # Minimum satellites for RAIM
    max_iterations: int = 10      # Max iterations for exclusion
    convergence_threshold: float = 1e-4  # Position convergence [m]
    use_weighted: bool = True     # Use weighted least squares
    elevation_mask: float = 5.0   # Elevation mask [deg]


class RAIM:
    """
    Receiver Autonomous Integrity Monitoring
    
    Implements snapshot RAIM with:
    - Least squares residual (LSR) fault detection
    - Sequential fault exclusion
    - Protection level computation
    - Dilution of Precision (DOP) analysis
    """
    
    def __init__(self, config: Optional[RAIMConfig] = None):
        self.config = config or RAIMConfig()
        
    def check_availability(self, satellites: List[Satellite]) -> Tuple[bool, str]:
        """
        Check if RAIM is available given current satellite geometry
        
        Args:
            satellites: List of visible satellites
            
        Returns:
            (available, reason): Availability flag and reason string
        """
        n = len(satellites)
        
        # Need minimum satellites for fault detection
        if n < self.config.min_satellites:
            return False, f"Insufficient satellites: {n} < {self.config.min_satellites}"
        
        # Need n >= 6 for fault exclusion capability
        if n < 6:
            return True, "FD only (no exclusion capability)"
            
        return True, "Full RAIM available (FD + FDE)"
    
    def compute_geometry_matrix(self, satellites: List[Satellite], 
                                 receiver_pos: np.ndarray) -> np.ndarray:
        """
        Compute the geometry (design) matrix H
        
        Args:
            satellites: List of satellite data
            receiver_pos: Current receiver position estimate [m]
            
        Returns:
            H: Geometry matrix (n x 4)
        """
        n = len(satellites)
        H = np.zeros((n, 4))
        
        for i, sat in enumerate(satellites):
            # Line-of-sight vector
            los = sat.position - receiver_pos
            range_est = np.linalg.norm(los)
            
            # Unit vector (negative because measurement = sat - receiver)
            H[i, 0] = -los[0] / range_est
            H[i, 1] = -los[1] / range_est
            H[i, 2] = -los[2] / range_est
            H[i, 3] = 1.0  # Clock bias
            
        return H
    
    def compute_weight_matrix(self, satellites: List[Satellite]) -> np.ndarray:
        """
        Compute measurement weight matrix based on elevation and CN0
        
        Args:
            satellites: List of satellite data
            
        Returns:
            W: Weight matrix (diagonal)
        """
        n = len(satellites)
        W = np.zeros((n, n))
        
        for i, sat in enumerate(satellites):
            # Elevation-dependent weighting (higher elevation = lower noise)
            el_rad = np.radians(sat.elevation)
            sigma_el = 1.0 / np.sin(el_rad) if sat.elevation > 5 else 10.0
            
            # CN0-dependent weighting
            sigma_cn0 = 10 ** ((45 - sat.cn0) / 20)
            
            # Combined sigma
            sigma = np.sqrt(sigma_el**2 + sigma_cn0**2)
            
            # Weight = 1/variance
            W[i, i] = 1.0 / (sigma**2) if self.config.use_weighted else 1.0
            
        return W
    
    def solve_position(self, satellites: List[Satellite],
                       initial_pos: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Solve for receiver position using weighted least squares
        
        Args:
            satellites: List of satellite data
            initial_pos: Initial position estimate (default: center of Earth)
            
        Returns:
            (position, clock_bias, residuals)
        """
        if initial_pos is None:
            initial_pos = np.array([0.0, 0.0, 0.0])
            
        pos = initial_pos.copy()
        clock_bias = 0.0
        
        for iteration in range(self.config.max_iterations):
            # Build geometry matrix
            H = self.compute_geometry_matrix(satellites, pos)
            W = self.compute_weight_matrix(satellites)
            
            # Compute predicted pseudoranges
            predicted = np.array([
                np.linalg.norm(sat.position - pos) + clock_bias
                for sat in satellites
            ])
            
            # Measurement residuals
            measured = np.array([sat.pseudorange for sat in satellites])
            dz = measured - predicted
            
            # Weighted least squares solution
            # dx = (H^T W H)^-1 H^T W dz
            HtWH = H.T @ W @ H
            HtWdz = H.T @ W @ dz
            
            try:
                dx = np.linalg.solve(HtWH, HtWdz)
            except np.linalg.LinAlgError:
                # Singular matrix - poor geometry
                break
                
            # Update state
            pos += dx[:3]
            clock_bias += dx[3]
            
            # Check convergence
            if np.linalg.norm(dx[:3]) < self.config.convergence_threshold:
                break
        
        # Compute post-fit residuals
        predicted = np.array([
            np.linalg.norm(sat.position - pos) + clock_bias
            for sat in satellites
        ])
        residuals = measured - predicted
        
        return pos, clock_bias, residuals
    
    def compute_test_statistic(self, residuals: np.ndarray, 
                                H: np.ndarray, W: np.ndarray) -> float:
        """
        Compute the RAIM test statistic (sum of squared weighted residuals)
        
        Args:
            residuals: Post-fit residuals
            H: Geometry matrix
            W: Weight matrix
            
        Returns:
            SSE: Sum of squared errors (chi-square distributed)
        """
        n = len(residuals)
        
        # Projection matrix: P = I - H(H^T W H)^-1 H^T W
        HtWH_inv = np.linalg.inv(H.T @ W @ H)
        P = np.eye(n) - H @ HtWH_inv @ H.T @ W
        
        # Weighted sum of squared residuals
        # Under H0 (no fault), SSE ~ chi-square(n-4)
        SSE = residuals.T @ W @ P @ W @ residuals
        
        return float(SSE)
    
    def compute_threshold(self, dof: int) -> float:
        """
        Compute detection threshold from chi-square distribution
        
        Args:
            dof: Degrees of freedom (n - 4 for position/clock)
            
        Returns:
            threshold: Detection threshold for given Pfa
        """
        if dof <= 0:
            return float('inf')
        return stats.chi2.ppf(1 - self.config.pfa, dof)
    
    def compute_dop(self, H: np.ndarray, receiver_lla: Optional[np.ndarray] = None) -> dict:
        """
        Compute Dilution of Precision values
        
        Args:
            H: Geometry matrix
            receiver_lla: Receiver position in LLA [rad, rad, m] for HDOP/VDOP
            
        Returns:
            Dictionary with GDOP, PDOP, HDOP, VDOP, TDOP
        """
        try:
            Q = np.linalg.inv(H.T @ H)
        except np.linalg.LinAlgError:
            return {'gdop': 99.9, 'pdop': 99.9, 'hdop': 99.9, 'vdop': 99.9, 'tdop': 99.9}
        
        gdop = np.sqrt(np.trace(Q))
        pdop = np.sqrt(Q[0,0] + Q[1,1] + Q[2,2])
        tdop = np.sqrt(Q[3,3])
        
        # For HDOP/VDOP, we need to rotate to local ENU frame
        # Simplified: assume Q is approximately diagonal
        hdop = np.sqrt(Q[0,0] + Q[1,1])
        vdop = np.sqrt(Q[2,2])
        
        return {
            'gdop': float(gdop),
            'pdop': float(pdop),
            'hdop': float(hdop),
            'vdop': float(vdop),
            'tdop': float(tdop)
        }
    
    def compute_slopes(self, H: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Compute slope values for each satellite (sensitivity to single fault)
        
        The slope relates range error to position error for each satellite.
        Used for protection level computation.
        
        Args:
            H: Geometry matrix
            W: Weight matrix
            
        Returns:
            slopes: Slope value for each satellite
        """
        n = H.shape[0]
        
        try:
            HtWH_inv = np.linalg.inv(H.T @ W @ H)
        except np.linalg.LinAlgError:
            return np.full(n, 99.9)
        
        # Covariance of position solution
        # Cov(x) = (H^T W H)^-1 H^T W Cov(z) W H (H^T W H)^-1
        # For unit variance measurements: Cov(x) = (H^T W H)^-1
        
        # Projection matrix for residuals
        P = np.eye(n) - H @ HtWH_inv @ H.T @ W
        
        slopes = np.zeros(n)
        for i in range(n):
            # Slope for satellite i: how much does removing it affect the solution?
            # Approximation: slope_i ≈ sqrt(H_i (H^T W H)^-1 H_i^T) / sqrt(P_ii)
            
            h_i = H[i, :3]  # Position part of geometry row
            w_i = W[i, i]
            p_ii = P[i, i]
            
            if p_ii > 1e-10:
                # Position error contribution
                pos_var = h_i @ HtWH_inv[:3, :3] @ h_i.T
                slopes[i] = np.sqrt(pos_var * w_i) / np.sqrt(p_ii)
            else:
                slopes[i] = 99.9
                
        return slopes
    
    def compute_protection_levels(self, H: np.ndarray, W: np.ndarray,
                                   dof: int) -> Tuple[float, float, float]:
        """
        Compute Horizontal and Vertical Protection Levels
        
        HPL and VPL bound the position error with probability (1 - Pmd)
        given that a fault has been detected.
        
        Args:
            H: Geometry matrix
            W: Weight matrix
            dof: Degrees of freedom
            
        Returns:
            (HPL, VPL, slope_max)
        """
        if dof <= 0:
            return float('inf'), float('inf'), 0.0
            
        # Threshold for missed detection
        k_md = stats.norm.ppf(1 - self.config.pmd / 2)
        
        # Detection threshold
        T = np.sqrt(self.compute_threshold(dof))
        
        # Compute slopes
        slopes = self.compute_slopes(H, W)
        slope_max = float(np.max(slopes))
        
        # Protection level = slope_max * (T + k_md * sigma)
        # Simplified: assuming unit variance measurements
        sigma = 1.0
        
        # Horizontal and vertical components
        # More sophisticated: separate horizontal and vertical slopes
        hpl = slope_max * (T + k_md * sigma)
        vpl = slope_max * (T + k_md * sigma) * 1.5  # VDOP typically worse
        
        return float(hpl), float(vpl), slope_max
    
    def detect_fault(self, satellites: List[Satellite],
                     initial_pos: Optional[np.ndarray] = None) -> RAIMResult:
        """
        Perform RAIM fault detection
        
        Args:
            satellites: List of satellite measurements
            initial_pos: Initial position estimate
            
        Returns:
            RAIMResult with detection status and diagnostics
        """
        n = len(satellites)
        
        # Check availability
        available, reason = self.check_availability(satellites)
        if not available:
            return RAIMResult(
                status=RAIMStatus.UNAVAILABLE,
                position=np.zeros(3),
                clock_bias=0.0,
                test_statistic=0.0,
                threshold=0.0,
                hpl=float('inf'),
                vpl=float('inf'),
                excluded_prns=[],
                residuals=np.array([]),
                dop={'gdop': 99.9, 'pdop': 99.9, 'hdop': 99.9, 'vdop': 99.9, 'tdop': 99.9},
                num_satellites=n,
                pfa=self.config.pfa,
                pmd=self.config.pmd,
                slope_max=0.0
            )
        
        # Solve position
        pos, clock_bias, residuals = self.solve_position(satellites, initial_pos)
        
        # Build matrices at solution
        H = self.compute_geometry_matrix(satellites, pos)
        W = self.compute_weight_matrix(satellites)
        
        # Degrees of freedom
        dof = n - 4
        
        # Compute test statistic and threshold
        test_stat = self.compute_test_statistic(residuals, H, W)
        threshold = self.compute_threshold(dof)
        
        # Compute DOP
        dop = self.compute_dop(H)
        
        # Compute protection levels
        hpl, vpl, slope_max = self.compute_protection_levels(H, W, dof)
        
        # Determine status
        if test_stat > threshold:
            status = RAIMStatus.FAULT_DETECTED
        elif dop['gdop'] > 10.0:
            status = RAIMStatus.GEOMETRY_POOR
        else:
            status = RAIMStatus.AVAILABLE
            
        return RAIMResult(
            status=status,
            position=pos,
            clock_bias=clock_bias,
            test_statistic=test_stat,
            threshold=threshold,
            hpl=hpl,
            vpl=vpl,
            excluded_prns=[],
            residuals=residuals,
            dop=dop,
            num_satellites=n,
            pfa=self.config.pfa,
            pmd=self.config.pmd,
            slope_max=slope_max
        )
    
    def exclude_fault(self, satellites: List[Satellite],
                      initial_pos: Optional[np.ndarray] = None) -> RAIMResult:
        """
        Perform RAIM fault detection and exclusion (FDE)
        
        Sequentially excludes satellites until no fault is detected
        or too few satellites remain.
        
        Args:
            satellites: List of satellite measurements
            initial_pos: Initial position estimate
            
        Returns:
            RAIMResult with exclusion status
        """
        current_sats = list(satellites)
        excluded_prns = []
        
        for iteration in range(self.config.max_iterations):
            # Run detection
            result = self.detect_fault(current_sats, initial_pos)
            
            # If no fault or unavailable, return result
            if result.status != RAIMStatus.FAULT_DETECTED:
                result.excluded_prns = excluded_prns
                if excluded_prns:
                    result.status = RAIMStatus.FAULT_EXCLUDED
                return result
            
            # Check if we can exclude more satellites
            if len(current_sats) <= self.config.min_satellites:
                result.excluded_prns = excluded_prns
                return result
            
            # Find satellite with largest residual
            max_idx = np.argmax(np.abs(result.residuals))
            excluded_prn = current_sats[max_idx].prn
            excluded_prns.append(excluded_prn)
            
            # Remove satellite
            current_sats = [s for s in current_sats if s.prn != excluded_prn]
            
            # Update initial position for next iteration
            initial_pos = result.position
        
        # Max iterations reached
        result = self.detect_fault(current_sats, initial_pos)
        result.excluded_prns = excluded_prns
        return result
    
    def compute_integrity_risk(self, result: RAIMResult) -> dict:
        """
        Compute integrity risk metrics
        
        Args:
            result: RAIM result from detection/exclusion
            
        Returns:
            Dictionary with integrity risk metrics
        """
        # Probability of hazardously misleading information (HMI)
        # P(HMI) = P(position error > AL | no alert)
        
        # Check against alert limits
        hpl_ratio = result.hpl / self.config.hal if self.config.hal > 0 else float('inf')
        vpl_ratio = result.vpl / self.config.val if self.config.val > 0 else float('inf')
        
        # Stanford diagram region
        if result.status == RAIMStatus.UNAVAILABLE:
            region = "unavailable"
        elif hpl_ratio > 1.0 or vpl_ratio > 1.0:
            region = "misleading_information"  # Protection level exceeds alert limit
        elif result.status == RAIMStatus.FAULT_DETECTED:
            region = "alert"
        else:
            region = "normal_operation"
        
        return {
            'hpl_ratio': hpl_ratio,
            'vpl_ratio': vpl_ratio,
            'stanford_region': region,
            'integrity_available': result.hpl < self.config.hal and result.vpl < self.config.val,
            'pfa': result.pfa,
            'pmd': result.pmd
        }


def simulate_pseudoranges(true_pos: np.ndarray, satellites: List[Satellite],
                          true_clock_bias: float = 0.0,
                          noise_std: float = 1.0,
                          fault_prn: Optional[int] = None,
                          fault_magnitude: float = 0.0,
                          seed: Optional[int] = None) -> List[Satellite]:
    """
    Simulate pseudorange measurements with optional fault injection
    
    Args:
        true_pos: True receiver position [m]
        satellites: List of satellites with positions
        true_clock_bias: True receiver clock bias [m]
        noise_std: Pseudorange noise standard deviation [m]
        fault_prn: PRN of satellite to inject fault (None = no fault)
        fault_magnitude: Magnitude of fault [m]
        seed: Random seed for reproducibility
        
    Returns:
        List of satellites with simulated pseudoranges
    """
    if seed is not None:
        np.random.seed(seed)
    
    result = []
    for sat in satellites:
        # True geometric range
        true_range = np.linalg.norm(sat.position - true_pos)
        
        # Add clock bias and noise
        noise = np.random.normal(0, noise_std)
        pseudorange = true_range + true_clock_bias + noise
        
        # Inject fault if specified
        if fault_prn is not None and sat.prn == fault_prn:
            pseudorange += fault_magnitude
        
        result.append(Satellite(
            prn=sat.prn,
            x=sat.x, y=sat.y, z=sat.z,
            pseudorange=pseudorange,
            cn0=sat.cn0,
            elevation=sat.elevation
        ))
    
    return result


def create_satellite_geometry(receiver_pos: np.ndarray,
                               num_satellites: int = 8,
                               seed: Optional[int] = None) -> List[Satellite]:
    """
    Create a realistic satellite geometry around a receiver position
    
    Args:
        receiver_pos: Receiver ECEF position [m]
        num_satellites: Number of satellites to generate
        seed: Random seed
        
    Returns:
        List of satellites with positions (no pseudoranges yet)
    """
    if seed is not None:
        np.random.seed(seed)
    
    EARTH_RADIUS = 6378137.0
    GPS_ALTITUDE = 20200000.0
    
    satellites = []
    
    for prn in range(1, num_satellites + 1):
        # Random azimuth and elevation
        azimuth = np.random.uniform(0, 2 * np.pi)
        elevation = np.random.uniform(15, 85)  # degrees
        
        # Convert to unit vector in local ENU
        el_rad = np.radians(elevation)
        e = np.sin(azimuth) * np.cos(el_rad)
        n = np.cos(azimuth) * np.cos(el_rad)
        u = np.sin(el_rad)
        
        # Approximate conversion to ECEF (simplified)
        # For accurate conversion, would need receiver lat/lon
        sat_range = GPS_ALTITUDE + EARTH_RADIUS
        
        # Simple rotation based on receiver position
        rx_unit = receiver_pos / np.linalg.norm(receiver_pos)
        
        # Create orthonormal basis (simplified)
        if abs(rx_unit[2]) < 0.9:
            east = np.cross([0, 0, 1], rx_unit)
        else:
            east = np.cross([0, 1, 0], rx_unit)
        east = east / np.linalg.norm(east)
        north = np.cross(rx_unit, east)
        
        # Satellite direction in ECEF
        sat_dir = e * east + n * north + u * rx_unit
        sat_pos = receiver_pos + sat_range * sat_dir
        
        satellites.append(Satellite(
            prn=prn,
            x=sat_pos[0], y=sat_pos[1], z=sat_pos[2],
            pseudorange=0.0,
            cn0=40 + np.random.uniform(0, 10),
            elevation=elevation
        ))
    
    return satellites
