# GNSS RAIM Implementation

A practical Python implementation of Receiver Autonomous Integrity Monitoring (RAIM) for evaluating GNSS positioning integrity in safety-critical applications.

## What is RAIM?

RAIM is a technique used in GPS/GNSS receivers to assess the integrity of positioning solutions without relying on external systems. It's essential in aviation and other applications where position accuracy must be guaranteed within specific bounds.

This implementation provides:
- **Fault Detection** using chi-square statistical testing
- **Fault Exclusion** to identify and remove bad satellites
- **Protection Level** calculation (HPL/VPL) for integrity monitoring
- **Visualization tools** for understanding RAIM behavior

## Key Visualizations

### Stanford Diagram: The Four Integrity Regions

The Stanford diagram is the signature visualization in GNSS integrity monitoring, showing how protection levels and position errors define operational safety.
<img width="2590" height="2662" alt="stanford_diagram" src="https://github.com/user-attachments/assets/a945f2a9-bac9-4007-b843-62c895651958" />


**Four critical regions:**
- **Normal Operation** (green): Position error and protection level both within limits
- **Misleading Information** (yellow): System claims unavailable, but position is actually good
- **Hazardously Misleading Information** (red): Undetected position error exceeds limits — the worst case scenario
- **System Unavailable** (orange): Fault detected, alert raised

### Protection Levels vs Satellite Count

More satellites dramatically improve integrity. This shows the engineering tradeoff between constellation size and alert limits.

<img width="2961" height="1760" alt="protection_levels" src="https://github.com/user-attachments/assets/ac7308e3-3007-4335-8fe3-bbfbd64e11bd" />


**Key insight:** At least 8 satellites are needed to meet Non-Precision Approach (NPA) integrity requirements (HAL = 40m).

## Quick Example

```python
import numpy as np
from src.raim import RAIM, RAIMConfig, create_satellite_geometry, simulate_pseudoranges

# Set up RAIM with aviation parameters
config = RAIMConfig(pfa=1e-5, pmd=1e-3, hal=40.0, val=50.0)
raim = RAIM(config)

# Simulate a receiver position and satellite constellation
receiver_pos = np.array([-742000.0, -5462000.0, 3198000.0])  # ECEF
sat_geometry = create_satellite_geometry(receiver_pos, num_satellites=8)

# Generate measurements (with optional fault injection)
satellites = simulate_pseudoranges(
    receiver_pos, sat_geometry,
    noise_std=2.0,
    fault_prn=3,           # Inject fault on satellite PRN 3
    fault_magnitude=50.0
)

# Run RAIM algorithm
result = raim.exclude_fault(satellites)

print(f"Status: {result.status.value}")
print(f"HPL: {result.hpl:.1f} m (limit: {config.hal} m)")
print(f"Satellites excluded: {result.excluded_prns}")
```

## Installation

```bash
git clone https://github.com/OrbitAR7/gnss-integrity-raim.git
cd gnss-integrity-raim
pip install -r requirements.txt
```

## Examples

Three example scripts demonstrate different aspects of RAIM. Each generates publication-quality visualizations.

### 1. Basic Fault Detection
```bash
python examples/basic_raim.py
```
Shows how RAIM detects satellite faults through residual analysis and chi-square testing. Generates:
- Satellite geometry (sky plot and 3D positions)
- Residual bar charts for normal/faulty scenarios
- Detection sensitivity curve

### 2. Fault Exclusion
```bash
python examples/fault_exclusion.py
```
Demonstrates the complete fault detection and exclusion (FDE) process with multiple scenarios. Generates:
- Before/after comparison plots
- Multi-step exclusion process visualization

### 3. Protection Level Analysis
```bash
python examples/protection_analysis.py
```
Analyzes how protection levels behave with different geometries. Generates:
- Protection levels vs satellite count
- Stanford diagram
- Geometry impact analysis

**Note:** All examples save PNG files in the current directory.

## Understanding the Output

### RAIMResult Object
Every RAIM computation returns a `RAIMResult` with:
- `status`: Current RAIM state (available, fault_detected, etc.)
- `position`: Computed ECEF position
- `hpl`, `vpl`: Horizontal and Vertical Protection Levels
- `test_statistic`: Chi-square test value
- `threshold`: Detection threshold
- `residuals`: Per-satellite measurement residuals
- `dop`: Dilution of Precision values (GDOP, HDOP, VDOP, etc.)

### Protection Levels
Protection levels define error bounds with specific confidence:
- **HPL** (Horizontal Protection Level): horizontal position error bound
- **VPL** (Vertical Protection Level): vertical position error bound

If HPL < HAL (Horizontal Alert Limit), the solution meets integrity requirements.

## How It Works

### Fault Detection
1. Compute weighted least squares position solution
2. Calculate residuals (observed - predicted measurements)
3. Form chi-square test statistic from weighted residual sum
4. Compare against threshold based on false alarm probability
5. Fault detected if test statistic exceeds threshold

### Fault Exclusion
When a fault is detected:
1. Identify satellite with largest residual
2. Remove it from solution
3. Recompute position and test statistic
4. Repeat until no fault detected or too few satellites

### Protection Levels
Computed using:
```
HPL = slope_max × sqrt(threshold + bias_term)
```
where `slope_max` represents the worst-case positioning error sensitivity to measurement errors.

## Satellite Requirements

| Satellites | Fault Detection | Fault Exclusion |
|------------|----------------|-----------------|
| 4          | No             | No              |
| 5          | Yes            | No              |
| 6+         | Yes            | Yes             |

Fault exclusion requires at least 6 satellites to maintain positioning after removing one faulty satellite.

## Configuration Parameters

Key `RAIMConfig` parameters:

- `pfa`: Probability of false alarm (typically 1e-5)
- `pmd`: Probability of missed detection (typically 1e-3)
- `hal`: Horizontal Alert Limit in meters (40m for aviation NPA)
- `val`: Vertical Alert Limit in meters (50m for aviation NPA)
- `min_satellites`: Minimum satellites for RAIM (5 or more)
- `use_weighted`: Use weighted least squares (recommended: True)

## References

This implementation follows standard RAIM algorithms used in aviation:
- RTCA DO-229D (WAAS MOPS)
- FAA TSO-C129a
- Standard chi-square residual-based fault detection

## Generating Documentation Plots

To regenerate the README visualizations:

```bash
python generate_readme_plots.py
```

This creates high-quality figures in the `docs/` directory.

## License

MIT License - see LICENSE file for details.


