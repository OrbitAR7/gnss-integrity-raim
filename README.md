# gnss-integrity-raim

A compact Python implementation of snapshot **Receiver Autonomous Integrity Monitoring (RAIM)**: weighted least-squares positioning, chi-square fault detection, iterative fault exclusion, and slope-based protection levels. I wrote it to understand how integrity monitoring decides when a navigation solution should and should not be trusted.

## What it does

- Weighted least-squares position and clock solution from pseudoranges
- **Fault detection:** chi-square test on the weighted sum of squared residuals, with the threshold set by the false-alarm probability
- **Fault exclusion:** removes the satellite with the largest residual and re-tests until the solution passes or too few satellites remain
- **Protection levels:** slope-based HPL/VPL from the detection threshold and missed-detection probability
- Simulation helpers for satellite geometry and fault injection, plus three example scripts

## Stanford diagram

![Stanford diagram](docs/stanford_diagram.png)

The Stanford diagram plots each epoch's position error (PE) against its protection level (PL):

| Region | Condition | Meaning |
|---|---|---|
| Normal operation | PE < PL < AL | Error bounded, service available |
| Misleading information (MI) | PL < PE < AL | Bound violated, but error still below the alert limit |
| Hazardously misleading information (HMI) | PL < AL < PE | Error exceeds the alert limit with no alert, the case integrity monitoring exists to prevent |
| System unavailable | PL > AL | Protection level too large: the service is declared unavailable |

## Protection level vs. number of satellites

![Protection levels](docs/protection_levels.png)

For one simulated geometry (fixed random seed), HPL drops below the 40 m horizontal alert limit once 7 satellites are in view. The exact count depends on geometry, so this is an illustration, not a general rule.

## Quick start

```bash
git clone https://github.com/OrbitAR7/gnss-integrity-raim.git
cd gnss-integrity-raim
pip install -r requirements.txt
```

```python
import numpy as np
from src.raim import RAIM, RAIMConfig, create_satellite_geometry, simulate_pseudoranges

config = RAIMConfig(pfa=1e-5, pmd=1e-3, hal=40.0, val=50.0)
raim = RAIM(config)

receiver_pos = np.array([-742000.0, -5462000.0, 3198000.0])  # ECEF, m
geometry = create_satellite_geometry(receiver_pos, num_satellites=8)
sats = simulate_pseudoranges(receiver_pos, geometry, noise_std=2.0,
                             fault_prn=3, fault_magnitude=50.0)  # inject a 50 m fault

result = raim.exclude_fault(sats)
print(result.status.value, f"HPL={result.hpl:.1f} m", "excluded:", result.excluded_prns)
```

## Examples

| Script | Shows |
|---|---|
| `examples/basic_raim.py` | Sky plot, residuals with and without a fault, detection sensitivity |
| `examples/fault_exclusion.py` | Step-by-step fault detection and exclusion |
| `examples/protection_analysis.py` | Protection levels vs. satellite count, Stanford diagram, geometry impact |

Run them from the `examples/` folder; each saves PNG files there. `python generate_readme_plots.py` regenerates the two figures above in `docs/`.

## Satellite requirements (snapshot RAIM, 4 unknowns)

| Satellites | Detection | Exclusion |
|---|---|---|
| 4 | No | No |
| 5 | Yes | No |
| 6+ | Yes | Yes |

## Limitations

This is a learning implementation, not certified avionics software:

- Protection levels use a single maximum slope over the full 3-D position, and VPL is approximated as 1.5 × HPL. A standards-grade implementation computes separate horizontal and vertical slopes.
- Exclusion removes the largest residual rather than testing every subset, which can pick the wrong satellite in poor geometry.
- Measurements are simulated with Gaussian noise and single-satellite step faults. There are no real receiver data, multipath models or multi-constellation weighting.

Natural extensions include separate horizontal/vertical slopes, subset-based exclusion, ARAIM-style multiple-hypothesis protection levels, and real RINEX observations.

## Background

The algorithms follow the classical residual-based (parity-space) RAIM approach used for GPS non-precision approach. See Brown, *A Baseline GPS RAIM Scheme* (NAVIGATION, 1992), and RTCA DO-229 for the operational context.

## License

MIT. See `LICENSE`.
