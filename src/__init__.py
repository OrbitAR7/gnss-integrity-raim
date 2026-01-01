"""
GNSS Integrity RAIM Package

Receiver Autonomous Integrity Monitoring for GNSS applications.
"""

from .raim import (
    RAIM,
    RAIMConfig,
    RAIMResult,
    RAIMStatus,
    Satellite,
    simulate_pseudoranges,
    create_satellite_geometry
)

__version__ = "1.0.0"
__author__ = "Hamoud Alshammari"

__all__ = [
    'RAIM',
    'RAIMConfig', 
    'RAIMResult',
    'RAIMStatus',
    'Satellite',
    'simulate_pseudoranges',
    'create_satellite_geometry'
]
