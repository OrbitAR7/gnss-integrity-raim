#!/usr/bin/env python3
"""
RAIM Fault Exclusion Example

Demonstrates the complete fault detection and exclusion (FDE) process.
When a fault is detected, RAIM sequentially removes suspected satellites
until the solution becomes reliable again.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from src.raim import (
    RAIM, RAIMConfig, RAIMStatus,
    simulate_pseudoranges, create_satellite_geometry
)


def plot_fde_comparison(result_before, result_after, receiver_pos):
    """Compare results before and after fault exclusion"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals before exclusion
    prns_before = np.arange(1, len(result_before.residuals) + 1)
    colors_before = ['red' if abs(r) > 20 else 'orange' if abs(r) > 10 else 'blue' 
                     for r in result_before.residuals]
    
    ax1.bar(prns_before, result_before.residuals, color=colors_before, 
            alpha=0.7, edgecolor='black')
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Satellite PRN', fontsize=11)
    ax1.set_ylabel('Residual (m)', fontsize=11)
    ax1.set_title(f'Before Exclusion\n(Status: {result_before.status.value})', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    stats_before = f"Test Stat: {result_before.test_statistic:.1f}\n"
    stats_before += f"Threshold: {result_before.threshold:.1f}\n"
    stats_before += f"Position Error: {np.linalg.norm(result_before.position - receiver_pos):.1f}m"
    ax1.text(0.02, 0.98, stats_before, transform=ax1.transAxes,
            verticalalignment='top', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Residuals after exclusion
    prns_after = [prn for prn in prns_before if prn not in result_after.excluded_prns]
    residuals_after = [result_after.residuals[i] for i, prn in enumerate(prns_before) 
                      if prn not in result_after.excluded_prns]
    colors_after = ['blue' for _ in residuals_after]
    
    ax2.bar(prns_after, residuals_after, color=colors_after, 
            alpha=0.7, edgecolor='black')
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Satellite PRN', fontsize=11)
    ax2.set_ylabel('Residual (m)', fontsize=11)
    ax2.set_title(f'After Exclusion\n(Status: {result_after.status.value})', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    stats_after = f"Test Stat: {result_after.test_statistic:.1f}\n"
    stats_after += f"Threshold: {result_after.threshold:.1f}\n"
    stats_after += f"Position Error: {np.linalg.norm(result_after.position - receiver_pos):.1f}m\n"
    stats_after += f"Excluded: {result_after.excluded_prns}"
    ax2.text(0.02, 0.98, stats_after, transform=ax2.transAxes,
            verticalalignment='top', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('fault_exclusion_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: fault_exclusion_comparison.png")
    plt.close()


def plot_exclusion_process(results_sequence, receiver_pos):
    """Show the iterative exclusion process"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, result in enumerate(results_sequence[:6]):
        ax = axes[idx]
        
        prns = np.arange(1, len(result.residuals) + 1)
        colors = ['red' if abs(r) > 15 else 'orange' if abs(r) > 8 else 'blue' 
                  for r in result.residuals]
        
        ax.bar(prns, result.residuals, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('PRN', fontsize=9)
        ax.set_ylabel('Residual (m)', fontsize=9)
        
        title = f"Step {idx+1}: {result.num_satellites} sats"
        if result.excluded_prns:
            title += f"\nExcluded: {result.excluded_prns}"
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        pos_err = np.linalg.norm(result.position - receiver_pos)
        status_text = f"PE: {pos_err:.1f}m\nTS: {result.test_statistic:.1f}"
        ax.text(0.98, 0.98, status_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               fontsize=8, family='monospace',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.suptitle('Iterative Fault Exclusion Process', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('exclusion_process.png', dpi=150, bbox_inches='tight')
    print("  Saved: exclusion_process.png")
    plt.close()


def main():
    print("\n" + "="*60)
    print("  RAIM Fault Detection & Exclusion")
    print("="*60)
    
    config = RAIMConfig(pfa=1e-5, pmd=1e-3, hal=40.0, val=50.0)
    raim = RAIM(config)
    
    receiver_pos = np.array([-742000.0, -5462000.0, 3198000.0])
    
    # Create good geometry with enough satellites for FDE
    sat_geometry = create_satellite_geometry(receiver_pos, num_satellites=9, seed=42)
    
    print(f"\n  Constellation: {len(sat_geometry)} satellites")
    print(f"  FDE requires: >= 6 satellites")
    print(f"  HAL: {config.hal} m")
    
    # Scenario 1: Single fault
    print("\n" + "-"*60)
    print("  Scenario: Single Fault (60m on PRN 4)")
    print("-"*60)
    
    satellites = simulate_pseudoranges(
        receiver_pos, sat_geometry,
        noise_std=2.0, fault_prn=4, fault_magnitude=60.0, seed=456
    )
    
    # Detection only
    result_detect = raim.detect_fault(satellites)
    print(f"\n  Detection-only result:")
    print(f"    Status: {result_detect.status.value}")
    print(f"    Test statistic: {result_detect.test_statistic:.2f} (threshold: {result_detect.threshold:.2f})")
    print(f"    Position error: {np.linalg.norm(result_detect.position - receiver_pos):.1f} m")
    
    # With exclusion
    result_fde = raim.exclude_fault(satellites)
    print(f"\n  Fault exclusion result:")
    print(f"    Status: {result_fde.status.value}")
    print(f"    Excluded satellites: {result_fde.excluded_prns}")
    print(f"    Test statistic: {result_fde.test_statistic:.2f} (threshold: {result_fde.threshold:.2f})")
    print(f"    Position error: {np.linalg.norm(result_fde.position - receiver_pos):.1f} m")
    print(f"    Improvement: {np.linalg.norm(result_detect.position - receiver_pos) - np.linalg.norm(result_fde.position - receiver_pos):.1f} m")
    
    print("\n  Generating comparison plots...")
    plot_fde_comparison(result_detect, result_fde, receiver_pos)
    
    # Scenario 2: Multiple faults
    print("\n" + "-"*60)
    print("  Scenario: Multiple Faults (PRN 2: 50m, PRN 7: 70m)")
    print("-"*60)
    
    satellites = simulate_pseudoranges(
        receiver_pos, sat_geometry,
        noise_std=2.0, fault_prn=2, fault_magnitude=50.0, seed=111
    )
    
    # Add second fault
    for sat in satellites:
        if sat.prn == 7:
            sat.pseudorange += 70.0
    
    result = raim.exclude_fault(satellites)
    print(f"\n  Result:")
    print(f"    Status: {result.status.value}")
    print(f"    Excluded: {result.excluded_prns}")
    print(f"    Position error: {np.linalg.norm(result.position - receiver_pos):.1f} m")
    
    if len(result.excluded_prns) >= 2:
        print(f"    ✓ Both faults successfully excluded")
    
    # Scenario 3: Insufficient satellites
    print("\n" + "-"*60)
    print("  Scenario: Only 5 Satellites Available")
    print("-"*60)
    
    limited_geometry = sat_geometry[:5]
    satellites = simulate_pseudoranges(
        receiver_pos, limited_geometry,
        noise_std=2.0, fault_prn=2, fault_magnitude=45.0, seed=222
    )
    
    result = raim.exclude_fault(satellites)
    print(f"\n  Result:")
    print(f"    Status: {result.status.value}")
    print(f"    Can detect fault: {result.test_statistic > result.threshold}")
    print(f"    Can exclude fault: {len(result.excluded_prns) > 0}")
    print(f"    Note: Need >= 6 satellites for exclusion")
    
    # Summary
    print("\n" + "="*60)
    print("  FDE Capability Summary")
    print("="*60)
    print("""
  # Sats    Detection    Exclusion    Notes
  -------    ---------    ---------    -----
     4         No           No         Min for position
     5         Yes          No         Can detect, can't exclude
     6+        Yes          Yes        Full FDE capability
     
  Key points:
  - FDE finds and removes bad satellites
  - Needs redundant satellites (6+)
  - Position accuracy improves after exclusion
  - Protection levels may increase slightly
    """)
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
