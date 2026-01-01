#!/usr/bin/env python3
"""
RAIM Protection Level Analysis

Shows how protection levels vary with satellite geometry and demonstrates
the Stanford diagram concept for integrity monitoring.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.raim import (
    RAIM, RAIMConfig,
    simulate_pseudoranges, create_satellite_geometry
)


def plot_protection_levels_vs_satellites():
    """Show how HPL/VPL change with number of satellites"""
    print("\n  Analyzing protection levels vs satellite count...")
    
    config = RAIMConfig(pfa=1e-5, pmd=1e-3, hal=40.0, val=50.0)
    raim = RAIM(config)
    receiver_pos = np.array([-742000.0, -5462000.0, 3198000.0])
    
    n_sats_range = range(5, 13)
    hpls = []
    vpls = []
    gdops = []
    
    for n_sats in n_sats_range:
        sat_geometry = create_satellite_geometry(receiver_pos, num_satellites=n_sats, seed=42)
        satellites = simulate_pseudoranges(receiver_pos, sat_geometry, noise_std=2.0, seed=100)
        result = raim.detect_fault(satellites)
        
        hpls.append(result.hpl)
        vpls.append(result.vpl)
        gdops.append(result.dop['gdop'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Protection levels
    ax1.plot(n_sats_range, hpls, 'b-o', linewidth=2, markersize=8, label='HPL')
    ax1.plot(n_sats_range, vpls, 'r-s', linewidth=2, markersize=8, label='VPL')
    ax1.axhline(config.hal, color='blue', linestyle='--', linewidth=2, alpha=0.5, label=f'HAL ({config.hal}m)')
    ax1.axhline(config.val, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'VAL ({config.val}m)')
    ax1.set_xlabel('Number of Satellites', fontsize=12)
    ax1.set_ylabel('Protection Level (m)', fontsize=12)
    ax1.set_title('Protection Levels vs Satellite Count', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # GDOP
    ax2.plot(n_sats_range, gdops, 'g-^', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Satellites', fontsize=12)
    ax2.set_ylabel('GDOP', fontsize=12)
    ax2.set_title('Geometric Dilution of Precision', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('protection_levels_vs_sats.png', dpi=150, bbox_inches='tight')
    print("  Saved: protection_levels_vs_sats.png")
    plt.close()


def plot_stanford_diagram():
    """Create Stanford diagram showing integrity regions"""
    print("\n  Creating Stanford diagram...")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    hal = 40.0  # meters
    
    # Define regions
    # Region 1: Normal operation (PE < HPL < HAL)
    ax.add_patch(Rectangle((0, 0), hal, hal, facecolor='green', alpha=0.3))
    ax.text(hal/2, hal/2, 'Normal\nOperation', ha='center', va='center', 
            fontsize=14, fontweight='bold')
    
    # Region 2: Misleading Information (HPL > HAL, PE < HAL)
    ax.add_patch(Rectangle((hal, 0), 60, hal, facecolor='yellow', alpha=0.3))
    ax.text(hal+30, hal/2, 'Misleading\nInformation\n(MI)', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # Region 3: Hazardously Misleading Information (PE > HAL, no alert)
    ax.add_patch(Rectangle((0, hal), hal, 60, facecolor='red', alpha=0.3))
    ax.text(hal/2, hal+30, 'Hazardously\nMisleading\nInformation\n(HMI)', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='darkred')
    
    # Region 4: Alert (fault detected, PE may be large)
    ax.add_patch(Rectangle((hal, hal), 60, 60, facecolor='orange', alpha=0.3))
    ax.text(hal+30, hal+30, 'Alert\n(Fault Detected)', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # Add threshold lines
    ax.axhline(hal, color='black', linestyle='-', linewidth=2, label='HAL')
    ax.axvline(hal, color='black', linestyle='-', linewidth=2, label='HPL = HAL')
    
    # Add example points
    examples = [
        (15, 12, 'Good\nSolution', 'blue'),
        (25, 35, 'Borderline', 'orange'),
        (75, 15, 'No Integrity', 'brown'),
        (20, 55, 'HMI!\n(Worst Case)', 'red'),
        (65, 75, 'Fault\nDetected', 'purple')
    ]
    
    for hpl, pe, label, color in examples:
        ax.plot(hpl, pe, 'o', markersize=12, color=color)
        ax.annotate(label, (hpl, pe), xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Horizontal Protection Level (HPL) [m]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Position Error (PE) [m]', fontsize=12, fontweight='bold')
    ax.set_title('Stanford Diagram - RAIM Integrity Regions', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='upper right')
    
    # Add annotation
    ax.text(2, 95, 'Goal: Minimize probability of HMI region', 
            fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    plt.savefig('stanford_diagram.png', dpi=150, bbox_inches='tight')
    print("  Saved: stanford_diagram.png")
    plt.close()


def plot_geometry_impact():
    """Show how geometry affects protection levels"""
    print("\n  Analyzing geometry impact...")
    
    config = RAIMConfig(pfa=1e-5, pmd=1e-3, hal=40.0, val=50.0)
    raim = RAIM(config)
    receiver_pos = np.array([-742000.0, -5462000.0, 3198000.0])
    
    seeds = range(10, 100, 5)
    hpls = []
    gdops = []
    hdops = []
    
    for seed in seeds:
        sat_geometry = create_satellite_geometry(receiver_pos, num_satellites=8, seed=seed)
        satellites = simulate_pseudoranges(receiver_pos, sat_geometry, noise_std=2.0, seed=100)
        result = raim.detect_fault(satellites)
        
        hpls.append(result.hpl)
        gdops.append(result.dop['gdop'])
        hdops.append(result.dop['hdop'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # HPL vs HDOP
    ax1.scatter(hdops, hpls, s=100, alpha=0.6, c=gdops, cmap='viridis')
    ax1.axhline(config.hal, color='r', linestyle='--', linewidth=2, label=f'HAL ({config.hal}m)')
    ax1.set_xlabel('HDOP', fontsize=12)
    ax1.set_ylabel('HPL (m)', fontsize=12)
    ax1.set_title('Protection Level vs Horizontal DOP', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar.set_label('GDOP', fontsize=10)
    
    # Distribution
    ax2.hist(hpls, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(config.hal, color='r', linestyle='--', linewidth=2, label=f'HAL ({config.hal}m)')
    ax2.axvline(np.mean(hpls), color='green', linestyle='-', linewidth=2, label=f'Mean ({np.mean(hpls):.1f}m)')
    ax2.set_xlabel('HPL (m)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('HPL Distribution (Different Geometries)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('geometry_impact.png', dpi=150, bbox_inches='tight')
    print("  Saved: geometry_impact.png")
    plt.close()


def main():
    print("\n" + "="*60)
    print("  RAIM Protection Level Analysis")
    print("="*60)
    
    plot_protection_levels_vs_satellites()
    plot_stanford_diagram()
    plot_geometry_impact()
    
    # Run some quick analysis
    print("\n" + "-"*60)
    print("  Summary Statistics")
    print("-"*60)
    
    config = RAIMConfig(pfa=1e-5, pmd=1e-3, hal=40.0, val=50.0)
    raim = RAIM(config)
    receiver_pos = np.array([-742000.0, -5462000.0, 3198000.0])
    
    # Compare 6 vs 12 satellites
    for n_sats in [6, 8, 12]:
        sat_geometry = create_satellite_geometry(receiver_pos, num_satellites=n_sats, seed=42)
        satellites = simulate_pseudoranges(receiver_pos, sat_geometry, noise_std=2.0, seed=100)
        result = raim.detect_fault(satellites)
        
        print(f"\n  {n_sats} Satellites:")
        print(f"    GDOP: {result.dop['gdop']:.2f}")
        print(f"    HDOP: {result.dop['hdop']:.2f}")
        print(f"    HPL:  {result.hpl:.1f} m ({result.hpl/config.hal*100:.0f}% of HAL)")
        print(f"    VPL:  {result.vpl:.1f} m ({result.vpl/config.val*100:.0f}% of VAL)")
        print(f"    Status: {'OK' if result.hpl < config.hal else 'EXCEEDS HAL'}")
    
    print("\n" + "="*60)
    print("  Analysis complete!")
    print("  Generated 3 visualization files")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
