#!/usr/bin/env python3
"""
Basic RAIM Fault Detection Example

Demonstrates how RAIM detects satellite faults using chi-square testing
on measurement residuals.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from src.raim import (
    RAIM, RAIMConfig, Satellite,
    simulate_pseudoranges, create_satellite_geometry
)


def plot_satellite_geometry(satellites, receiver_pos, title="Satellite Constellation"):
    """Plot satellite positions in sky plot (azimuth/elevation)"""
    fig = plt.figure(figsize=(14, 5))
    
    # Create polar subplot for sky plot
    ax1 = plt.subplot(121, projection='polar')
    ax2 = plt.subplot(122, projection='3d')
    
    # Sky plot
    for sat in satellites:
        dx = sat.x - receiver_pos[0]
        dy = sat.y - receiver_pos[1]
        dz = sat.z - receiver_pos[2]
        
        # Convert to azimuth/elevation
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        elevation = np.arcsin(dz / dist) * 180 / np.pi
        azimuth = np.arctan2(dx, dy) * 180 / np.pi
        
        # Polar plot
        r = 90 - elevation
        theta = azimuth * np.pi / 180
        ax1.plot(theta, r, 'o', markersize=10)
        ax1.text(theta, r-5, f"PRN{sat.prn}", ha='center', fontsize=8)
    
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_ylim(0, 90)
    ax1.set_yticks([0, 30, 60, 90])
    ax1.set_yticklabels(['90°', '60°', '30°', '0°'])
    ax1.set_title('Sky Plot (Azimuth/Elevation)')
    ax1.grid(True)
    
    # 3D scatter
    for sat in satellites:
        ax2.scatter(sat.x/1e6, sat.y/1e6, sat.z/1e6, s=50, c='blue')
        ax2.text(sat.x/1e6, sat.y/1e6, sat.z/1e6, f" {sat.prn}", fontsize=8)
    
    ax2.scatter(receiver_pos[0]/1e6, receiver_pos[1]/1e6, receiver_pos[2]/1e6, 
                s=100, c='red', marker='^', label='Receiver')
    ax2.set_xlabel('X (1000 km)')
    ax2.set_ylabel('Y (1000 km)')
    ax2.set_zlabel('Z (1000 km)')
    ax2.set_title('3D Positions (ECEF)')
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('satellite_geometry.png', dpi=150, bbox_inches='tight')
    print("  Saved: satellite_geometry.png")


def plot_residuals(result, title="Residual Analysis"):
    """Plot measurement residuals"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    prns = np.arange(1, len(result.residuals) + 1)
    colors = ['red' if abs(r) > 10 else 'blue' for r in result.residuals]
    
    ax.bar(prns, result.residuals, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Satellite PRN')
    ax.set_ylabel('Residual (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add status text
    status_text = f"Test Stat: {result.test_statistic:.2f}\n"
    status_text += f"Threshold: {result.threshold:.2f}\n"
    status_text += f"Fault: {'YES' if result.test_statistic > result.threshold else 'NO'}"
    ax.text(0.98, 0.97, status_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, family='monospace')
    
    plt.tight_layout()
    return fig


def main():
    print("\n" + "="*60)
    print("  RAIM Fault Detection Demo")
    print("="*60)
    
    # Setup
    config = RAIMConfig(pfa=1e-5, pmd=1e-3, hal=40.0, val=50.0)
    raim = RAIM(config)
    
    receiver_pos = np.array([-742000.0, -5462000.0, 3198000.0])
    sat_geometry = create_satellite_geometry(receiver_pos, num_satellites=8, seed=42)
    
    print(f"\n  Receiver position: {receiver_pos/1e3}")
    print(f"  Satellites: {len(sat_geometry)}")
    print(f"  Alert limits: HAL={config.hal}m, VAL={config.val}m")
    
    # Plot geometry
    print("\n  Generating geometry plot...")
    plot_satellite_geometry(sat_geometry, receiver_pos)
    
    # Test 1: Normal operation
    print("\n" + "-"*60)
    print("  Scenario 1: Normal Operation (no faults)")
    print("-"*60)
    
    satellites = simulate_pseudoranges(
        receiver_pos, sat_geometry,
        noise_std=2.0, seed=123
    )
    
    result = raim.detect_fault(satellites)
    print(f"  Status: {result.status.value}")
    print(f"  Test statistic: {result.test_statistic:.2f}")
    print(f"  Threshold: {result.threshold:.2f}")
    print(f"  HPL: {result.hpl:.1f} m (limit: {config.hal} m)")
    print(f"  Position error: {np.linalg.norm(result.position - receiver_pos):.2f} m")
    
    fig1 = plot_residuals(result, "Normal Operation - No Fault")
    plt.savefig('residuals_normal.png', dpi=150, bbox_inches='tight')
    print("  Saved: residuals_normal.png")
    plt.close()
    
    # Test 2: Small fault
    print("\n" + "-"*60)
    print("  Scenario 2: Small Fault (15m on PRN 3)")
    print("-"*60)
    
    satellites = simulate_pseudoranges(
        receiver_pos, sat_geometry,
        noise_std=2.0, fault_prn=3, fault_magnitude=15.0, seed=123
    )
    
    result = raim.detect_fault(satellites)
    print(f"  Status: {result.status.value}")
    print(f"  Test statistic: {result.test_statistic:.2f}")
    print(f"  Threshold: {result.threshold:.2f}")
    print(f"  Position error: {np.linalg.norm(result.position - receiver_pos):.2f} m")
    
    fig2 = plot_residuals(result, "Small Fault (15m) - Below Detection Threshold")
    plt.savefig('residuals_small_fault.png', dpi=150, bbox_inches='tight')
    print("  Saved: residuals_small_fault.png")
    plt.close()
    
    # Test 3: Large fault
    print("\n" + "-"*60)
    print("  Scenario 3: Large Fault (80m on PRN 3)")
    print("-"*60)
    
    satellites = simulate_pseudoranges(
        receiver_pos, sat_geometry,
        noise_std=2.0, fault_prn=3, fault_magnitude=80.0, seed=123
    )
    
    result = raim.detect_fault(satellites)
    print(f"  Status: {result.status.value}")
    print(f"  Test statistic: {result.test_statistic:.2f}")
    print(f"  Threshold: {result.threshold:.2f}")
    print(f"  Position error: {np.linalg.norm(result.position - receiver_pos):.2f} m")
    print(f"  Largest residual: PRN {np.argmax(np.abs(result.residuals))+1}")
    
    fig3 = plot_residuals(result, "Large Fault (80m) - Fault Detected!")
    plt.savefig('residuals_large_fault.png', dpi=150, bbox_inches='tight')
    print("  Saved: residuals_large_fault.png")
    plt.close()
    
    # Fault detection sweep
    print("\n" + "-"*60)
    print("  Fault Detection Sensitivity Analysis")
    print("-"*60)
    
    fault_mags = np.linspace(0, 100, 50)
    test_stats = []
    
    for fault_mag in fault_mags:
        satellites = simulate_pseudoranges(
            receiver_pos, sat_geometry,
            noise_std=2.0, fault_prn=3, fault_magnitude=float(fault_mag), seed=123
        )
        result = raim.detect_fault(satellites)
        test_stats.append(result.test_statistic)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fault_mags, test_stats, 'b-', linewidth=2, label='Test Statistic')
    ax.axhline(result.threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({result.threshold:.1f})')
    ax.fill_between(fault_mags, 0, result.threshold, alpha=0.2, color='green', label='No Detection')
    ax.fill_between(fault_mags, result.threshold, max(test_stats), alpha=0.2, color='red', label='Fault Detected')
    ax.set_xlabel('Fault Magnitude (m)', fontsize=12)
    ax.set_ylabel('Test Statistic', fontsize=12)
    ax.set_title('RAIM Detection Sensitivity', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('detection_sensitivity.png', dpi=150, bbox_inches='tight')
    print("  Saved: detection_sensitivity.png")
    plt.close()
    
    print("\n" + "="*60)
    print("  Analysis complete!")
    print("  Generated 5 visualization files")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
