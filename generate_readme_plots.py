#!/usr/bin/env python3
"""
Generate publication-quality visualizations for README
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.raim import RAIM, RAIMConfig, create_satellite_geometry, simulate_pseudoranges

# Publication-quality settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['lines.linewidth'] = 2

# Color-blind safe palette (from ColorBrewer)
COLORS = {
    'safe_green': '#4daf4a',   # Standardized green for safe/acceptable/nominal regions (both figures)
    'warning': '#ff7f00',      # Orange (misleading info)
    'danger': '#e41a1c',       # Red (HMI)
    'unavailable': '#984ea3',  # Purple (system unavailable)
    'hpl': '#377eb8',          # Blue (horizontal)
    'vpl': '#e41a1c',          # Red (vertical)
    'threshold': '#333333',    # Dark gray (limits)
    'grid': '#cccccc'          # Light gray (gridlines)
}


def generate_stanford_diagram():
    """Create publication-quality Stanford diagram for GNSS integrity"""
    print("Generating Stanford diagram...")
    
    fig, ax = plt.subplots(figsize=(9, 9))
    
    hal = 40.0
    
    # Define the four integrity regions with subtle colors (using standardized green)
    # Region 1: Normal Operation
    ax.add_patch(Rectangle((0, 0), hal, hal, 
                          facecolor=COLORS['safe_green'], alpha=0.25, edgecolor='none'))
    ax.text(hal/2, hal/2, 'Normal\nOperation', 
            ha='center', va='center', fontsize=13, fontweight='600')
    
    # Region 2: Misleading Information
    ax.add_patch(Rectangle((hal, 0), 80, hal, 
                          facecolor=COLORS['warning'], alpha=0.25, edgecolor='none'))
    ax.text(hal+40, hal/2, 'Misleading\nInformation', 
            ha='center', va='center', fontsize=13, fontweight='600')
    
    # Region 3: Hazardously Misleading Information (HMI) - full standards-compliant term
    ax.add_patch(Rectangle((0, hal), hal, 80, 
                          facecolor=COLORS['danger'], alpha=0.25, edgecolor='none'))
    ax.text(hal/2, hal+40, 'Hazardously Misleading\nInformation (HMI)', 
            ha='center', va='center', fontsize=12, fontweight='600', color='#8B0000')
    
    # Region 4: System Unavailable - increased contrast with darker text
    ax.add_patch(Rectangle((hal, hal), 80, 80, 
                          facecolor=COLORS['unavailable'], alpha=0.2, edgecolor='none'))
    ax.text(hal+40, hal+40, 'System\nUnavailable', 
            ha='center', va='center', fontsize=12, fontweight='700', color='#1a0d20')
    
    # Critical threshold lines - thinner than region boundaries (1.8 vs potential 2.5)
    ax.axhline(hal, color=COLORS['threshold'], linestyle='--', linewidth=1.8, 
               label='Alert Limit (HAL)', zorder=5)
    ax.axvline(hal, color=COLORS['threshold'], linestyle='--', linewidth=1.8, zorder=5)
    
    # Threshold labels - positioned cleanly
    ax.text(hal, -2.5, 'HAL', fontsize=11, fontweight='600', ha='center', va='top')
    ax.text(-2.5, hal, 'HAL', fontsize=11, fontweight='600', ha='right', va='center')
    
    # Plot representative operational points - reduced marker sizes by ~12%
    examples = [
        (22, 12, 'Nominal', COLORS['safe_green'], 'o', 9.5),
        (15, 70, 'HMI Risk', COLORS['danger'], 'X', 10.5),
        (85, 85, 'Detected', COLORS['unavailable'], 's', 8)
    ]
    
    for hpl, pe, label, color, marker, msize in examples:
        ax.plot(hpl, pe, marker, markersize=msize, color=color, 
                markeredgecolor='white', markeredgewidth=1.5, zorder=10)
        
        # Minimal annotations with arrows
        if label == 'Nominal':
            ax.annotate('', xy=(hpl, pe), xytext=(10, 25),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color=color))
            ax.text(10, 27, label, fontsize=10, color=color, fontweight='600')
        elif label == 'HMI Risk':
            ax.annotate('', xy=(hpl, pe), xytext=(8, 95),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color=color))
            ax.text(8, 97, label, fontsize=10, color=color, fontweight='600')
    
    ax.set_xlabel('Horizontal Protection Level, HPL (m)', fontsize=12, fontweight='600')
    ax.set_ylabel('Position Error, PE (m)', fontsize=12, fontweight='600')
    
    # Title hierarchy: bold 700 for title
    ax.set_title('Stanford Integrity Diagram (Horizontal Plane)', 
                 fontsize=14, fontweight='700', pad=15)
    
    ax.set_xlim(-5, 125)
    ax.set_ylim(-5, 125)
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.8, color=COLORS['grid'])
    ax.set_aspect('equal')
    
    # Simplified annotation
    textstr = 'Goal: Minimize P(HMI)'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                 alpha=0.9, edgecolor=COLORS['threshold'], linewidth=1.5)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', 
            bbox=props, style='italic')
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Export as high-resolution PNG (300 DPI)
    plt.savefig('docs/stanford_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: docs/stanford_diagram.png")
    plt.close()


def generate_protection_levels_plot():
    """Generate publication-quality protection levels vs satellite count"""
    print("Generating protection levels plot...")
    
    config = RAIMConfig(pfa=1e-5, pmd=1e-3, hal=40.0, val=50.0)
    raim = RAIM(config)
    receiver_pos = np.array([-742000.0, -5462000.0, 3198000.0])
    
    n_sats_range = range(5, 13)
    hpls = []
    vpls = []
    
    for n_sats in n_sats_range:
        sat_geometry = create_satellite_geometry(receiver_pos, num_satellites=n_sats, seed=42)
        satellites = simulate_pseudoranges(receiver_pos, sat_geometry, noise_std=2.0, seed=100)
        result = raim.detect_fault(satellites)
        
        hpls.append(result.hpl)
        vpls.append(result.vpl)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot protection levels with reduced marker sizes (~12% smaller)
    ax.plot(n_sats_range, hpls, 'o-', linewidth=2.5, markersize=7, 
            color=COLORS['hpl'], label='HPL', markeredgecolor='white', 
            markeredgewidth=1.5, zorder=3)
    ax.plot(n_sats_range, vpls, 's-', linewidth=2.5, markersize=7, 
            color=COLORS['vpl'], label='VPL', markeredgecolor='white', 
            markeredgewidth=1.5, zorder=3)
    
    # Alert limit reference lines - clean and minimal
    ax.axhline(config.hal, color=COLORS['hpl'], linestyle='--', linewidth=2, 
               alpha=0.7, label='HAL (40 m)', zorder=2)
    ax.axhline(config.val, color=COLORS['vpl'], linestyle='--', linewidth=2, 
               alpha=0.7, label='VAL (50 m)', zorder=2)
    
    # Subtle shading for acceptable region (using standardized green)
    ax.fill_between(n_sats_range, 1, config.hal, alpha=0.08, 
                    color=COLORS['safe_green'], zorder=1)
    
    ax.set_xlabel('Number of Satellites', fontsize=12, fontweight='600')
    ax.set_ylabel('Protection Level (m)', fontsize=12, fontweight='600')
    
    # Title hierarchy: bold 700 for title
    ax.set_title('Protection Levels vs Satellite Count (NPA Integrity)', 
                 fontsize=14, fontweight='700', pad=12)
    
    # Legend with lighter border and subtle transparency
    legend = ax.legend(fontsize=10, loc='upper right', framealpha=0.92, 
                       edgecolor='#bbbbbb', fancybox=False, frameon=True)
    legend.get_frame().set_linewidth(1.0)
    
    ax.grid(True, alpha=0.2, which='both', linestyle=':', linewidth=0.8, 
            color=COLORS['grid'])
    ax.set_yscale('log')
    ax.set_ylim(15, 300)
    ax.set_xlim(4.8, 12.2)
    ax.set_xticks(n_sats_range)
    
    # Annotation pointing precisely at HPL value at N=8
    hpl_at_8 = hpls[3]  # Index 3 corresponds to n_sats=8
    ax.annotate('NPA integrity satisfied at ≥8 satellites', 
                xy=(8, hpl_at_8), xytext=(9.8, 25),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['safe_green']),
                fontsize=10, color=COLORS['safe_green'], fontweight='600',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         alpha=0.9, edgecolor=COLORS['safe_green'], linewidth=1.5))
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Export as high-resolution PNG (300 DPI)
    plt.savefig('docs/protection_levels.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: docs/protection_levels.png")
    plt.close()


def main():
    print("\n" + "="*60)
    print("  Generating README Visualizations")
    print("="*60 + "\n")
    
    import os
    os.makedirs('docs', exist_ok=True)
    
    generate_stanford_diagram()
    generate_protection_levels_plot()
    
    print("\n" + "="*60)
    print("  ✓ All visualizations generated successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
