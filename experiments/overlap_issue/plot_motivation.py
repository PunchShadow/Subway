#!/usr/bin/env python3
"""
Motivation Experiment: CPU Bottleneck Profiling for Subway BFS
Generates a stacked bar chart showing per-iteration breakdown of:
  - CPU Packing Time (subgraph generation on CPU)
  - H2D Transfer Time (host-to-device edge copy)
  - GPU Compute Time (BFS kernel execution)

Usage:
    python3 plot_motivation.py [csv_file] [output_png]
"""

import sys
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "bfs_profile_uk2007_source1895.csv"
    output_png = sys.argv[2] if len(sys.argv) > 2 else "motivation_cpu_bottleneck.png"

    iters = []
    active_nodes = []
    active_edges = []
    cpu_packing = []
    h2d = []
    gpu_compute = []

    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            iters.append(int(row['gItr']))
            active_nodes.append(int(row['numActiveNodes']))
            active_edges.append(int(row['numActiveEdges']))
            cpu_packing.append(float(row['cpuPackingMs']))
            h2d.append(float(row['h2dMs']))
            gpu_compute.append(float(row['gpuComputeMs']))

    # Skip iteration 1 (initial setup with ALL nodes active — not a real BFS frontier)
    iters = iters[1:]
    active_nodes = active_nodes[1:]
    active_edges = active_edges[1:]
    cpu_packing = cpu_packing[1:]
    h2d = h2d[1:]
    gpu_compute = gpu_compute[1:]

    cpu_packing = np.array(cpu_packing)
    h2d = np.array(h2d)
    gpu_compute = np.array(gpu_compute)
    total = cpu_packing + h2d + gpu_compute

    x = np.arange(len(iters))
    width = 0.6

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Subway BFS Motivation: CPU Packing Bottleneck\n(uk-2007, source=1895, RTX A4000)',
                 fontsize=14, fontweight='bold')

    # --- Top: Stacked bar chart ---
    bars_gpu = ax1.bar(x, gpu_compute, width, label='GPU Compute', color='#2ecc71', edgecolor='white', linewidth=0.5)
    bars_h2d = ax1.bar(x, h2d, width, bottom=gpu_compute, label='H2D Transfer', color='#3498db', edgecolor='white', linewidth=0.5)
    bars_cpu = ax1.bar(x, cpu_packing, width, bottom=gpu_compute + h2d, label='CPU Packing (Subgraph Gen)', color='#e74c3c', edgecolor='white', linewidth=0.5)

    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_xlabel('BFS Global Iteration', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i) for i in iters])
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_title('Per-Iteration Time Breakdown', fontsize=12)

    # Add percentage labels for CPU packing ratio
    for i in range(len(iters)):
        pct = cpu_packing[i] / total[i] * 100
        if total[i] > 0.5:  # Only label bars with meaningful time
            ax1.text(x[i], total[i] + 0.3, f'{pct:.0f}%',
                     ha='center', va='bottom', fontsize=7, color='#c0392b', fontweight='bold')

    ax1.set_ylim(0, max(total) * 1.15)

    # --- Bottom: Active edges per iteration (context) ---
    ax2.bar(x, [e / 1e6 for e in active_edges], width, color='#95a5a6', edgecolor='white', linewidth=0.5)
    ax2.set_ylabel('Active Edges (M)', fontsize=11)
    ax2.set_xlabel('BFS Global Iteration', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(i) for i in iters])
    ax2.set_title('Frontier Size (Active Edges per Iteration)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    print(f"Plot saved to: {output_png}")

    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"{'Iter':>4} {'ActiveNodes':>12} {'ActiveEdges':>12} {'CpuPack(ms)':>12} {'H2D(ms)':>10} {'GPU(ms)':>10} {'CPU%':>6}")
    for i in range(len(iters)):
        pct = cpu_packing[i] / total[i] * 100
        print(f"{iters[i]:4d} {active_nodes[i]:12,} {active_edges[i]:12,} {cpu_packing[i]:12.3f} {h2d[i]:10.3f} {gpu_compute[i]:10.3f} {pct:5.1f}%")

    print(f"\nOverall CPU Packing fraction: {cpu_packing.sum() / total.sum() * 100:.1f}%")
    print(f"Overall H2D fraction: {h2d.sum() / total.sum() * 100:.1f}%")
    print(f"Overall GPU Compute fraction: {gpu_compute.sum() / total.sum() * 100:.1f}%")

if __name__ == '__main__':
    main()
