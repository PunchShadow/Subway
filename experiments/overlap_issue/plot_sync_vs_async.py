#!/usr/bin/env python3
"""
Comparison plot: BFS-sync vs BFS-async per-iteration profiling.
Generates:
  1. Side-by-side stacked bar charts (time breakdown)
  2. Frontier size comparison
  3. Cumulative time comparison

Usage:
    python3 plot_sync_vs_async.py [sync_csv] [async_csv] [output_prefix]
"""

import sys
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_csv(path):
    data = {'gItr': [], 'numActiveNodes': [], 'numActiveEdges': [],
            'cpuPackingMs': [], 'h2dMs': [], 'gpuComputeMs': []}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['gItr'].append(int(row['gItr']))
            data['numActiveNodes'].append(int(row['numActiveNodes']))
            data['numActiveEdges'].append(int(row['numActiveEdges']))
            data['cpuPackingMs'].append(float(row['cpuPackingMs']))
            data['h2dMs'].append(float(row['h2dMs']))
            data['gpuComputeMs'].append(float(row['gpuComputeMs']))
    for k in data:
        data[k] = np.array(data[k])
    return data


def main():
    sync_csv = sys.argv[1] if len(sys.argv) > 1 else "bfs_sync_profile_uk2007_source10.csv"
    async_csv = sys.argv[2] if len(sys.argv) > 2 else "bfs_async_profile_uk2007_source10.csv"
    prefix = sys.argv[3] if len(sys.argv) > 3 else "sync_vs_async"

    sync = load_csv(sync_csv)
    async_ = load_csv(async_csv)

    # Skip async iter 1 (initial setup with all 105M nodes)
    for k in async_:
        async_[k] = async_[k][1:]

    # =========================================================================
    # Figure 1: Side-by-side stacked bar charts
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('BFS Sync vs Async: Per-Iteration CPU Bottleneck Comparison\n'
                 '(uk-2007, source=10, RTX A4000)', fontsize=14, fontweight='bold')

    colors = {'cpu': '#e74c3c', 'h2d': '#3498db', 'gpu': '#2ecc71'}

    for col, (label, d) in enumerate([('BFS-sync', sync), ('BFS-async', async_)]):
        ax = axes[0][col]
        n = len(d['gItr'])
        x = np.arange(n)
        total = d['cpuPackingMs'] + d['h2dMs'] + d['gpuComputeMs']

        ax.bar(x, d['gpuComputeMs'], 0.7, label='GPU Compute', color=colors['gpu'], edgecolor='white', linewidth=0.3)
        ax.bar(x, d['h2dMs'], 0.7, bottom=d['gpuComputeMs'], label='H2D Transfer', color=colors['h2d'], edgecolor='white', linewidth=0.3)
        ax.bar(x, d['cpuPackingMs'], 0.7, bottom=d['gpuComputeMs'] + d['h2dMs'], label='CPU Packing', color=colors['cpu'], edgecolor='white', linewidth=0.3)

        # Percentage labels (only for bars > 2ms total, to avoid clutter)
        for i in range(n):
            if total[i] > 2:
                pct = d['cpuPackingMs'][i] / total[i] * 100
                ax.text(x[i], total[i] + max(total)*0.01, f'{pct:.0f}%',
                        ha='center', va='bottom', fontsize=5.5, color='#c0392b', fontweight='bold')

        step = max(1, n // 20)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(d['gItr'].astype(int)[::step], fontsize=7)
        ax.set_ylabel('Time (ms)')
        ax.set_xlabel('Iteration')
        ax.set_title(f'{label} — Time Breakdown ({n} iters, total {total.sum():.1f} ms)', fontsize=11)
        ax.legend(fontsize=8, loc='upper right')

        # Bottom row: frontier size
        ax2 = axes[1][col]
        ax2.bar(x, d['numActiveEdges'] / 1e6, 0.7, color='#95a5a6', edgecolor='white', linewidth=0.3)
        ax2.set_xticks(x[::step])
        ax2.set_xticklabels(d['gItr'].astype(int)[::step], fontsize=7)
        ax2.set_ylabel('Active Edges (M)')
        ax2.set_xlabel('Iteration')
        ax2.set_title(f'{label} — Frontier Size', fontsize=11)

    plt.tight_layout()
    fig.savefig(f'{prefix}_breakdown.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {prefix}_breakdown.png")

    # =========================================================================
    # Figure 2: Aggregated comparison bar chart
    # =========================================================================
    fig2, (ax_agg, ax_pct) = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle('BFS Sync vs Async: Aggregate Comparison (uk-2007, source=10)',
                  fontsize=13, fontweight='bold')

    labels_fw = ['BFS-sync', 'BFS-async']
    for idx, (label, d) in enumerate([(labels_fw[0], sync), (labels_fw[1], async_)]):
        cpu_sum = d['cpuPackingMs'].sum()
        h2d_sum = d['h2dMs'].sum()
        gpu_sum = d['gpuComputeMs'].sum()
        total_sum = cpu_sum + h2d_sum + gpu_sum

        ax_agg.bar(idx - 0.2, gpu_sum, 0.35, color=colors['gpu'], label='GPU Compute' if idx == 0 else '')
        ax_agg.bar(idx, h2d_sum, 0.35, color=colors['h2d'], label='H2D Transfer' if idx == 0 else '')
        ax_agg.bar(idx + 0.2, cpu_sum, 0.35, color=colors['cpu'], label='CPU Packing' if idx == 0 else '')

        # Text annotation
        ax_agg.text(idx - 0.2, gpu_sum + 5, f'{gpu_sum:.0f}', ha='center', fontsize=8)
        ax_agg.text(idx, h2d_sum + 5, f'{h2d_sum:.0f}', ha='center', fontsize=8)
        ax_agg.text(idx + 0.2, cpu_sum + 5, f'{cpu_sum:.0f}', ha='center', fontsize=8)

        # Percentage stacked bar
        ax_pct.barh(idx, gpu_sum / total_sum * 100, 0.5, color=colors['gpu'])
        ax_pct.barh(idx, h2d_sum / total_sum * 100, 0.5, left=gpu_sum / total_sum * 100, color=colors['h2d'])
        ax_pct.barh(idx, cpu_sum / total_sum * 100, 0.5,
                    left=(gpu_sum + h2d_sum) / total_sum * 100, color=colors['cpu'])
        ax_pct.text(50, idx, f'CPU: {cpu_sum/total_sum*100:.1f}%  H2D: {h2d_sum/total_sum*100:.1f}%  GPU: {gpu_sum/total_sum*100:.1f}%',
                    ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    ax_agg.set_xticks([0, 1])
    ax_agg.set_xticklabels(labels_fw)
    ax_agg.set_ylabel('Total Time (ms)')
    ax_agg.set_title('Absolute Time by Component')
    ax_agg.legend(fontsize=9)

    ax_pct.set_yticks([0, 1])
    ax_pct.set_yticklabels(labels_fw)
    ax_pct.set_xlabel('Percentage (%)')
    ax_pct.set_title('Relative Time Distribution')
    ax_pct.set_xlim(0, 100)

    plt.tight_layout()
    fig2.savefig(f'{prefix}_aggregate.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {prefix}_aggregate.png")

    # =========================================================================
    # Print summary table
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY: BFS-sync vs BFS-async (uk-2007, source=10, RTX A4000)")
    print("="*80)

    for label, d in [('BFS-sync', sync), ('BFS-async', async_)]:
        cpu_sum = d['cpuPackingMs'].sum()
        h2d_sum = d['h2dMs'].sum()
        gpu_sum = d['gpuComputeMs'].sum()
        total_sum = cpu_sum + h2d_sum + gpu_sum
        n = len(d['gItr'])
        peak_edges = d['numActiveEdges'].max()
        peak_iter = d['gItr'][d['numActiveEdges'].argmax()]

        print(f"\n--- {label} ---")
        print(f"  Iterations:       {n}")
        print(f"  Total time:       {total_sum:.1f} ms")
        print(f"  CPU Packing:      {cpu_sum:.1f} ms ({cpu_sum/total_sum*100:.1f}%)")
        print(f"  H2D Transfer:     {h2d_sum:.1f} ms ({h2d_sum/total_sum*100:.1f}%)")
        print(f"  GPU Compute:      {gpu_sum:.1f} ms ({gpu_sum/total_sum*100:.1f}%)")
        print(f"  Peak frontier:    {peak_edges:,} edges (iter {int(peak_iter)})")
        print(f"  Avg CPU Pack/iter:{cpu_sum/n:.2f} ms")


if __name__ == '__main__':
    main()
