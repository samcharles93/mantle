#!/usr/bin/env python3
"""Parse nsys stats output and identify bottlenecks."""
import sys
import re
from collections import defaultdict

def parse_nsys_stats(filename):
    """Parse nsys stats file and extract key metrics."""

    with open(filename, 'r') as f:
        content = f.read()

    # Parse CUDA kernel times
    kernels = {}
    in_kernel_section = False

    for line in content.split('\n'):
        if 'CUDA GPU Kernel Summary' in line:
            in_kernel_section = True
            continue

        if in_kernel_section and line.strip() and not line.startswith('-'):
            # Parse kernel line: Time(%) Total Time Instances Avg Min Max Name
            parts = line.split()
            if len(parts) >= 7 and parts[0].replace('.', '').isdigit():
                try:
                    time_pct = float(parts[0])
                    total_time_ns = float(parts[1].replace(',', ''))
                    instances = int(parts[2].replace(',', ''))
                    name = ' '.join(parts[6:])

                    kernels[name] = {
                        'time_pct': time_pct,
                        'total_ns': total_time_ns,
                        'instances': instances,
                        'avg_ns': total_time_ns / instances if instances > 0 else 0
                    }
                except (ValueError, IndexError):
                    pass

        if in_kernel_section and line.strip() == '':
            in_kernel_section = False

    # Parse CUDA API calls
    api_calls = {}
    in_api_section = False

    for line in content.split('\n'):
        if 'CUDA API Summary' in line:
            in_api_section = True
            continue

        if in_api_section and line.strip() and not line.startswith('-'):
            parts = line.split()
            if len(parts) >= 7 and parts[0].replace('.', '').isdigit():
                try:
                    time_pct = float(parts[0])
                    total_time_ns = float(parts[1].replace(',', ''))
                    instances = int(parts[2].replace(',', ''))
                    name = ' '.join(parts[6:])

                    api_calls[name] = {
                        'time_pct': time_pct,
                        'total_ns': total_time_ns,
                        'instances': instances,
                        'avg_ns': total_time_ns / instances if instances > 0 else 0
                    }
                except (ValueError, IndexError):
                    pass

        if in_api_section and line.strip() == '':
            in_api_section = False

    # Parse memory operations
    mem_ops = {}
    in_mem_section = False

    for line in content.split('\n'):
        if 'CUDA GPU Memory' in line or 'Memory Operation' in line:
            in_mem_section = True
            continue

        if in_mem_section and line.strip() and not line.startswith('-'):
            parts = line.split()
            if len(parts) >= 7 and parts[0].replace('.', '').isdigit():
                try:
                    time_pct = float(parts[0])
                    total_time_ns = float(parts[1].replace(',', ''))
                    count = int(parts[2].replace(',', ''))
                    name = ' '.join(parts[6:])

                    mem_ops[name] = {
                        'time_pct': time_pct,
                        'total_ns': total_time_ns,
                        'count': count,
                        'avg_ns': total_time_ns / count if count > 0 else 0
                    }
                except (ValueError, IndexError):
                    pass

        if in_mem_section and line.strip() == '':
            in_mem_section = False

    return kernels, api_calls, mem_ops

def analyze_bottleneck(kernels, api_calls, mem_ops):
    """Identify the primary bottleneck."""

    # Calculate total times
    total_kernel_time = sum(k['total_ns'] for k in kernels.values())
    total_api_time = sum(a['total_ns'] for a in api_calls.values())
    total_mem_time = sum(m['total_ns'] for m in mem_ops.values())

    # Get top kernels
    sorted_kernels = sorted(kernels.items(), key=lambda x: x[1]['total_ns'], reverse=True)

    # Get sync operations
    sync_ops = {k: v for k, v in api_calls.items() if 'Sync' in k or 'sync' in k}
    total_sync_time = sum(s['total_ns'] for s in sync_ops.values())

    # Get memory copy operations
    memcpy_ops = {k: v for k, v in api_calls.items() if 'Memcpy' in k or 'memcpy' in k}
    total_memcpy_time = sum(m['total_ns'] for m in memcpy_ops.values())

    print("=" * 60)
    print("CUDA PERFORMANCE ANALYSIS (from nsys)")
    print("=" * 60)
    print()

    print("Time Distribution:")
    print(f"  Kernel execution:  {total_kernel_time/1e6:>10.2f} ms ({total_kernel_time/(total_kernel_time+total_api_time)*100:.1f}%)")
    print(f"  API overhead:      {total_api_time/1e6:>10.2f} ms ({total_api_time/(total_kernel_time+total_api_time)*100:.1f}%)")
    print(f"  Memory transfers:  {total_memcpy_time/1e6:>10.2f} ms ({total_memcpy_time/total_api_time*100:.1f}% of API)")
    print(f"  Synchronization:   {total_sync_time/1e6:>10.2f} ms ({total_sync_time/total_api_time*100:.1f}% of API)")
    print()

    print("=" * 60)
    print("TOP 10 KERNELS BY TIME")
    print("=" * 60)
    print(f"{'Kernel':<40} {'Time (ms)':<12} {'Calls':<8} {'%'}")
    print("-" * 60)

    for name, data in sorted_kernels[:10]:
        kernel_name = name[:40]
        print(f"{kernel_name:<40} {data['total_ns']/1e6:>10.2f}  {data['instances']:>6}  {data['time_pct']:>5.1f}%")

    print()
    print("=" * 60)
    print("BOTTLENECK ANALYSIS")
    print("=" * 60)

    # Determine bottleneck
    if total_sync_time > total_kernel_time * 0.2:
        print("⚠️  PRIMARY BOTTLENECK: Synchronization overhead")
        print(f"   Sync time: {total_sync_time/1e6:.1f}ms ({total_sync_time/total_kernel_time*100:.0f}% of kernel time)")
        print("   → Reduce cudaStreamSynchronize calls")
        print("   → Batch operations, use async APIs")
        print()

    if total_memcpy_time > total_kernel_time * 0.3:
        print("⚠️  PRIMARY BOTTLENECK: Memory transfers")
        print(f"   Transfer time: {total_memcpy_time/1e6:.1f}ms ({total_memcpy_time/total_kernel_time*100:.0f}% of kernel time)")
        print("   → Keep data on GPU")
        print("   → Fuse kernels to eliminate transfers")
        print()

    # Check for small kernel spam
    small_kernels = [k for k, v in kernels.items() if v['avg_ns'] < 10000]  # < 10us
    if len(small_kernels) > 100:
        print(f"⚠️  WARNING: {len(small_kernels)} kernels with <10μs runtime")
        print("   → Kernel launch overhead dominates")
        print("   → Fuse small kernels together")
        print()

    # Identify hottest kernel
    if sorted_kernels:
        hottest = sorted_kernels[0]
        if hottest[1]['time_pct'] > 20:
            print(f"⚠️  HOTSPOT: {hottest[0]}")
            print(f"   Takes {hottest[1]['time_pct']:.1f}% of GPU time ({hottest[1]['total_ns']/1e6:.1f}ms)")
            print(f"   Called {hottest[1]['instances']} times, {hottest[1]['avg_ns']/1000:.1f}μs each")
            print("   → Optimize this kernel for best gains")
            print()

    print("=" * 60)
    print("OPTIMIZATION PRIORITY")
    print("=" * 60)

    # Rank optimizations by impact
    priorities = []

    if total_sync_time/total_kernel_time > 0.2:
        priorities.append(("Reduce synchronizations", total_sync_time/1e6))

    if total_memcpy_time/total_kernel_time > 0.3:
        priorities.append(("Reduce memory transfers", total_memcpy_time/1e6))

    if sorted_kernels and sorted_kernels[0][1]['time_pct'] > 20:
        priorities.append((f"Optimize {sorted_kernels[0][0][:30]}", sorted_kernels[0][1]['total_ns']/1e6))

    if len(small_kernels) > 100:
        total_small = sum(kernels[k]['total_ns'] for k in small_kernels)
        priorities.append(("Fuse small kernels", total_small/1e6))

    priorities.sort(key=lambda x: x[1], reverse=True)

    for i, (opt, time_ms) in enumerate(priorities, 1):
        print(f"{i}. {opt}")
        print(f"   Potential savings: {time_ms:.1f}ms")

    if not priorities:
        print("No major bottlenecks detected. Performance is well-balanced.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: parse_nsys_stats.py <nsys_stats_file>")
        sys.exit(1)

    stats_file = sys.argv[1]

    try:
        kernels, api_calls, mem_ops = parse_nsys_stats(stats_file)
        analyze_bottleneck(kernels, api_calls, mem_ops)
    except FileNotFoundError:
        print(f"Error: File not found: {stats_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing stats: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
