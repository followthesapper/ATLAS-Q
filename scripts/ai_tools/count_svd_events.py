#!/usr/bin/env python
"""Quick count of SVD events in logs."""
import gzip, json
from pathlib import Path

log_dir = Path("runs/svd_logs")
log_files = list(log_dir.glob("*.jsonl.gz"))

print(f"📊 Found {len(log_files)} log files\n")

total_events = 0
file_stats = []
corrupted = []

for log_file in log_files:
    try:
        with gzip.open(log_file, 'rt') as f:
            events = sum(1 for _ in f)
            total_events += events
            file_stats.append((log_file.name, events))
    except (EOFError, gzip.BadGzipFile) as e:
        corrupted.append(log_file.name)

# Sort by event count
file_stats.sort(key=lambda x: x[1], reverse=True)

print(f"{'File':<50} {'Events':>10}")
print("="*62)
for name, count in file_stats[:10]:  # Show top 10
    print(f"{name:<50} {count:>10,}")

if len(file_stats) > 10:
    print(f"... and {len(file_stats)-10} more files")

print("="*62)
print(f"{'TOTAL':<50} {total_events:>10,}")
print("="*62)

if corrupted:
    print(f"\n⚠️  {len(corrupted)} corrupted files (skipped):")
    for name in corrupted[:5]:
        print(f"  - {name}")
    if len(corrupted) > 5:
        print(f"  ... and {len(corrupted)-5} more")

print(f"\n✅ You have {total_events:,} valid SVD events!")
print(f"   From {len(file_stats)} valid files out of {len(log_files)} total")
print(f"   That's enough to retrain the AI on REAL circuit data!")
