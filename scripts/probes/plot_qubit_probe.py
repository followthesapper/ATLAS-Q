import re, sys, json
from pathlib import Path
import argparse
import datetime as dt

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

STATUS_RE = re.compile(r"\[status \+(?P<secs>[\d\.]+)s\] mem: used=(?P<used>[\d\.]+) GB free=(?P<free>[\d\.]+) GB total=(?P<tot>[\d\.]+) GB")
FULL_TRY_RE = re.compile(r"\s*\[full\]\s+try n=(?P<n>\d+)\s+need~(?P<need>[\d\.]+)\s*(?P<unit>MB|GB)\s+budget~(?P<budget>[\d\.]+)\s*GB")
FULL_OK_RE  = re.compile(r"\s*\[full\]\s+n=(?P<n>\d+)\s*->\s*OK")
FULL_OOM_RE = re.compile(r"\s*\[full\]\s+n=(?P<n>\d+)\s*->\s*OOM")
FULL_MAX_RE = re.compile(r"\[full\]\s+max n ≈ (?P<n>\d+)\s*qubits.*")

MPS_START_RE = re.compile(r"\[mps\]\s+dtype=(?P<dtype>\S+),\s+chi=\[(?P<chi>[^]]+)\],\s+sanity=(?P<sanity>\d+),\s+repeats=(?P<rep>\d+)")
MPS_CHI_START_RE = re.compile(r"\s*\[mps χ=(?P<chi>\d+)\] start")
MPS_OK_RE = re.compile(r"\s*\[mps χ=(?P<chi>\d+)\]\s+n=(?P<n>\d+)\s*->\s*OK")

def parse_log(text):
    status = []
    full_ok = []
    full_oom = []
    mps_ok = []
    full_max = None
    runs = []

    for line in text.splitlines():
        m = STATUS_RE.search(line)
        if m:
            status.append({
                "t_sec": float(m.group("secs")),
                "mem_used_gb": float(m.group("used")),
                "mem_free_gb": float(m.group("free")),
                "mem_total_gb": float(m.group("tot")),
            })
            continue

        m = FULL_TRY_RE.search(line)  # we don't store TRY rows, but you could
        m = FULL_OK_RE.search(line) or m
        if m and "n" in m.groupdict() and "-> OK" in line:
            full_ok.append(int(m.group("n")))
            continue

        m = FULL_OOM_RE.search(line)
        if m:
            full_oom.append(int(m.group("n")))
            continue

        m = FULL_MAX_RE.search(line)
        if m:
            full_max = int(m.group("n"))
            continue

        m = MPS_CHI_START_RE.search(line)
        if m:
            runs.append({"chi": int(m.group("chi")), "ok_ns": []})
            continue

        m = MPS_OK_RE.search(line)
        if m:
            chi = int(m.group("chi")); n = int(m.group("n"))
            if not runs or runs[-1]["chi"] != chi:
                runs.append({"chi": chi, "ok_ns": []})
            runs[-1]["ok_ns"].append(n)
            mps_ok.append((chi, n))
            continue

    # summarize max per chi
    per_chi_max = {}
    for chi, n in mps_ok:
        per_chi_max[chi] = max(per_chi_max.get(chi, 0), n)

    return {
        "status": status,
        "full_ok": full_ok,
        "full_oom": full_oom,
        "full_max": full_max,
        "mps_ok": mps_ok,
        "per_chi_max": per_chi_max,
        "runs": runs,
    }

def write_csv(out_csv: Path, data):
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("kind,chi,n,t_sec,mem_used_gb,mem_free_gb\n")
        for s in data["status"]:
            f.write(f"status,,,{s['t_sec']:.3f},{s['mem_used_gb']:.3f},{s['mem_free_gb']:.3f}\n")
        for chi, n in data["mps_ok"]:
            f.write(f"mps,{chi},{n},,,\n")
        for n in data["full_ok"]:
            f.write(f"full,,{n},,,\n")

def plot_figs(prefix: Path, data, title_suffix=""):
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # 1) Memory over time
    if data["status"]:
        t = [s["t_sec"] for s in data["status"]]
        used = [s["mem_used_gb"] for s in data["status"]]
        free = [s["mem_free_gb"] for s in data["status"]]

        plt.figure()
        plt.plot(t, used, label="GPU mem used (GB)")
        plt.plot(t, free, label="GPU mem free (GB)")
        plt.xlabel("Time (s)"); plt.ylabel("GB")
        plt.title(f"GPU Memory vs Time{title_suffix}")
        plt.legend(); plt.tight_layout()
        plt.savefig(prefix.with_name(prefix.name + "_mem.png"))
        plt.close()

    # 2) Capacity per chi (scatter of OK points + max marker)
    if data["mps_ok"]:
        from collections import defaultdict
        bins = defaultdict(list)
        for chi, n in data["mps_ok"]:
            bins[chi].append(n)

        plt.figure()
        for chi, ns in sorted(bins.items()):
            xs = [chi]*len(ns)
            plt.scatter(xs, ns, s=10, alpha=0.6, label=f"χ={chi}")
            # max point
            plt.scatter([chi], [max(ns)], s=80, marker="^")

        plt.xlabel("Bond dimension χ")
        plt.ylabel("Max sequence length n (OK)")
        plt.title(f"MPS Capacity by χ{title_suffix}")
        plt.xticks(sorted(bins.keys()))
        plt.tight_layout()
        plt.savefig(prefix.with_name(prefix.name + "_mps_capacity.png"))
        plt.close()

    # 3) Full-state headline
    if data["full_max"] is not None:
        plt.figure()
        plt.bar(["full (complex64)"], [data["full_max"]])
        plt.ylabel("Max qubits (n)")
        plt.title(f"Full-state Capacity{title_suffix}")
        plt.tight_layout()
        plt.savefig(prefix.with_name(prefix.name + "_full_capacity.png"))
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", type=Path, help="Path to tee'd log (e.g., runs/qubit_probe_*.log)")
    ap.add_argument("--prefix", type=Path, default=None, help="Prefix for outputs (default: runs/<logname_basename>)")
    args = ap.parse_args()

    text = args.logfile.read_text(encoding="utf-8", errors="ignore")
    data = parse_log(text)

    if args.prefix is None:
        prefix = Path("runs") / args.logfile.stem
    else:
        prefix = args.prefix

    # Save CSV + JSON summary
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".summary.json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(csv_path, data)
    json_path.write_text(json.dumps({
        "full_max_qubits": data["full_max"],
        "mps_per_chi_max": data["per_chi_max"],
    }, indent=2))

    # Make plots
    ts = dt.datetime.now().strftime(" (%Y-%m-%d %H:%M)")
    plot_figs(prefix, data, title_suffix=ts)

    print(f"✅ Wrote: {csv_path}")
    print(f"✅ Wrote: {json_path}")
    for fig in ["_mem.png", "_mps_capacity.png", "_full_capacity.png"]:
        print(f"✅ Figure: {prefix.name}{fig}")

if __name__ == "__main__":
    main()
