import argparse, time, math, gc, signal, sys
import torch
from contextlib import contextmanager

BEST = {"full_n": None, "mps": []}
VERBOSE = False
STATUS_INTERVAL = 0

def log(*a, **k):
    if VERBOSE: print(*a, **k, flush=True)

def bytes_fmt(b):
    for unit in ["B","KB","MB","GB","TB","PB"]:
        if b < 1024: return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} EB"

@contextmanager
def cuda_safety():
    try:
        yield
    except torch.cuda.OutOfMemoryError:
        raise
    finally:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

def gpu_mem_info():
    free, total = torch.cuda.mem_get_info()
    used = total - free
    return free, used, total

def heartbeat(start_t):
    if STATUS_INTERVAL <= 0: return
    now = time.time()
    if now - heartbeat.last >= STATUS_INTERVAL:
        free, used, total = gpu_mem_info()
        print(f"[status +{now-start_t:.1f}s] mem: used={bytes_fmt(used)} free={bytes_fmt(free)} total={bytes_fmt(total)}", flush=True)
        heartbeat.last = now
heartbeat.last = 0.0

def try_full_state(n, dtype, safety_frac, start_t):
    elem_bytes = (8 if dtype==torch.complex64 else 16)
    need = (2**n) * elem_bytes
    free, _, _ = gpu_mem_info()
    budget = int(free * safety_frac)
    log(f"  [full] try n={n} need~{bytes_fmt(need)} budget~{bytes_fmt(budget)}")
    if need > budget:
        return False, need, budget
    try:
        with cuda_safety():
            x = torch.empty((2**n,), device="cuda", dtype=dtype)
            # do a bit of compute to light up SMs
            it = 30
            for _ in range(it):
                x.mul_(1.0000001 + 0j if dtype.is_complex else 1.0000001)
            del x
        heartbeat(start_t)
        return True, need, budget
    except torch.cuda.OutOfMemoryError:
        heartbeat(start_t)
        return False, need, budget

def max_full_state_qubits(dtype, safety_frac, start_t):
    free, _, _ = gpu_mem_info()
    elem = (8 if dtype==torch.complex64 else 16)
    try_upper = int(math.log2(max(1, safety_frac*free/elem)))
    lo, hi, best = 0, max(1, try_upper+2), -1
    while lo <= hi:
        mid = (lo+hi)//2
        ok, _, _ = try_full_state(mid, dtype, safety_frac, start_t)
        print(f"    [full] n={mid} -> {'OK' if ok else 'OOM'}", flush=True)
        if ok:
            best = mid; lo = mid + 1
            BEST["full_n"] = max(BEST["full_n"] or -1, best)
        else:
            hi = mid - 1
    return best

def mps_memory_bytes(n, chi, d, dtype_bytes):
    return n * d * (chi**2) * dtype_bytes

def try_mps_chain(n, chi, d, dtype, sanity_contractions, sweep_repeats, start_t):
    device="cuda"
    dtype_bytes = 8 if dtype==torch.float64 else 4
    need = mps_memory_bytes(n, chi, d, dtype_bytes)
    free, _, _ = gpu_mem_info()
    if need > 0.98*free:
        log(f"    [mps χ={chi}] skip n={n}, mem need~{bytes_fmt(need)} > free~{bytes_fmt(free)}")
        return False, need
    try:
        with cuda_safety():
            one = 1
            cores = []
            for i in range(n):
                left = chi if i>0 else one
                right = chi if i<n-1 else one
                core = torch.randn(left, d, right, device=device, dtype=dtype) * 0.01
                cores.append(core)
            v0 = torch.zeros(d, device=device, dtype=dtype); v0[0]=1
            for s in range(sanity_contractions):
                vecs = [v0 if (i%2==0) else torch.nn.functional.softmax(
                        torch.randn(d, device=device, dtype=dtype), dim=0) for i in range(n)]
                msg = torch.ones(1, device=device, dtype=dtype)
                for _rep in range(sweep_repeats):
                    for i in range(n):
                        core = cores[i]                        # [l,d,r]
                        mv = torch.tensordot(core, vecs[i], dims=([1],[0]))  # [l,r]
                        msg = torch.tensordot(msg, mv, dims=([0],[0]))       # [r]
            _ = (msg**2).sum().item()
            del cores
        heartbeat(start_t)
        return True, need
    except torch.cuda.OutOfMemoryError:
        heartbeat(start_t)
        return False, need

def sweep_mps_max_n(chi_list, d, dtype, sanity_contractions, sweep_repeats, start_t):
    results = []
    for chi in chi_list:
        print(f"[mps] χ={chi} start", flush=True)
        n = 8
        last_ok = 0
        while True:
            ok, _ = try_mps_chain(n, chi, d, dtype, sanity_contractions, sweep_repeats, start_t)
            print(f"    [mps χ={chi}] n={n} -> {'OK' if ok else 'OOM/skip'}", flush=True)
            if ok:
                last_ok = n
                n = int(n * 1.5) + 1
                if n > 200_000: break
            else:
                break
        lo, hi = last_ok, max(last_ok+1, n-1)
        best = last_ok
        while lo <= hi:
            mid = (lo+hi)//2
            ok, _ = try_mps_chain(mid, chi, d, dtype, sanity_contractions, sweep_repeats, start_t)
            print(f"    [mps χ={chi}] refine n={mid} -> {'OK' if ok else 'OOM/skip'}", flush=True)
            if ok:
                best = mid; lo = mid + 1
            else:
                hi = mid - 1
        results.append((chi, best))
        BEST["mps"].append((chi, best))
        print(f"[mps] χ={chi} best n≈{best}", flush=True)
    return results

def on_interrupt(signum, frame):
    print("\n— Interrupted — partial results:", flush=True)
    if BEST["full_n"] is not None:
        print(f"  full-state max n so far ≈ {BEST['full_n']}")
    if BEST["mps"]:
        for chi, n in BEST["mps"]:
            print(f"  χ={chi} -> n≈{n}")
    sys.exit(1)

def main():
    global VERBOSE, STATUS_INTERVAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--status-interval", type=float, default=0, help="seconds between mem status lines (0=off)")

    ap.add_argument("--safety-frac", type=float, default=0.80, help="VRAM headroom for full-state alloc")
    ap.add_argument("--full-dtype", choices=["c64","c128"], default="c64")
    ap.add_argument("--mps-dtype", choices=["f32","f64"], default="f32")
    ap.add_argument("--chi", type=str, default="32,64,128")
    ap.add_argument("--sanity", type=int, default=16, help="number of local contraction passes")
    ap.add_argument("--sweep-repeats", type=int, default=4, help="inner repeats to increase compute per pass")
    args = ap.parse_args()

    VERBOSE = args.verbose
    STATUS_INTERVAL = args.status_interval

    signal.signal(signal.SIGINT, on_interrupt)

    dtype_full = torch.complex64 if args.full_dtype=="c64" else torch.complex128
    dtype_mps = torch.float32 if args.mps_dtype=="f32" else torch.float64
    chis = [int(x) for x in args.chi.split(",") if x.strip()]

    print("Torch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Device:", torch.cuda.get_device_name(0))
    free, used, total = gpu_mem_info()
    print(f"GPU Memory: free={bytes_fmt(free)}, used={bytes_fmt(used)}, total={bytes_fmt(total)}", flush=True)

    start_t = time.time()
    # Full state-vector capacity
    print(f"[full] dtype={'complex64' if dtype_full==torch.complex64 else 'complex128'} safety={args.safety_frac}", flush=True)
    n_full = max_full_state_qubits(dtype_full, args.safety_frac, start_t)
    if n_full >= 0:
        elems = 2**n_full
        each = 8 if dtype_full==torch.complex64 else 16
        print(f"[full] max n ≈ {n_full} qubits ({'c64' if each==8 else 'c128'}, {each} B/amp), len={elems:,}", flush=True)
    else:
        print("[full] could not allocate safely.", flush=True)

    # MPS capacity
    print(f"[mps] dtype={dtype_mps}, chi={chis}, sanity={args.sanity}, repeats={args.sweep_repeats}", flush=True)
    mps_results = sweep_mps_max_n(chis, d=2, dtype=dtype_mps,
                                  sanity_contractions=args.sanity,
                                  sweep_repeats=args.sweep_repeats,
                                  start_t=start_t)
    for chi, n_mps in mps_results:
        approx_mem = mps_memory_bytes(n_mps, chi, d=2,
                                      dtype_bytes=(8 if dtype_mps==torch.float64 else 4))
        print(f"  χ={chi:>4} -> max n ≈ {n_mps:>7} qubits (approx MPS mem {bytes_fmt(approx_mem)})", flush=True)

    # Structured/product states: memory-only bound (64 B/qubit model)
    per_qb_bytes = 64
    free, used, total = gpu_mem_info()
    linear_max_n = int((0.9*free) // per_qb_bytes)
    print(f"[structured/QIH] memory-only upper bound ≈ {linear_max_n} qubits (@~{per_qb_bytes} B/qubit)", flush=True)

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA GPU required"
    torch.backends.cudnn.benchmark = True
    main()
