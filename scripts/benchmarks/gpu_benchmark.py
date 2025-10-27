import argparse, time, math
import torch
from torch import nn

def pretty(v): return f"{v:.2f}"

def bench_matmul(n=4096, it=30, dtype=torch.float16):
    a = torch.randn(n,n, device="cuda", dtype=dtype)
    b = torch.randn(n,n, device="cuda", dtype=dtype)
    for _ in range(5): (a@b).sum().item()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(it): (a@b).sum().item()
    torch.cuda.synchronize(); dt = time.time()-t0
    tflops = (2*(n**3)*it)/dt/1e12
    return tflops, dt/it

def bench_conv(it=50, b=64, c=128, h=112, w=112, k=3, dtype=torch.float16):
    x = torch.randn(b,c,h,w, device="cuda", dtype=dtype)
    conv = nn.Conv2d(c, c, k, padding=k//2, bias=False).to("cuda", dtype=dtype)
    for _ in range(10): conv(x).sum().item()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(it): conv(x).sum().item()
    torch.cuda.synchronize(); dt = time.time()-t0
    flops = b*h*w*c*(c*(k*k))*2
    return (flops*it)/dt/1e12, dt/it

def bench_fft(n=4096, it=30):
    x = (torch.randn(n,n, device="cuda") + 1j*torch.randn(n,n, device="cuda")).to(torch.complex64)
    for _ in range(5): torch.fft.fft2(x).real.sum().item()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(it): torch.fft.fft2(x).real.sum().item()
    torch.cuda.synchronize(); dt = time.time()-t0
    flops = 10*(n**2)*math.log2(n)  # rough
    return (flops*it)/dt/1e12, dt/it

def bench_einsum(n=256, r=32, it=50, dtype=torch.float16):
    # contraction similar to TN core: 'bqdn,qdor->bron'
    b = 16
    X = torch.randn(b, r, n, r, device="cuda", dtype=dtype)
    core = torch.randn(r, n, n, r, device="cuda", dtype=dtype)
    for _ in range(10): torch.einsum('bqdn,qdor->bron', X, core).sum().item()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(it): torch.einsum('bqdn,qdor->bron', X, core).sum().item()
    torch.cuda.synchronize(); dt = time.time()-t0
    return dt/it

def bench_bandwidth(n_bytes=int(2e9)):
    x = torch.empty(n_bytes//4, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize(); t0 = time.time()
    _ = x.clone()
    torch.cuda.synchronize(); dt = time.time()-t0
    gbps = (n_bytes/1e9)/dt
    return gbps, dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--size", type=int, default=4096)
    args = ap.parse_args()

    print("Torch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Device:", torch.cuda.get_device_name(0))

    # 2.9+ precision controls
    torch.backends.cudnn.benchmark = True
    if args.fp16:
        # FP16 path (Tensor Cores)
        pass
    else:
        # FP32 path (allow TF32 for speed)
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.conv.fp32_precision = "tf32"

    dtype = torch.float16 if args.fp16 else torch.float32

    tflops, tt = bench_matmul(args.size, dtype=dtype)
    print(f"[Matmul {('fp16' if args.fp16 else 'fp32')}] {pretty(tflops)} TFLOP/s, {pretty(tt*1e3)} ms/iter")

    tflops, tt = bench_conv(dtype=dtype)
    print(f"[Conv2d 3x3 ch={128 if not args.fp16 else 128} {('fp16' if args.fp16 else 'fp32')}] {pretty(tflops)} TFLOP/s (rough), {pretty(tt*1e3)} ms/iter")

    tflops, tt = bench_fft(args.size)
    print(f"[FFT 2D ({args.size}x{args.size})] {pretty(tflops)} TFLOP/s (rough), {pretty(tt*1e3)} ms/iter")

    tt = bench_einsum(dtype=dtype)
    print(f"[Einsum TN-style] {pretty(tt*1e3)} ms/iter")

    gbps, dt = bench_bandwidth()
    print(f"[Mem clone] {pretty(gbps)} GB/s over ~2 GB buffer (time {pretty(dt*1e3)} ms)")

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available"
    main()
