import time
import torch

device = "cuda"
torch.cuda.synchronize()  # warm up HIP context first

N = 100
size = 1_000_000  # 1M float32 = 4MB程度、実モデルの中央値に近いサイズ

# --- パターン1: 1個ずつ .to(device) + 毎回 synchronize ---
xs = [torch.empty(size, dtype=torch.float32) for _ in range(N)]
t0 = time.perf_counter()
for x in xs:
    y = x.to(device)
    torch.cuda.synchronize()
elapsed_sync_each = time.perf_counter() - t0
print(f"[1] {N}個、毎回synchronize: {elapsed_sync_each:.3f}s  (1個あたり {elapsed_sync_each/N*1000:.1f}ms)", flush=True)

# --- パターン2: 全部投げてから最後に1回だけ synchronize ---
xs2 = [torch.empty(size, dtype=torch.float32) for _ in range(N)]
t0 = time.perf_counter()
ys = [x.to(device) for x in xs2]
torch.cuda.synchronize()
elapsed_batch = time.perf_counter() - t0
print(f"[2] {N}個、最後にsynchronize: {elapsed_batch:.3f}s  (1個あたり {elapsed_batch/N*1000:.1f}ms)", flush=True)
