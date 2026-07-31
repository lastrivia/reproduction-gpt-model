import time
import torch
# import torch_npu

device = torch.device("cuda:0")
torch.cuda.set_device(device)

n = 8192
a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
b = torch.randn(n, n, device=device, dtype=torch.bfloat16)

for _ in range(10):
    c = a @ b

torch.cuda.synchronize()

loop_count = 500

start = time.perf_counter()
for _ in range(loop_count):
    c = a @ b
torch.cuda.synchronize()

elapsed = time.perf_counter() - start
print("elapsed:", elapsed)
print("TFLOP/s:", 2 * n**3 * loop_count / elapsed / 1e12)