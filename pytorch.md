# PyTorch’s autocast context manager
```
with torch.autocast(device_type="cuda", dtype=torch.float16):
%timeit mixed32(torch.randn(1000, 1000, dtype=torch.float32, device='cuda'))
```
