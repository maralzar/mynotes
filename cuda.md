
including_emulation=False: Specifies whether to consider emulated bfloat16 support. When False, the function only returns True if the hardware natively supports bfloat16 operations. If True, it would also return True for devices that emulate bfloat16 (e.g., via software or lower-precision operations)
, but this is less common and typically slower.
```
supported = torch.cuda.is_bf16_supported(including_emulation=False)
dtype16 = (torch.bfloat16 if supported else torch.float16)
```
