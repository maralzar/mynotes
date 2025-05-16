# PyTorch’s autocast context manager
```
with torch.autocast(device_type="cuda", dtype=torch.float16):
%timeit mixed32(torch.randn(1000, 1000, dtype=torch.float32, device='cuda'))
```
## 8-bit Quantization and LLM.int8()

### What is 8-bit Quantization?
Quantization reduces number precision in neural networks to save memory and speed up computation. 8-bit quantization converts high-precision numbers (e.g., float32) to 8-bit integers (int8).

- **Why?**
  - **Less memory**: Int8 uses 1/4 the space of float32.
  - **Faster**: 8-bit operations are quicker on GPUs/TPUs.
  - **Trade-off**: Lower precision may reduce accuracy.

- **How it works**:
  1. Start with float32/float16 weights and activations.
  2. Scale and round to int8 (-128 to 127).
  3. Compute with int8 for speed.
  4. Convert results back to float16 for further use.

### What is LLM.int8()?
LLM.int8() is a quantization method for large language models (LLMs) that uses 8-bit integers with minimal accuracy loss.

- **How it works**:
  1. **Split data**: Identify outliers (extreme values) and normal values in weights/inputs.
  2. **Process**:
     - Outliers: Use 16-bit (float16) for precision.
     - Normal values: Use 8-bit (int8).
  3. **Combine**: Dequantize 8-bit results to 16-bit, merge with 16-bit outlier results.

- **Why it’s clever**:
  - Saves memory and speeds up computation.
  - Preserves accuracy by handling outliers in 16-bit.
  - Enables big LLMs to run on smaller hardware.
```
bnb_config = BitsAndBytesConfig()
bnb_config_q8 = BitsAndBytesConfig(load_in_8bit=True)
model_q8 = AutoModelForCausalLM.from_pretrained(
"facebook/opt-350m", device_map='cuda:0', quantization_config=bnb_config_q8
)
```
