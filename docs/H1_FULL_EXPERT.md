A20 optional full expert projection gate, CPU prepared only.

After HARDWARE_GATE and reservation of a V100, from the relocated H1 bundle:
  VF_H1_RESIDENT_IO=safetensors VF_H1_FULL_EXPERT=1 bash run_h1.sh
VF_GPU_PYTHON and VF_GPU_UUID are required by the existing launcher.
VF_H1_FULL_EXPERT_SAMPLE may override the path, but its bytes must match the pinned fixture.
Default remains the existing tiny canary. No GPU access occurs during CPU qualification.

The optional gate exports and admits the pinned Qwen3.6 expert0/layer0 before provider creation.
It exercises gate/up [512,2048] and down [2048,512], six single nonzero inputs per plane,
amplitudes +/-0.5 and columns spanning all four split-K partitions. Each projection synchronizes
before strict comparison. This is three projection tests, not a complete MoE layer or model.
CPU reference follows F32 global multiplication, FP16 global/scale/product intermediates, and
FP32 one-term product then FP16 output. Absolute and relative tolerance are zero.
The native CPU oracle and a separately indexed packed Torch oracle agree for all 18432 outputs.
A mismatch on GPU fails this gate; it is not permission to silently change tolerance.
PTX leaves MMA accumulation rounding and subnormal handling unspecified. V100 behavior is pending.
Chosen real fixture columns contain no subnormal decoded weights or outputs. The biased FP4
decoder operand for magnitude 0.5 is subnormal (0.5/16384); its actual probe count is reported.
Scalar CPU tests also cover subnormal and underflow admission. GPU subnormal handling is pending. A green gate does not close all H1.

The CPU test provider is explicitly test-only. Service weights remain compressed; only bounded
diagnostic inputs/outputs and selected-column CPU reference outputs are materialized.
Checkpoint bytes are SHA-pinned once per preparation; the oracle and Rust exporter consume
those same bounded bytes through a Linux anonymous memfd sealed against writes, growth and shrink,
passed as an inherited descriptor at the export boundary. There is no replaceable snapshot pathname
or cleanup unlink race. The source is pinned again before GPU execution. Artifacts and Python
tensors remain mutable: fingerprints detect prior changes, not a concurrent change during copying,
so artifacts and tensors must stay quiescent through launch. A process with ptrace control over the
caller or root access remains outside this local ownership boundary. No new dependency, dense
resident fallback, performance claim or hardware change.
