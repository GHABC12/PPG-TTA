# SIGMA-PPG upstream compatibility patch

Upstream commit: `b1cc8d90da3878958b890c6e352cd3ae2a644005`.

The published checkpoint and the cloned model-selection code did not match in two places:

1. Checkpoint attention blocks contain QKV weights but no learned q/v bias tensors. The default model selector created q/v bias parameters, which stayed randomly initialized. The local selector passes `qkv_bias=False` to match the checkpoint architecture.
2. The selector's position embedding resize omitted the CLS token, and the single-channel forward path expanded positions across time a second time. The adapter restores the original CLS position and interpolates patch positions to `n_patches`; the forward path now expands positions only for explicit multi-channel input.

The checkpoint patch projection has input shape `(360, 156)`. With the official temporal convolution, this corresponds to 100 input samples per patch, so the 20-second, 50 Hz pilot uses 10 patches of 2 seconds each. The README's 50-sample patch quick start leaves this projection unmatched.

Smoke evidence is in `results/sigma_smoke.log`: the adapted `sigma_ppg_pro` backbone loads 253/253 backbone tensors (100%), and a `(2, 1, 1000)` input returns finite `(2, 360)` features. The two missing task-head tensors are expected because the SpO2 head is trained from source labels.

