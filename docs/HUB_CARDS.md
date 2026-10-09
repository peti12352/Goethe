# Hub card contract

Maintainers with write access to `meshapplied` on Hugging Face should keep cards aligned with this repo.

## Canonical ids

- `meshapplied/Qwen3.8-Flash-Next-Ternary-Latest`
- `meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest`

## Required card text

- DeepSeek: first sentence must say **ternary expert overlay**, not NVFP4 weights.
- `fetch()` examples must use the canonical ids above.
- Clone URL: `https://github.com/meshapplied/Goethe.git`
- Flash-Next ΔNLL “better than bf16” must stay **unpublished or caveated** until reproduced with the eval script in-tree.
- DeepSeek held-out ΔNLL vs FP4 (overall +0.202, general +0.415) stays; that is the honest quality signal.

## Legacy

`meshapplied/DeepSeek-V4.1-Flash-NVFP4` should remain a Hub redirect only.
