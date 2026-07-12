# Milestone 0 Baseline Record

Capture: `evidence/m0/20260712T052926Z/manifest.json`

The initial capture records both reference revisions and complete binary dirty patches, the legacy package environment,
host/GPU identity, modified llama.cpp build caches, the converter, `nanoquant.cu`, three benchmark JSON files, and the
Experiment 019 launcher. The capture tool intentionally never reads `.env` or credentials.

Key identities:

| Evidence | Identity |
| --- | --- |
| Legacy revision | `c89b72085725a689b1fd99ec8e64c5671b18d4b4` |
| Legacy dirty patch | `sha256:21d556c11fde0c49624adaf293693568eeda011406d268ca44be0249d80d4550` |
| Modified llama.cpp revision | `5c6ae79816ee0f2b3d4bb8ec9061c294185d320b` |
| llama.cpp dirty patch | `sha256:511da84ca41185e5335cb41dc63112809512925665c5f539a94302d493becbad` |
| `nanoquant.cu` | `sha256:5c87336c2b6b8fb33805c6ee6a8752d4bd364beed63fd4cca03c2b36be966619` |
| GGUF conversion script | `sha256:92b0d31c1ce83d0fe3668bbb20cee6a4da24ec3e9476f6699890d01540241e4d` |
| Experiment 019 effective config | `sha256:6e78bdf615e611cddd8edc9764ff058c57cf561b69b3a93155dd8dd47f57dbc4` |
| Experiment 019 Markdown golden | `sha256:5ae66b5070fba8c6a22918e27b67386c642b977414280b1179e8c8476397c5e8` |
| Experiment 019 CSV golden | `sha256:82d531ed0b7beb65dafd57eecb862ed62623442c19f28caa1f79189d8965d7ce` |

The complete effective legacy configuration contains all 95 dataclass fields after launcher overrides. It is stored at
`evidence/m0/20260712T052926Z/legacy/experiment-019-effective-config.json`; this also exposes the launcher/code mismatch
where the Experiment 019 file prints an Experiment 016 title, which the rewrite's launcher-number validation prevents.

## Host assignments

The RTX 4000 Ada laptop captured in the manifest is the reference development host and may be used for functional CUDA
checks. It is not accepted as the stable performance host because laptop power/thermal behavior is variable. The stable
performance host and large-model host remain unassigned; tasks requiring those hosts stay open. A performance host must
record fixed power limit, application clocks where supported, driver, CUDA, thermal steady state, and idle-health checks.

## Reproduction

```powershell
.\.venv\Scripts\python.exe tools\capture_m0_baseline.py `
  --legacy D:\dev\research\NanoQuant-OfficalCode `
  --llama D:\dev\research\llama.cpp
```

Use `tools/extract_legacy_experiment.py` under the legacy virtual environment to rematerialize every effective Experiment
019 field without loading model weights or starting quantization.
