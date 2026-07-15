# Gemma v28 evaluation campaign

- Candidate: `gemma-pageable-v28`
- Baseline: `contemporary-legacy-018-and-compatible-llama.cpp`
- Outcome: **full-promotion**
- Campaign identity: `sha256:d40956eab035349c948a3fd1ae5117a6580113ba3cc1d84f20ce9af2433b9480`
- Recommended next action: Candidate passed the complete evaluation campaign; proceed to migration/release qualification.

## Promotion path

| Stage | Decision | Metrics | Policy |
| --- | --- | --- | --- |
| layer-replay | promotion | `{"mean_absolute_block_loss_percent_delta": 0.3540303168941772}` | captured replay bound |
| quick | promotion | `{"artifact_complete": 1.0, "effective_bpw": 0.9963181446312268, "maximum_block_loss_percent_delta": 4.218750167013708, "rank_mismatch_count": 0.0}` | `sha256:cd7e42beb9fd8cec9ce730ea8679d84ac0e68d4e3966e57c1d2b22563fad3be4` |
| standard | promotion | `{"wikitext2_percent_delta": 2.07912036388717, "wikitext2_scored_target_count": 8128.0}` | `sha256:6e12ea2c682ab1ef44befec4883dad522c6deb492fa97fd8cfafff1007911ef9` |
| full | promotion | `{"long_context_exact": 1.0, "long_context_peak_device_bytes": 1592178176.0, "runtime_reference_ratio": 0.8712380699565503, "unexpected_fallback_count": 0.0}` | `sha256:da65f4d2a9142c79e3c9e61128d93dfcd671023fa6adda11a0734db1f4878f54` |

## Retained inputs

| Input | Bytes | SHA-256 |
| --- | ---: | --- |
| `inputs/full-parity-summary.json` | 2402 | `e5d41f15860dbfa4dd65e9aa49bee86012b0cc7b95265b271da46412fd89855d` |
| `inputs/legacy-comparison.json` | 8770 | `6cb0e8ec4df90a50eadc7d9bdd80ad03a5d6f2e00d5649a7fe1606c1f4c9dc70` |
| `inputs/llamacpp-runtime.json` | 5814 | `39f8eca06a140893ba6db8ae8bb4ddbb0b19548b5dd18d992dc1d50c05059b24` |
| `inputs/long-context.json` | 2818 | `c17f2cf200fcd151d880c4389ae191a4c6dfb296628047abc950803b06db3cf4` |
| `inputs/rewrite-runtime.json` | 6899 | `95d464e1a0bf90c2afa85973ff910358198243a2e9302b20b67a2844fa264892` |
| `inputs/validation.json` | 4387 | `a4dc8f2bdaf92808a5c2a2c8c983a3b1949ec5313f108a5a7be2f8f9979307f1` |
| `inputs/wikitext2.json` | 1362 | `b0a1b22bd6208bfc437429aad8c2ea3e7c545d97d1ce2105167924dd58af1fae` |

The directory contains copied immutable inputs, canonical evaluator outputs, structured lifecycle and
promotion events, the resolved campaign intent/environment, cost observations, and this comparison.
