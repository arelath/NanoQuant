## Per-layer objective-weighted reconstruction

| Block | Layer | Rank | Export weighted normalized error | Raw normalized error | Bits |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `mlp.gate_proj` | 2208 | 0.183407 | 0.287688 | 28297728 |
| 0 | `mlp.up_proj` | 2176 | 0.251015 | 0.300865 | 27887616 |
| 0 | `mlp.down_proj` | 1792 | 0.189517 | 0.362583 | 22966272 |
| 0 | `self_attn.v_proj` | 704 | 0.219753 | 0.252966 | 2534400 |
| 0 | `self_attn.o_proj` | 1056 | 0.168078 | 0.284537 | 4882944 |
| 0 | `self_attn.q_proj` | 960 | 0.210400 | 0.305146 | 4439040 |
| 0 | `self_attn.k_proj` | 608 | 0.250335 | 0.326526 | 2188800 |

## Final frozen block error before model-level KD

| Block | Source reference | Block entry pre-quantization | Final frozen pre-KD | Final − block entry | Relative vs block entry | Final − source reference | Relative vs source |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000000 | 0.000000 | 9.044044 | 9.044044 | n/a | 9.044044 | n/a |
