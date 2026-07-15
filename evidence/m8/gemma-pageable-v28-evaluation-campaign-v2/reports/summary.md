# Run gemma-pageable-v28-evaluation-campaign-v2

- Status: `completed`
- Experiment: `none` — Gemma v28 replay-to-full gate
- Purpose: Demonstrate ordered replay, quick, standard, and full evaluation promotion.
- Hypothesis: The parity candidate passes frozen quality, size, runtime, and memory gates.
- Baseline: `contemporary-legacy-018-and-compatible-llama.cpp`
- Config hash: `sha256:aee0382c2527d9637217993b0804c788ccf8a26eee3645aa5cf347cbf353b61d`

## Outcome

- Conclusion: Candidate progressed from retained layer replay through quick, standard, and full evaluation.
- Recommended next action: Candidate passed the complete evaluation campaign; proceed to migration/release qualification.

## Launcher

- Kind: `python`
- Zero-argument numbered runfile: **no**
- Repository-relative path: `tools/run_evaluation_campaign.py`
- Content hash: `sha256:4640d28b74c4e9c58c3152259dca666edb2e4593ea52da80be64113a682ada6a`
- Code revision: `31892cbe063fd1415c76a059ffc1aabb31b4d3eb`
- Arguments: `[]`

## Execution

- Created: `2026-07-15T19:54:24.654595+00:00`
- Updated: `2026-07-15T19:54:24.674594+00:00`
- Attempts: 1
- Resume attempts: 0
- Events: 7 (0 warnings, 0 errors)
- Lifecycle: `run.started, run.completed`
- Terminal event: `run.completed`
- Terminal context: `{"conclusion": "Candidate progressed from retained layer replay through quick, standard, and full evaluation.", "host.peak_private_bytes": 1739096064, "host.peak_working_set_bytes": 533057536, "host.private_bytes": 1737969664, "host.working_set_bytes": 531992576, "peak_device_bytes": 1592178176, "recommended_next_action": "Candidate passed the complete evaluation campaign; proceed to migration/release qualification.", "wall_seconds": 0.01618610000150511}`

## Environment

```json
{
  "environment": {},
  "machine": "AMD64",
  "packages": {
    "Jinja2": "3.1.6",
    "MarkupSafe": "3.0.3",
    "PyYAML": "6.0.3",
    "Pygments": "2.20.0",
    "aiohappyeyeballs": "2.7.1",
    "aiohttp": "3.14.1",
    "aiosignal": "1.4.0",
    "anyio": "4.14.1",
    "ast_serialize": "0.6.0",
    "attrs": "26.1.0",
    "certifi": "2026.6.17",
    "charset-normalizer": "3.4.9",
    "colorama": "0.4.6",
    "cut-cross-entropy": "25.1.1",
    "datasets": "5.0.0",
    "dill": "0.4.1",
    "filelock": "3.29.0",
    "frozenlist": "1.8.0",
    "fsspec": "2026.4.0",
    "h11": "0.16.0",
    "httpcore": "1.0.9",
    "httpx": "0.28.1",
    "huggingface_hub": "0.36.2",
    "hypothesis": "6.156.6",
    "idna": "3.18",
    "iniconfig": "2.3.0",
    "librt": "0.13.0",
    "mpmath": "1.3.0",
    "multidict": "6.7.1",
    "multiprocess": "0.70.19",
    "mypy": "2.2.0",
    "mypy_extensions": "1.1.0",
    "nanoquant-rewrite": "0.1.0",
    "nanoquant-runtime": "0.1.0",
    "networkx": "3.6.1",
    "numpy": "2.4.4",
    "packaging": "26.2",
    "pandas": "3.0.3",
    "pathspec": "1.1.1",
    "pillow": "12.2.0",
    "pip": "26.1.2",
    "pluggy": "1.6.0",
    "propcache": "0.5.2",
    "pyarrow": "25.0.0",
    "pytest": "9.1.1",
    "python-dateutil": "2.9.0.post0",
    "regex": "2026.7.10",
    "requests": "2.34.2",
    "ruff": "0.15.21",
    "safetensors": "0.8.0",
    "sentencepiece": "0.2.2",
    "setuptools": "78.1.0",
    "six": "1.17.0",
    "sortedcontainers": "2.4.0",
    "sympy": "1.14.0",
    "tokenizers": "0.21.4",
    "torch": "2.12.1+cu130",
    "torchvision": "0.28.0+cu130",
    "tqdm": "4.68.4",
    "transformers": "4.51.3",
    "triton-windows": "3.7.1.post27",
    "types-PyYAML": "6.0.12.20260518",
    "typing_extensions": "4.15.0",
    "tzdata": "2026.3",
    "urllib3": "2.7.0",
    "xxhash": "3.8.1",
    "yarl": "1.24.2"
  },
  "platform": "Windows-11-10.0.26200-SP0",
  "python": "3.12.1 (tags/v3.12.1:2305ca5, Dec  7 2023, 22:03:25) [MSC v.1937 64 bit (AMD64)]"
}
```

## Cost

- Manifest elapsed seconds: 0.019999
- Peak device bytes: 1592178176
- Peak host bytes: 1739096064
- Peak temporary disk bytes: n/a

| Sequence | Stage | Event | Metric | Value | Unit |
| ---: | --- | --- | --- | ---: | --- |
| 1 | `run` | `run.started` | `host.peak_private_bytes` | 1738092544 | bytes |
| 1 | `run` | `run.started` | `host.peak_working_set_bytes` | 532955136 | bytes |
| 1 | `run` | `run.started` | `host.private_bytes` | 1737969664 | bytes |
| 1 | `run` | `run.started` | `host.working_set_bytes` | 531980288 | bytes |
| 5 | `evaluation` | `evaluation.full.completed` | `long_context_peak_device_bytes` | 1592178176 | bytes |
| 7 | `run` | `run.completed` | `host.peak_private_bytes` | 1739096064 | bytes |
| 7 | `run` | `run.completed` | `host.peak_working_set_bytes` | 533057536 | bytes |
| 7 | `run` | `run.completed` | `host.private_bytes` | 1737969664 | bytes |
| 7 | `run` | `run.completed` | `host.working_set_bytes` | 531992576 | bytes |
| 7 | `run` | `run.completed` | `peak_device_bytes` | 1592178176 | bytes |
| 7 | `run` | `run.completed` | `wall_seconds` | 0.01618610000150511 | seconds |

## Lineage

No parent run; this is a root run.

## Issues

No warning or error events.

## Artifacts

- `sha256:e5d41f15860dbfa4dd65e9aa49bee86012b0cc7b95265b271da46412fd89855d`
- `sha256:6cb0e8ec4df90a50eadc7d9bdd80ad03a5d6f2e00d5649a7fe1606c1f4c9dc70`
- `sha256:39f8eca06a140893ba6db8ae8bb4ddbb0b19548b5dd18d992dc1d50c05059b24`
- `sha256:c17f2cf200fcd151d880c4389ae191a4c6dfb296628047abc950803b06db3cf4`
- `sha256:95d464e1a0bf90c2afa85973ff910358198243a2e9302b20b67a2844fa264892`
- `sha256:a4dc8f2bdaf92808a5c2a2c8c983a3b1949ec5313f108a5a7be2f8f9979307f1`
- `sha256:b0a1b22bd6208bfc437429aad8c2ea3e7c545d97d1ce2105167924dd58af1fae`
- `sha256:ccacea00ee1b4e7678ae9f662827fd45cf5ec28071918c88ef67f799a0f35cc8`
