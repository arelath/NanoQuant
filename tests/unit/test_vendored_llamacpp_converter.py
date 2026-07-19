from pathlib import Path

from nanoquant.infrastructure.io_utils import hash_canonical_text_file
from nanoquant.runtime.packed import PACKED_REFERENCE_CONVERTER_SHA256


def test_vendored_llamacpp_converter_matches_packed_provenance() -> None:
    repository = Path(__file__).resolve().parents[2]
    converter = repository / "tools" / "llamacpp" / "convert_nanoquant_to_gguf.py"

    assert hash_canonical_text_file(converter) == PACKED_REFERENCE_CONVERTER_SHA256
    assert (converter.parent / "LICENSE").is_file()
