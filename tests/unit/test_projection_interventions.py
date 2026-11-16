import pytest
import torch
from measure_gguf_kl import swapped
from measure_projection_interventions import validate_plan


def test_rejects_overlapping_or_unknown_replacements():
    plan = {
        "sources": {"a": {}, "b": {}},
        "tasks": [{"name": "swap", "base": "a", "replacements": {"a": ["0:q"], "b": ["0:q"]}}],
    }
    with pytest.raises(ValueError, match="only once"):
        validate_plan(plan, ["0:q"])
    plan["tasks"][0]["replacements"] = {"a": ["1:q"]}
    with pytest.raises(ValueError, match="invalid replacement"):
        validate_plan(plan, ["0:q"])


def test_nested_intervention_restores_compressed_base_then_teacher_on_failure():
    weight = torch.tensor([[1.0, 2.0]])
    original = weight.clone()
    compressed = torch.tensor([[3.0, 4.0]])
    replacement = torch.tensor([[5.0, 6.0]])
    with swapped({"q": (weight, compressed)}):
        with pytest.raises(RuntimeError), swapped({"q": (weight, replacement)}):
            assert torch.equal(weight, replacement)
            raise RuntimeError("scoring failed")
        assert torch.equal(weight, compressed)
    assert torch.equal(weight, original)


def test_accepts_clean_and_compressed_contexts_but_rejects_ambiguous_names():
    plan = {
        "sources": {"a": {}},
        "tasks": [
            {"name": "isolated", "base": "clean", "replacements": {"a": ["0:q"]}},
            {"name": "joint", "base": "a"},
        ],
    }
    validate_plan(plan, ["0:q"])
    plan["tasks"].append(plan["tasks"][0])
    with pytest.raises(ValueError, match="unique"):
        validate_plan(plan, ["0:q"])
