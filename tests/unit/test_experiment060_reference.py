"""A matched successor must inherit the retained 060 recipe without template drift."""

from dataclasses import replace
from pathlib import Path

from recipes._delta import retained_config

from nanoquant.config.codec import config_hash
from tests.support.experiments import load_experiment

EXPERIMENT060_CONFIG_HASH = "sha256:3a48eafd5f6060c0e7617bec0ec94187310bee4990c7ea0c96cf147c05b2058a"


def test_frozen_060_reference_matches_the_retained_config_hash_and_current_recipe():
    reference = retained_config(Path("experiments/recipes/references/experiment060.json"), EXPERIMENT060_CONFIG_HASH)
    assert config_hash(reference) == EXPERIMENT060_CONFIG_HASH
    current = load_experiment(60).config
    rebound = replace(current, allocation=replace(
        current.allocation, kl_profile_artifact=reference.allocation.kl_profile_artifact,
        kl_profile_key=reference.allocation.kl_profile_key))
    assert config_hash(rebound) == EXPERIMENT060_CONFIG_HASH
    assert reference == rebound
