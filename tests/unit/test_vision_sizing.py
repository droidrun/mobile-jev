"""Tests for the per-model vision resize policy (issue #365)."""

from types import SimpleNamespace

from mobilerun.agents.utils.vision_sizing import (
    VisionResizePolicy,
    model_effective_dims,
)


def test_model_effective_dims_per_provider():
    # Standard Anthropic model downsizes; high-res Anthropic + others do not.
    assert model_effective_dims("claude-sonnet-4-6", 1080, 2400) == (706, 1568)
    assert model_effective_dims("claude-opus-4-7", 1080, 2400) == (922, 2048)
    assert model_effective_dims("gpt-5.5", 1080, 2400) == (922, 2048)
    assert model_effective_dims("gemini-3.5-flash-low", 1080, 2400) == (922, 2048)
    # Unknown Anthropic id is treated as STANDARD (never assume high-res).
    assert model_effective_dims("claude-future-9", 1080, 2400) == (706, 1568)


def test_claude_5_models_use_high_resolution_budget():
    for model in (
        "claude-fable-5-1",
        "claude-opus-5",
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-fable-5",
    ):
        assert model_effective_dims(model, 1080, 2400) == (922, 2048)


def test_sonnet_5_grounds_on_a_1568_long_edge():
    assert model_effective_dims("claude-sonnet-5", 1080, 2400) == (706, 1568)
    assert model_effective_dims("claude-sonnet-5", 2560, 1600) == (1568, 980)
    assert model_effective_dims("claude-sonnet-5", 720, 1280) == (720, 1280)
    assert VisionResizePolicy(["claude-opus-5-5", "claude-sonnet-5"]).effective_dims(
        1080, 2400
    ) == (706, 1568)


def test_unregistered_mythos_model_retains_high_resolution_budget():
    assert model_effective_dims("claude-mythos-5", 1080, 2400) == (922, 2048)


def test_policy_single_model():
    assert VisionResizePolicy(["claude-sonnet-4-6"]).effective_dims(1080, 2400) == (
        706,
        1568,
    )
    assert VisionResizePolicy(["gpt-5.5"]).effective_dims(1080, 2400) == (922, 2048)


def test_policy_composite_takes_smallest():
    # Mixed manager/executor models → declare the smallest so both are correct.
    policy = VisionResizePolicy(["claude-opus-4-7", "claude-sonnet-4-6"])
    assert policy.effective_dims(1080, 2400) == (706, 1568)
    # All high-res / non-downsizing → stays at 2048.
    assert VisionResizePolicy(["claude-opus-4-7", "gpt-5.5"]).effective_dims(1080, 2400) == (
        922,
        2048,
    )


def test_policy_empty_defaults_to_2048_fit():
    assert VisionResizePolicy([]).effective_dims(1080, 2400) == (922, 2048)


def test_policy_max_side_cap_shrinks_further():
    policy = VisionResizePolicy(["claude-sonnet-4-6"], max_side_cap=1280)
    w, h = policy.effective_dims(1080, 2400)
    assert max(w, h) == 1280


def test_policy_from_llms_reads_model_attr():
    llms = [
        SimpleNamespace(model="claude-sonnet-4-6"),
        SimpleNamespace(model="gpt-5.5"),
    ]
    policy = VisionResizePolicy.from_llms(llms)
    # min across sonnet(1568) and gpt(2048) → 1568.
    assert policy.effective_dims(1080, 2400) == (706, 1568)
