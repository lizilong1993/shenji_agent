"""Patch for training/selfplay.py to accept ``--ablation-mode`` argument.

This module adds ablation-mode support to the self-play evaluation pipeline
without modifying ``training/selfplay.py`` directly.  It works by:

1. Adding ``--ablation-mode`` to the argument parser.
2. Extending ``_build_participant_spec`` to carry the ``ablation_mode`` field.
3. Monkey-patching ``Agent.setup`` so that the ablation mode is forwarded to
   ``LearnedPolicy.configure`` via the setup_info dict.

Usage
-----
Import and call ``apply_selfplay_ablation_patch()`` *before* invoking
``training.selfplay.main()``::

    import ablation_selfplay_patch
    ablation_selfplay_patch.apply_selfplay_ablation_patch()

    from training.selfplay import main
    main()

Or run as a standalone entry-point::

    python -m ablation_selfplay_patch --config configs/selfplay_ablation.yaml \\
        --candidate ai/weights --incumbent ai/weights_incumbent_rule \\
        --ablation-mode no_intent
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Callable, Dict, Optional
from unittest.mock import patch

log = logging.getLogger(__name__)

# Valid ablation modes (must match ABLATION_MODES in ablation_policy_patch.py)
VALID_ABLATION_MODES = ("full", "no_intent", "no_situation", "rule_only")


# ---------------------------------------------------------------------------
# Patched _build_participant_spec that carries ablation_mode
# ---------------------------------------------------------------------------

_original_build_participant_spec: Optional[Callable[..., Dict[str, Any]]] = None


def _patched_build_participant_spec(
    artifacts: str,
    profile: str,
    fallback_commit_sha: str,
    user_name: str,
    *,
    ablation_mode: str = "full",
) -> Dict[str, Any]:
    """Wrapper around the original ``_build_participant_spec``.

    Adds an ``ablation_mode`` field to the returned spec dict so that
    downstream agent setup can forward it to ``LearnedPolicy.configure``.
    """
    assert _original_build_participant_spec is not None, (
        "Original _build_participant_spec not captured; call apply_selfplay_ablation_patch() first"
    )
    spec = _original_build_participant_spec(artifacts, profile, fallback_commit_sha, user_name)
    spec["ablation_mode"] = ablation_mode
    return spec


# ---------------------------------------------------------------------------
# Patched Agent.setup that forwards ablation_mode to policy.configure
# ---------------------------------------------------------------------------

_original_agent_setup: Optional[Callable[..., None]] = None


def _patched_agent_setup(self: Any, setup_info: Dict[str, Any]) -> None:
    """Wrapper around ``Agent.setup`` that extracts ``ablation_mode``.

    The ablation_mode is extracted from setup_info and passed through to
    ``LearnedPolicy.configure`` as a keyword argument.  If the policy has been
    patched with ``ablation_policy_patch``, the configure method will accept and
    honour the parameter.
    """
    ablation_mode = setup_info.pop("ablation_mode", "full")

    # Call the original Agent.setup (which calls policy.configure internally)
    assert _original_agent_setup is not None
    _original_agent_setup(self, setup_info)

    # If ablation_mode is not "full", we need to re-configure the policy with
    # the ablation mode applied.  The original setup already called
    # policy.configure(faction, artifacts_dir, profile), so we call it again
    # with the ablation_mode parameter.
    if ablation_mode != "full":
        policy_configure = getattr(self.policy, "configure", None)
        if policy_configure is not None:
            try:
                policy_configure(
                    faction=self.faction,
                    artifact_dir=self.policy_artifacts_dir,
                    policy_profile=self.policy_profile,
                    ablation_mode=ablation_mode,
                )
                log.info(
                    "Agent.setup [ABLATION]: re-configured policy with ablation_mode=%s for seat=%s",
                    ablation_mode,
                    self.seat,
                )
            except TypeError:
                # policy.configure hasn't been patched with ablation support;
                # fall back to manual model disabling
                log.warning(
                    "Agent.setup [ABLATION]: policy.configure does not accept ablation_mode; "
                    "applying manual model disabling for mode=%s",
                    ablation_mode,
                )
                _apply_ablation_manual(self.policy, ablation_mode)


def _apply_ablation_manual(policy: Any, ablation_mode: str) -> None:
    """Manually disable models on a policy when the configure patch is absent.

    This is a safety net in case ``ablation_policy_patch`` was not applied.
    """
    from ablation_policy_patch import ABLATION_MODES

    mode_mask = ABLATION_MODES.get(ablation_mode)
    if mode_mask is None:
        log.error("Unknown ablation_mode %r; skipping manual disable", ablation_mode)
        return

    disabled: list[str] = []
    if not mode_mask["intent"] and policy.intent_model is not None:
        policy.intent_model = None
        disabled.append("intent")
    if not mode_mask["situation"] and policy.situation_model is not None:
        policy.situation_model = None
        disabled.append("situation")
    if not mode_mask["tactical"] and policy.tactical_model is not None:
        policy.tactical_model = None
        disabled.append("tactical")
    if not mode_mask["objective"]:
        policy.objective_weight = 0.0
        disabled.append("objective")

    policy.ablation_mode = ablation_mode
    log.info(
        "Manual ablation applied: mode=%s disabled=%s",
        ablation_mode,
        disabled or "none",
    )


# ---------------------------------------------------------------------------
# Patched main() with --ablation-mode argument
# ---------------------------------------------------------------------------

_original_main: Optional[Callable[[], None]] = None


def _patched_main() -> None:
    """Replacement for ``training.selfplay.main()``.

    Adds ``--ablation-mode`` to the argument parser and threads the value
    through to participant specs and agent setup.
    """
    import training.selfplay as selfplay_mod

    # Capture the original argparse.ArgumentParser.__init__ so we can inject
    # our additional argument after the parser is created.
    _orig_parser_init = argparse.ArgumentParser.__init__

    _ablation_mode_captured: Dict[str, str] = {}

    class _AblationArgumentParser(argparse.ArgumentParser):
        """ArgumentParser subclass that auto-injects --ablation-mode."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.add_argument(
                "--ablation-mode",
                default="full",
                choices=list(VALID_ABLATION_MODES),
                help=(
                    "Ablation mode for the candidate agent. "
                    "'full' = all models enabled; "
                    "'no_intent' = intent model disabled; "
                    "'no_situation' = situation model disabled; "
                    "'rule_only' = all learned models disabled."
                ),
            )

        def parse_args(self, args=None, namespace=None):  # type: ignore[override]
            parsed = super().parse_args(args, namespace)
            _ablation_mode_captured["mode"] = getattr(parsed, "ablation_mode", "full")
            return parsed

    # Patch the parser class used in selfplay.main
    argparse.ArgumentParser = _AblationArgumentParser  # type: ignore[misc]

    # Also patch _build_participant_spec to inject ablation_mode
    original_bps = selfplay_mod._build_participant_spec

    def _bps_with_ablation(artifacts, profile, fallback_commit_sha, user_name):  # type: ignore[no-untyped-def]
        spec = original_bps(artifacts, profile, fallback_commit_sha, user_name)
        spec["ablation_mode"] = _ablation_mode_captured.get("mode", "full")
        return spec

    selfplay_mod._build_participant_spec = _bps_with_ablation  # type: ignore[assignment]

    # Patch _run_single to forward ablation_mode to agent setup
    original_run_single = selfplay_mod._run_single

    def _run_single_with_ablation(*args, **kwargs):  # type: ignore[no-untyped-def]
        """Wraps _run_single to pass ablation_mode through setup_info."""
        # _run_single signature:
        #   (scenario_path, map_id, red_profile, blue_profile, max_steps, match_id=None)
        # red_profile and blue_profile are participant spec dicts
        # We inject ablation_mode into the profile dicts so Agent.setup sees it
        if len(args) >= 4:
            red_prof = dict(args[2])
            blue_prof = dict(args[3])
            red_prof["ablation_mode"] = red_prof.get("ablation_mode", _ablation_mode_captured.get("mode", "full"))
            blue_prof["ablation_mode"] = blue_prof.get("ablation_mode", "full")
            args = (args[0], args[1], red_prof, blue_prof) + args[4:]
        return original_run_single(*args, **kwargs)

    selfplay_mod._run_single = _run_single_with_ablation  # type: ignore[assignment]

    try:
        assert _original_main is not None
        _original_main()
    finally:
        # Restore originals
        argparse.ArgumentParser = _orig_parser_init.__self__  # type: ignore[assignment]
        selfplay_mod._build_participant_spec = original_bps  # type: ignore[assignment]
        selfplay_mod._run_single = original_run_single  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Patch installer
# ---------------------------------------------------------------------------

def apply_selfplay_ablation_patch() -> None:
    """Apply all patches needed for ``--ablation-mode`` support in selfplay.

    This patches:
    - ``training.selfplay._build_participant_spec`` to carry ablation_mode
    - ``training.selfplay.main`` to accept ``--ablation-mode`` argument
    - ``ai.agent.Agent.setup`` to forward ablation_mode to policy.configure

    It is recommended to also call ``ablation_policy_patch.apply_ablation_patch()``
    so that ``LearnedPolicy.configure`` natively accepts the ``ablation_mode``
    keyword argument.
    """
    global _original_build_participant_spec, _original_agent_setup, _original_main

    import training.selfplay as selfplay_mod
    from ai.agent import Agent

    # Capture originals
    _original_build_participant_spec = selfplay_mod._build_participant_spec
    _original_main = selfplay_mod.main

    _original_agent_setup = Agent.setup

    # Apply patches
    selfplay_mod._build_participant_spec = _patched_build_participant_spec  # type: ignore[assignment]
    selfplay_mod.main = _patched_main  # type: ignore[assignment]
    Agent.setup = _patched_agent_setup  # type: ignore[assignment]

    log.info("ablation_selfplay_patch: all patches applied (--ablation-mode support enabled)")


# ---------------------------------------------------------------------------
# Standalone entry-point: run selfplay with ablation mode from CLI
# ---------------------------------------------------------------------------

def run_selfplay_with_ablation() -> None:
    """Standalone entry-point: parse args, apply patches, and run selfplay."""
    # Pre-parse just --ablation-mode so we can apply patches before main()
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--ablation-mode", default="full", choices=list(VALID_ABLATION_MODES))
    pre_args, _ = pre_parser.parse_known_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    log.info("Running selfplay with ablation_mode=%s", pre_args.ablation_mode)

    # Apply the policy-level patch first (so configure accepts ablation_mode)
    try:
        from ablation_policy_patch import apply_ablation_patch
        apply_ablation_patch()
    except ImportError:
        log.warning(
            "ablation_policy_patch not importable; will use manual model disabling"
        )

    apply_selfplay_ablation_patch()

    from training.selfplay import main
    main()


if __name__ == "__main__":
    run_selfplay_with_ablation()
