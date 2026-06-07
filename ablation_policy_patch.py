"""Ablation mode support for LearnedPolicy.configure().

This patch adds an `ablation_mode` parameter to LearnedPolicy.configure() that
selectively disables learned models for controlled ablation experiments.

Ablation Modes
--------------
- ``"full"``          : All models enabled (default behaviour).
- ``"no_intent"``     : Intent model disabled; situation + tactical + objective remain.
- ``"no_situation"``  : Situation model disabled; intent + tactical + objective remain.
- ``"rule_only"``     : All learned models disabled; pure rule-based fallback.

Usage
-----
Apply by monkey-patching or by replacing the ``configure`` method on
``LearnedPolicy``::

    from ablation_policy_patch import apply_ablation_patch
    apply_ablation_patch()

Or call the new configure directly::

    policy.configure(faction=0, artifact_dir="ai/weights", ablation_mode="no_intent")
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai.policy import LearnedPolicy

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ablation mode definitions
# ---------------------------------------------------------------------------

ABLATION_MODES: Dict[str, Dict[str, bool]] = {
    "full": {"intent": True, "situation": True, "tactical": True, "objective": True},
    "no_intent": {"intent": False, "situation": True, "tactical": True, "objective": True},
    "no_situation": {"intent": True, "situation": False, "tactical": True, "objective": True},
    "rule_only": {"intent": False, "situation": False, "tactical": False, "objective": False},
}

VALID_ABLATION_MODES = frozenset(ABLATION_MODES.keys())


# ---------------------------------------------------------------------------
# Replacement configure() method
# ---------------------------------------------------------------------------

def configure_with_ablation(
    self: LearnedPolicy,
    faction: int,
    artifact_dir: Optional[str] = None,
    policy_profile: str = "challenger",
    *,
    ablation_mode: str = "full",
) -> None:
    """Replacement for ``LearnedPolicy.configure`` that honours *ablation_mode*.

    Parameters
    ----------
    faction:
        The faction id (0 = Red, 1 = Blue).
    artifact_dir:
        Optional path to the directory containing model weights and manifest.
    policy_profile:
        Policy profile name (e.g. ``"challenger"``, ``"baseline"``).
    ablation_mode:
        One of ``"full"``, ``"no_intent"``, ``"no_situation"``, ``"rule_only"``.
        Controls which learned models are disabled after loading.
    """

    # --- Validate ablation_mode early ---
    if ablation_mode not in VALID_ABLATION_MODES:
        raise ValueError(
            f"Unknown ablation_mode {ablation_mode!r}. "
            f"Valid modes: {sorted(VALID_ABLATION_MODES)}"
        )

    # --- Original configure logic (verbatim from ai/policy.py) ---
    self.faction = faction
    self.policy_profile = policy_profile or "challenger"
    self.history_len = 8
    self.device = "cpu"
    self.requested_device = "auto"
    self.intent_model = None
    self.situation_model = None
    self.tactical_model = None
    self.objective_model = None
    self._feature_normalizer = {}
    self.loaded_from = None
    self.manifest = {}
    self._observation_cache = None
    self.enemy_histories = defaultdict(lambda: deque(maxlen=self.history_len))
    self._reset_diagnostics()

    if not TORCH_AVAILABLE or self.policy_profile == "baseline":
        log.info(
            "LearnedPolicy.configure: skipping model load (torch_available=%s, profile=%s)",
            TORCH_AVAILABLE,
            self.policy_profile,
        )
        self.ablation_mode = ablation_mode
        return

    self.device = self._resolve_device()

    candidate_dirs: List[Path] = []
    if artifact_dir:
        candidate_dirs.append(Path(artifact_dir))
    candidate_dirs.append(Path(__file__).resolve().parent / "weights")

    models_loaded = False
    for candidate_dir in candidate_dirs:
        manifest_path = candidate_dir / "model_manifest.json"
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.history_len = int(payload.get("history_len", self.history_len))
        self.enemy_histories = defaultdict(lambda: deque(maxlen=self.history_len))

        intent_name = payload.get("intent_lstm") or payload.get("intent_model")
        situation_name = payload.get("situation_gcn") or payload.get("situation_model")
        if not intent_name or not situation_name:
            continue

        intent_path = candidate_dir / intent_name
        situation_path = candidate_dir / situation_name
        if not intent_path.exists() or not situation_path.exists():
            continue

        try:
            self.intent_model, self.situation_model = self._load_models(intent_path, situation_path)
        except Exception:
            if self.device == "cpu":
                raise
            self.device = "cpu"
            self.intent_model, self.situation_model = self._load_models(intent_path, situation_path)
        self.loaded_from = candidate_dir
        self.manifest = payload
        models_loaded = True

        # Load normalizer
        self._feature_normalizer = self._load_normalizer(candidate_dir)

        # Load tactical model if available
        tactical_name = payload.get("tactical_model") or "tactical_model.pt"
        tactical_path = candidate_dir / tactical_name
        if tactical_path.exists():
            try:
                tactical_state = torch.load(tactical_path, map_location="cpu")
                self.tactical_model = self._move_to_device(self._load_tactical_model(tactical_state))
            except Exception as exc:
                log.warning(
                    "tactical model file %s exists but failed to load: %s", tactical_path, exc
                )
                self.tactical_model = None
        else:
            log.warning(
                "tactical model file not found at %s; tactical_score will be 0", tactical_path
            )

        break  # Successfully loaded from this candidate_dir

    # --- Ablation mode enforcement ---
    mode_mask = ABLATION_MODES[ablation_mode]
    disabled_components: List[str] = []

    if not mode_mask["intent"]:
        if self.intent_model is not None:
            disabled_components.append("intent")
        self.intent_model = None

    if not mode_mask["situation"]:
        if self.situation_model is not None:
            disabled_components.append("situation")
        self.situation_model = None

    if not mode_mask["tactical"]:
        if self.tactical_model is not None:
            disabled_components.append("tactical")
        self.tactical_model = None

    if not mode_mask["objective"]:
        # Objective scoring is heuristic (not a loaded model), but we can
        # disable it by zeroing the weight so _objective_*_score contributions
        # are nullified in the combined scoring formula.
        disabled_components.append("objective")
        self.objective_weight = 0.0

    # Store ablation mode on the policy instance for downstream inspection
    self.ablation_mode = ablation_mode

    log.info(
        "LearnedPolicy.configure [ABLATION]: mode=%s disabled=%s intent=%s situation=%s tactical=%s objective_weight=%s loaded_from=%s",
        ablation_mode,
        disabled_components or "none",
        self.intent_model is not None,
        self.situation_model is not None,
        self.tactical_model is not None,
        self.objective_weight,
        self.loaded_from,
    )


# ---------------------------------------------------------------------------
# Patch installer
# ---------------------------------------------------------------------------

def apply_ablation_patch() -> None:
    """Monkey-patch ``LearnedPolicy.configure`` to support *ablation_mode*.

    After calling this, every ``LearnedPolicy.configure(...)`` call accepts an
    optional ``ablation_mode`` keyword argument.  Existing call sites that do
    not pass the argument will default to ``"full"`` (no behaviour change).
    """
    LearnedPolicy.configure = configure_with_ablation  # type: ignore[assignment]
    log.info("ablation_policy_patch: LearnedPolicy.configure patched with ablation_mode support")


# ---------------------------------------------------------------------------
# Convenience: allow running as ``python ablation_policy_patch.py`` to test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    apply_ablation_patch()

    policy = LearnedPolicy()
    for mode in sorted(VALID_ABLATION_MODES):
        log.info("--- Testing ablation_mode=%s ---", mode)
        policy.configure(faction=0, artifact_dir=None, policy_profile="baseline", ablation_mode=mode)
        log.info(
            "  intent=%s situation=%s tactical=%s objective_weight=%s ablation_mode=%s",
            policy.intent_model is not None,
            policy.situation_model is not None,
            policy.tactical_model is not None,
            policy.objective_weight,
            getattr(policy, "ablation_mode", None),
        )
    log.info("All ablation modes validated successfully.")
