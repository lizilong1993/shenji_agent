"""Shenji dataset construction and validation toolkit."""

from .collection import CollectionDependencies, run_collection_job
from .coverage import build_coverage_report
from .jobs import CollectionJob, expand_jobs, load_collection_manifest
from shenji_dataset.labeling import derive_grade_a_intent_plan
from .normalizer import CANONICAL_TABLES, normalize_episode
from .ontology import load_ontology
from .recorder import EpisodeRecorder
from .release import load_release, validate_release, verify_release, write_release

__all__ = [
    "CANONICAL_TABLES",
    "CollectionDependencies",
    "CollectionJob",
    "EpisodeRecorder",
    "build_coverage_report",
    "derive_grade_a_intent_plan",
    "expand_jobs",
    "load_collection_manifest",
    "load_ontology",
    "load_release",
    "normalize_episode",
    "run_collection_job",
    "validate_release",
    "verify_release",
    "write_release",
]

__version__ = "1.0.0"
