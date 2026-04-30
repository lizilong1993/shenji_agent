from __future__ import annotations

import json
import pickle
import random
import subprocess
import sys
import tempfile
import types
import unittest
from csv import DictReader
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_fake_torch() -> None:
    if "torch" in sys.modules:
        return
    torch_module = types.ModuleType("torch")
    torch_module.float32 = "float32"
    torch_module.long = "long"
    torch_module.tensor = lambda *args, **kwargs: None
    torch_module.manual_seed = lambda seed: None
    torch_module.cuda = types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda seed: None)
    torch_module.mm = lambda left, right: left
    torch_module.cat = lambda tensors, dim=0: tensors[0] if tensors else None
    torch_module.sigmoid = lambda value: value
    torch_module.softmax = lambda value, dim=-1: value
    torch_module.load = lambda *args, **kwargs: {}
    utils_module = types.ModuleType("torch.utils")
    data_module = types.ModuleType("torch.utils.data")
    nn_module = types.ModuleType("torch.nn")
    functional_module = types.ModuleType("torch.nn.functional")

    class Dataset:
        pass

    class Module:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def eval(self):
            return self

        def load_state_dict(self, state: object) -> None:
            return None

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, *args, **kwargs):
            return args[0] if args else None

    class LSTM(Module):
        def forward(self, x, *args, **kwargs):
            return x, None

    class MultiheadAttention(Module):
        def forward(self, query, key, value, *args, **kwargs):
            return query, None

    class Linear(Module):
        pass

    class LayerNorm(Module):
        pass

    class ReLU(Module):
        pass

    class Dropout(Module):
        pass

    class Sequential(Module):
        def __init__(self, *layers) -> None:
            self.layers = layers

        def forward(self, x):
            value = x
            for layer in self.layers:
                value = layer(value)
            return value

    data_module.Dataset = Dataset
    utils_module.data = data_module
    torch_module.utils = utils_module
    nn_module.Module = Module
    nn_module.LSTM = LSTM
    nn_module.MultiheadAttention = MultiheadAttention
    nn_module.Linear = Linear
    nn_module.LayerNorm = LayerNorm
    nn_module.ReLU = ReLU
    nn_module.Dropout = Dropout
    nn_module.Sequential = Sequential
    torch_module.nn = nn_module
    functional_module.relu = lambda value: value
    sys.modules["torch"] = torch_module
    sys.modules["torch.utils"] = utils_module
    sys.modules["torch.utils.data"] = data_module
    sys.modules["torch.nn"] = nn_module
    sys.modules["torch.nn.functional"] = functional_module


def _install_fake_numpy() -> None:
    if "numpy" in sys.modules:
        return

    class FakeArray:
        def __init__(self, data: object):
            self._data = _materialize_data(data)

        @property
        def ndim(self) -> int:
            return len(self.shape)

        @property
        def shape(self) -> tuple[int, ...]:
            return _shape_of(self._data)

        @property
        def T(self) -> "FakeArray":
            if self.ndim != 2:
                return FakeArray(self._data)
            rows, cols = self.shape
            return FakeArray([[self._data[row][col] for row in range(rows)] for col in range(cols)])

        def astype(self, dtype: object) -> "FakeArray":
            return FakeArray(self._data)

        def tolist(self) -> object:
            return _clone_data(self._data)

        def mean(self, axis: int | None = None) -> object:
            return _mean(self._data, axis=axis)

        def std(self, axis: int | None = None) -> object:
            return _std(self._data, axis=axis)

        def __iter__(self):
            for item in self._data:
                yield item

        def __len__(self) -> int:
            return len(self._data)

        def __getitem__(self, key: object) -> object:
            if isinstance(key, tuple):
                row_key, col_key = key
                rows = _apply_selector(self._data, row_key)
                if isinstance(rows, list) and rows and isinstance(rows[0], list):
                    selected = [_apply_selector(row, col_key) for row in rows]
                    return FakeArray(selected)
                return _apply_selector(rows, col_key)
            value = self._data[key]
            if isinstance(value, list):
                return FakeArray(value)
            return value

        def __setitem__(self, key: object, value: object) -> None:
            if isinstance(key, tuple):
                row_idx, col_idx = key
                if isinstance(col_idx, list):
                    for idx in col_idx:
                        self._data[row_idx][idx] = value
                    return
                self._data[row_idx][col_idx] = value
                return
            self._data[key] = value

        def __matmul__(self, other: object) -> "FakeArray":
            other_data = _coerce_data(other)
            left = self._data
            right = other_data
            rows = len(left)
            inner = len(left[0]) if rows else 0
            cols = len(right[0]) if right else 0
            return FakeArray(
                [
                    [
                        sum(float(left[row][k]) * float(right[k][col]) for k in range(inner))
                        for col in range(cols)
                    ]
                    for row in range(rows)
                ]
            )

    def _clone_data(data: object) -> object:
        if isinstance(data, list):
            return [_clone_data(item) for item in data]
        return data

    def _coerce_data(data: object) -> object:
        if isinstance(data, FakeArray):
            return data._data
        return data

    def _materialize_data(data: object) -> object:
        data = _coerce_data(data)
        if isinstance(data, tuple):
            return [_materialize_data(item) for item in data]
        if isinstance(data, list):
            return [_materialize_data(item) for item in data]
        return data

    def _shape_of(data: object) -> tuple[int, ...]:
        if not isinstance(data, list):
            return ()
        if not data:
            return (0,)
        return (len(data),) + _shape_of(data[0])

    def _flatten(data: object) -> list[float]:
        if isinstance(data, FakeArray):
            data = data._data
        if isinstance(data, list):
            flat: list[float] = []
            for item in data:
                flat.extend(_flatten(item))
            return flat
        return [float(data)]

    def _mean(data: object, axis: int | None = None) -> object:
        data = _coerce_data(data)
        if axis is None:
            values = _flatten(data)
            return sum(values) / len(values) if values else 0.0
        if axis == 0:
            rows = data if isinstance(data, list) else []
            if not rows:
                return FakeArray([])
            cols = len(rows[0])
            return FakeArray(
                [
                    sum(float(rows[row][col]) for row in range(len(rows))) / len(rows)
                    for col in range(cols)
                ]
            )
        if axis == 1:
            rows = data if isinstance(data, list) else []
            return FakeArray(
                [
                    sum(float(value) for value in row) / len(row) if row else 0.0
                    for row in rows
                ]
            )
        raise ValueError(f"unsupported axis={axis}")

    def _std(data: object, axis: int | None = None) -> object:
        if axis is None:
            values = _flatten(data)
            if not values:
                return 0.0
            mean = sum(values) / len(values)
            return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
        means = _coerce_data(_mean(data, axis=axis))
        rows = _coerce_data(data)
        if axis == 0:
            cols = len(means)
            count = len(rows)
            return FakeArray(
                [
                    (
                        sum((float(rows[row][col]) - float(means[col])) ** 2 for row in range(count)) / count
                    )
                    ** 0.5
                    for col in range(cols)
                ]
            )
        if axis == 1:
            return FakeArray(
                [
                    (
                        sum((float(value) - float(means[row_idx])) ** 2 for value in row) / len(row)
                    )
                    ** 0.5
                    if row
                    else 0.0
                    for row_idx, row in enumerate(rows)
                ]
            )
        raise ValueError(f"unsupported axis={axis}")

    def _apply_selector(values: object, selector: object) -> object:
        if isinstance(selector, slice):
            return values[selector]
        if isinstance(selector, list):
            return [values[idx] for idx in selector]
        return values[selector]

    def _to_2d_list(data: object) -> list[list[float]]:
        rows = _coerce_data(data)
        return [[float(value) for value in row] for row in rows]

    class FakeRandom:
        def __init__(self) -> None:
            self._rng = random.Random()

        def seed(self, seed: int) -> None:
            self._rng.seed(seed)

        def choice(self, values: object, size: int | None = None, replace: bool = True) -> object:
            population = _flatten(values)
            if size is None:
                return self._rng.choice(population)
            if replace:
                return FakeArray([self._rng.choice(population) for _ in range(size)])
            return FakeArray(self._rng.sample(population, k=size))

        def default_rng(self, seed: int) -> object:
            rng = random.Random(seed)

            class Generator:
                def choice(self, population_size: int, size: int, replace: bool = False) -> FakeArray:
                    population = list(range(population_size))
                    if replace:
                        return FakeArray([rng.choice(population) for _ in range(size)])
                    return FakeArray(rng.sample(population, k=size))

            return Generator()

    numpy_module = types.ModuleType("numpy")
    numpy_module.ndarray = FakeArray
    numpy_module.float32 = "float32"
    numpy_module.int64 = "int64"
    numpy_module.random = FakeRandom()
    numpy_module.array = lambda data, dtype=None, copy=False: FakeArray(data)
    numpy_module.zeros = lambda shape, dtype=None: FakeArray(_zeros(shape))
    numpy_module.zeros_like = lambda data: FakeArray(_zeros(_shape_of(_coerce_data(data))))
    numpy_module.eye = lambda n, dtype=None: FakeArray(
        [[1.0 if row == col else 0.0 for col in range(n)] for row in range(n)]
    )
    numpy_module.stack = lambda items, axis=0: FakeArray(_stack(items, axis=axis))
    numpy_module.pad = lambda data, pad_width: FakeArray(_pad(_coerce_data(data), pad_width))
    numpy_module.mean = lambda data, axis=None: _mean(data, axis=axis)
    numpy_module.std = lambda data, axis=None: _std(data, axis=axis)
    numpy_module.quantile = lambda data, q: _quantile(data, q)
    numpy_module.argmax = lambda values: max(range(len(values)), key=lambda idx: values[idx])
    numpy_module.argsort = lambda values: FakeArray(
        sorted(range(len(_flatten(values))), key=lambda idx: _flatten(values)[idx])
    )
    numpy_module.fill_diagonal = lambda matrix, value: _fill_diagonal(matrix, value)
    numpy_module.load = lambda path: {"data": FakeArray([])}
    sys.modules["numpy"] = numpy_module

    def _zeros(shape: object) -> object:
        if isinstance(shape, int):
            return [0.0 for _ in range(shape)]
        if len(shape) == 1:
            return [0.0 for _ in range(shape[0])]
        return [_zeros(shape[1:]) for _ in range(shape[0])]

    def _stack(items: object, axis: int = 0) -> object:
        rows = [_coerce_data(item) for item in items]
        if axis == 0:
            return rows
        if axis == 1:
            length = len(rows[0]) if rows else 0
            return [[rows[col][row] for col in range(len(rows))] for row in range(length)]
        raise ValueError(f"unsupported axis={axis}")

    def _pad(data: object, pad_width: tuple[tuple[int, int], tuple[int, int]]) -> object:
        rows = _to_2d_list(data)
        (top, bottom), (left, right) = pad_width
        width = len(rows[0]) + left + right if rows else left + right
        padded = [[0.0 for _ in range(width)] for _ in range(top)]
        for row in rows:
            padded.append(([0.0] * left) + [float(value) for value in row] + ([0.0] * right))
        padded.extend([[0.0 for _ in range(width)] for _ in range(bottom)])
        return padded

    def _quantile(data: object, q: float) -> float:
        values = sorted(_flatten(data))
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        position = q * (len(values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    def _fill_diagonal(matrix: object, value: float) -> None:
        data = _coerce_data(matrix)
        size = min(len(data), len(data[0]) if data else 0)
        for idx in range(size):
            data[idx][idx] = value


def _install_fake_pandas() -> None:
    if "pandas" in sys.modules:
        return

    class FakeSeries:
        def __init__(self, values: list[object]):
            self._values = list(values)

        def notna(self) -> "FakeSeries":
            return FakeSeries([value is not None and value != "" for value in self._values])

        def mean(self) -> float:
            if not self._values:
                return 0.0
            return sum(float(bool(value)) for value in self._values) / len(self._values)

        def fillna(self, value: object) -> "FakeSeries":
            return FakeSeries([value if item is None or item == "" else item for item in self._values])

        def to_numpy(self, dtype: object = None):
            _install_fake_numpy()
            import numpy as np

            return np.array(
                [
                    float(item) if isinstance(item, (int, float)) or str(item).strip() not in {"", "nan"} else 0.0
                    for item in self._values
                ],
                dtype=dtype,
            )

        def map(self, func):
            return FakeSeries([func(item) for item in self._values])

    class FakeDataFrame:
        def __init__(self, rows: list[dict[str, object]], columns: list[str]):
            self._rows = rows
            self.columns = columns

        def __len__(self) -> int:
            return len(self._rows)

        def __getitem__(self, column: str) -> FakeSeries:
            return FakeSeries([row.get(column) for row in self._rows])

    def read_csv(path: Path, encoding: str = "utf-8", nrows: int | None = None, **kwargs) -> FakeDataFrame:
        with Path(path).open("r", encoding=encoding, newline="") as fh:
            reader = DictReader(fh)
            rows = list(reader)
        if nrows is not None:
            rows = rows[:nrows]
        columns = list(rows[0].keys()) if rows else []
        return FakeDataFrame(rows, columns)

    def to_numeric(values: object, errors: str = "raise") -> FakeSeries:
        if isinstance(values, FakeSeries):
            raw_values = values._values
        else:
            raw_values = list(values)
        converted: list[object] = []
        for value in raw_values:
            try:
                converted.append(float(value))
            except Exception:
                converted.append(None if errors == "coerce" else value)
        return FakeSeries(converted)

    pandas_module = types.ModuleType("pandas")
    pandas_module.DataFrame = FakeDataFrame
    pandas_module.Series = FakeSeries
    pandas_module.read_csv = read_csv
    pandas_module.to_numeric = to_numeric
    sys.modules["pandas"] = pandas_module


_install_fake_numpy()
_install_fake_torch()
_install_fake_pandas()


class MainlineContractTest(unittest.TestCase):
    def test_training_modules_do_not_eager_import_torch(self) -> None:
        import builtins
        import importlib

        module_names = [
            "training.common",
            "training.evaluate",
            "training.selfplay",
            "torch",
            "torch.utils",
            "torch.utils.data",
            "torch.nn",
            "torch.nn.functional",
        ]
        saved_modules = {name: sys.modules.pop(name, None) for name in module_names}

        def restore_modules() -> None:
            for name, module in saved_modules.items():
                if module is not None:
                    sys.modules[name] = module

        self.addCleanup(restore_modules)
        self.addCleanup(_install_fake_torch)

        real_import = builtins.__import__

        def guarded_import(name: str, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError(f"unexpected eager torch import: {name}")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            importlib.import_module("training.evaluate")
            importlib.import_module("training.selfplay")

    def test_selfplay_builds_ordered_unique_first_map_list(self) -> None:
        from training.selfplay import _build_paired_map_list, _collect_scenario_candidates

        with tempfile.TemporaryDirectory() as tmp:
            scenario_root = Path(tmp)
            bucket_lookup = {}
            for idx in range(1, 121):
                (scenario_root / f"scenario_{idx}.json").write_text("{}", encoding="utf-8")
                bucket_lookup[idx] = "complex" if idx % 3 == 0 else ("medium" if idx % 3 == 1 else "simple")
            candidates = _collect_scenario_candidates(scenario_root, bucket_lookup)

        random.seed(42)
        payload = _build_paired_map_list(candidates, 100)
        items = payload["items"]

        self.assertEqual(payload["sampling_protocol"], "unique_first_then_replacement")
        self.assertEqual(payload["candidate_pool_size"], 120)
        self.assertFalse(payload["duplicates_used"])
        self.assertEqual(len(items), 100)
        self.assertEqual([item["order_index"] for item in items], list(range(100)))
        self.assertEqual(len({item["scenario"] for item in items}), 100)
        self.assertTrue(
            all({"scenario", "map_id", "bucket", "order_index"}.issubset(item.keys()) for item in items)
        )

    def test_selfplay_backfills_duplicates_only_after_unique_pool(self) -> None:
        from training.selfplay import _build_paired_map_list, _collect_scenario_candidates

        with tempfile.TemporaryDirectory() as tmp:
            scenario_root = Path(tmp)
            bucket_lookup = {1: "simple", 2: "medium", 3: "complex"}
            for idx in bucket_lookup:
                (scenario_root / f"scenario_{idx}.json").write_text("{}", encoding="utf-8")
            candidates = _collect_scenario_candidates(scenario_root, bucket_lookup)

        random.seed(7)
        payload = _build_paired_map_list(candidates, 10)
        items = payload["items"]

        self.assertTrue(payload["duplicates_used"])
        self.assertEqual(payload["candidate_pool_size"], 3)
        self.assertEqual(payload["unique_item_count_before_replacement"], 3)
        self.assertEqual(len(items), 10)
        self.assertEqual(len({item["scenario"] for item in items[:3]}), 3)
        self.assertEqual(
            {item["scenario"] for item in items},
            {"scenario_1.json", "scenario_2.json", "scenario_3.json"},
        )

    def test_selfplay_resolves_incumbent_profile_from_snapshot_role(self) -> None:
        from training.selfplay import _participant_payload, _resolve_incumbent_profile, _resolve_source_commit_sha

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            learned_dir = root / "learned"
            learned_dir.mkdir()
            (learned_dir / "model_manifest.json").write_text(
                json.dumps(
                    {
                        "snapshot_role": "accepted_learned_mainline",
                        "generated_at": "20260423_000000",
                        "source_commit_sha": "abc123",
                    }
                ),
                encoding="utf-8",
            )
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "model_manifest.json").write_text(
                json.dumps({"snapshot_role": "bootstrap_rule_baseline"}),
                encoding="utf-8",
            )

            self.assertEqual(_resolve_incumbent_profile(str(learned_dir), None, "baseline"), "incumbent")
            self.assertEqual(_resolve_incumbent_profile(str(baseline_dir), None, "baseline"), "baseline")
            self.assertEqual(_resolve_source_commit_sha(str(learned_dir), "fallback"), "abc123")
            self.assertEqual(_resolve_source_commit_sha(str(baseline_dir), "fallback"), "fallback")
            participant = _participant_payload(str(learned_dir), "incumbent", "abc123", "git:abc123")
            self.assertEqual(participant["profile"], "incumbent")
            self.assertEqual(participant["snapshot_role"], "accepted_learned_mainline")
            self.assertEqual(participant["source_commit_sha"], "abc123")
            self.assertEqual(participant["code_origin"], "git:abc123")
            self.assertTrue(participant["manifest_loaded"])
            self.assertTrue(str(participant["manifest_path"]).endswith("model_manifest.json"))

    def test_selfplay_main_passes_configured_incumbent_profile_as_explicit_request(self) -> None:
        from training import selfplay

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiments").mkdir()
            (root / "league").mkdir()
            config = {
                "seed": 42,
                "output_root": "experiments",
                "league_dir": "league",
                "paired_matches": 1,
                "max_steps": 10,
                "candidate_profile": "challenger",
                "incumbent_profile": "baseline",
                "promotion_gate": {
                    "overall_point_estimate_min": 0.5,
                    "overall_ci_lower_min": 0.4,
                    "complex_point_estimate_min": 0.5,
                    "complex_ci_lower_min": 0.4,
                    "consecutive_windows": 1,
                },
            }

            with (
                mock.patch.object(selfplay, "repo_root", return_value=root),
                mock.patch.object(selfplay, "load_config", return_value=config),
                mock.patch.object(selfplay, "_configure_torch_cpu_inference_threads", return_value=None),
                mock.patch.object(selfplay, "set_seed", return_value=None),
                mock.patch.object(selfplay, "git_head_sha", return_value="headsha"),
                mock.patch.object(selfplay, "_bootstrap_league", return_value={"champion": "ai/weights", "historical_mains": [], "exploiters": [], "specialists": [], "consecutive_windows": 0}),
                mock.patch.object(selfplay, "_resolve_incumbent_profile", return_value="baseline") as resolve_profile,
                mock.patch.object(selfplay, "_build_participant_spec", side_effect=RuntimeError("stop_after_profile_resolution")),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "training.selfplay",
                        "--config",
                        "ignored.yaml",
                        "--candidate",
                        "ai/weights",
                        "--incumbent",
                        "artifacts/best_agent/weights",
                    ],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop_after_profile_resolution"):
                    selfplay.main()

        resolve_profile.assert_called_once_with("artifacts/best_agent/weights", "baseline", "baseline")

    def test_selfplay_prefers_train_env_backend_over_mock(self) -> None:
        from training.selfplay import _resolve_train_env_class

        class TrainEnv:
            pass

        def fake_import(name: str) -> object:
            if name == "land_wargame_train_env":
                raise ImportError("linux alias missing")
            if name == "train_env":
                return types.SimpleNamespace(TrainEnv=TrainEnv)
            raise AssertionError(f"unexpected import: {name}")

        env_class, backend = _resolve_train_env_class(fake_import)
        self.assertIs(env_class, TrainEnv)
        self.assertEqual(backend, "train_env.TrainEnv")

    def test_selfplay_falls_back_to_mock_only_when_real_env_missing(self) -> None:
        from training.selfplay import _resolve_train_env_class

        class MockTrainEnv:
            pass

        def fake_import(name: str) -> object:
            if name in {"land_wargame_train_env", "train_env"}:
                raise ImportError("missing")
            if name == "mock_train_env":
                return types.SimpleNamespace(MockTrainEnv=MockTrainEnv)
            raise AssertionError(f"unexpected import: {name}")

        env_class, backend = _resolve_train_env_class(fake_import)
        self.assertIs(env_class, MockTrainEnv)
        self.assertEqual(backend, "mock_train_env.MockTrainEnv")

    def test_selfplay_can_load_agent_class_from_git_commit(self) -> None:
        from training.common import git_head_sha
        from training.selfplay import _load_agent_class_from_commit

        _saved_torch = {}
        for module_name in list(sys.modules):
            if module_name == "torch" or module_name.startswith("torch."):
                _saved_torch[module_name] = sys.modules.pop(module_name, None)
        _install_fake_torch()

        def _restore_real_torch() -> None:
            for module_name in list(sys.modules):
                if module_name == "torch" or module_name.startswith("torch."):
                    sys.modules.pop(module_name, None)
            for name, mod in _saved_torch.items():
                if mod is not None:
                    sys.modules[name] = mod

        self.addCleanup(_restore_real_torch)
        commit_sha = git_head_sha(REPO_ROOT)
        agent_class = _load_agent_class_from_commit(commit_sha)
        self.assertEqual(agent_class.__name__, "Agent")
        self.assertTrue(agent_class.__module__.startswith("_selfplay_snapshot_"))

    def _write_selfplay_case_fixture(self, root: Path, *, map_id: int, basic_data: dict[str, object]) -> Path:
        data_root = root / "land_wargame_sdk" / "Data" / "Data"
        scenario_root = data_root / "scenarios"
        map_root = data_root / "maps" / f"map_{map_id}"
        scenario_root.mkdir(parents=True, exist_ok=True)
        map_root.mkdir(parents=True, exist_ok=True)
        scenario_path = scenario_root / f"{map_id}000000000.json"
        scenario_path.write_text("{}", encoding="utf-8")
        (map_root / "basic.json").write_text(json.dumps(basic_data), encoding="utf-8")
        with (map_root / "cost.pickle").open("wb") as fh:
            pickle.dump({"ground": {}}, fh)
        return scenario_path

    def _make_map_record(self, *, map_id: int, rows: int, cols: int, bucket: str = "complex"):
        from training.maps import MapRecord

        return MapRecord(
            map_id=map_id,
            name=f"map_{map_id}",
            rows=rows,
            cols=cols,
            terrain_entropy=0.0,
            elevation_std=0.0,
            cover_density=0.0,
            road_sparsity=0.0,
            river_density=0.0,
            connectivity_penalty=0.0,
            bucket=bucket,
        )

    def test_infer_map_id_from_scenario_id_applies_explicit_scenario_overrides(self) -> None:
        from training.maps import infer_map_id_from_scenario_id

        map_ids = [21, 53, 94, 212, 221]

        self.assertEqual(infer_map_id_from_scenario_id("2120511121", map_ids), 21)
        self.assertEqual(infer_map_id_from_scenario_id("2120131194", map_ids), 94)
        self.assertEqual(infer_map_id_from_scenario_id("2120131294", map_ids), 94)

    def test_selfplay_collect_scenario_candidates_applies_conservative_middle_token_fallback(self) -> None:
        from training.selfplay import _collect_scenario_candidates

        with tempfile.TemporaryDirectory() as tmp:
            scenario_root = Path(tmp)
            (scenario_root / "2120531121.json").write_text(
                json.dumps(
                    {
                        "cities": [{"coord": 2450}, {"coord": 2749}, {"coord": 2546}, {"coord": 2248}, {"coord": 2754}],
                        "operators": [{"cur_hex": 751}, {"cur_hex": 1056}, {"cur_hex": 552}, {"cur_hex": 1058}],
                    }
                ),
                encoding="utf-8",
            )
            (scenario_root / "2120511121.json").write_text(
                json.dumps(
                    {
                        "cities": [{"coord": 5140}, {"coord": 4838}, {"coord": 4341}, {"coord": 5038}, {"coord": 5539}],
                        "operators": [
                            *({"cur_hex": coord} for coord in range(3700, 3737)),
                            *({"cur_hex": coord} for coord in range(5000, 5021)),
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (scenario_root / "2120131194.json").write_text(
                json.dumps(
                    {
                        "cities": [{"coord": 1227}],
                        "operators": [{"cur_hex": 7500}, {"cur_hex": 4900}],
                    }
                ),
                encoding="utf-8",
            )
            (scenario_root / "2120131294.json").write_text(
                json.dumps(
                    {
                        "cities": [{"coord": 1127}],
                        "operators": [{"cur_hex": 7600}, {"cur_hex": 4800}],
                    }
                ),
                encoding="utf-8",
            )
            (scenario_root / "2230531221.json").write_text(
                json.dumps(
                    {
                        "cities": [
                            {"coord": 5140},
                            {"coord": 4838},
                            {"coord": 5144},
                            {"coord": 4942},
                            {"coord": 4341},
                            {"coord": 5038},
                            {"coord": 5539},
                        ],
                        "operators": [
                            *({"cur_hex": coord} for coord in range(1000, 1040)),
                            *({"cur_hex": coord} for coord in range(7000, 7028)),
                        ],
                    }
                ),
                encoding="utf-8",
            )
            bucket_lookup = {21: "medium", 94: "simple", 212: "medium", 53: "complex", 221: "complex"}
            records = [
                self._make_map_record(map_id=21, rows=92, cols=77, bucket="medium"),
                self._make_map_record(map_id=94, rows=92, cols=77, bucket="simple"),
                self._make_map_record(map_id=212, rows=50, cols=50, bucket="medium"),
                self._make_map_record(map_id=53, rows=92, cols=77, bucket="complex"),
                self._make_map_record(map_id=221, rows=70, cols=60, bucket="complex"),
            ]

            candidates = _collect_scenario_candidates(scenario_root, bucket_lookup, records)

        resolved = {item["scenario"]: item["map_id"] for item in candidates}
        self.assertEqual(resolved["2120131194.json"], 94)
        self.assertEqual(resolved["2120131294.json"], 94)
        self.assertEqual(resolved["2120531121.json"], 53)
        self.assertEqual(resolved["2120511121.json"], 21)
        self.assertEqual(resolved["2230531221.json"], 53)

    def test_infer_map_id_from_scenario_with_fallback_requires_full_operator_recovery_when_cities_already_fit(self) -> None:
        from training.maps import infer_map_id_from_scenario_with_fallback

        records = [
            self._make_map_record(map_id=53, rows=92, cols=77, bucket="complex"),
            self._make_map_record(map_id=221, rows=70, cols=60, bucket="complex"),
        ]
        scenario_data = {
            "cities": [
                {"coord": 5140},
                {"coord": 4838},
                {"coord": 5144},
                {"coord": 4942},
                {"coord": 4341},
                {"coord": 5038},
                {"coord": 5539},
            ],
            "operators": [
                {"cur_hex": 1000},
                {"cur_hex": 1001},
                {"cur_hex": 7000},
                {"cur_hex": 7001},
                {"cur_hex": 9200},
            ],
        }

        resolved = infer_map_id_from_scenario_with_fallback("2230531221", scenario_data, records)

        self.assertEqual(resolved, 221)

    def test_selfplay_load_case_data_normalizes_legacy_basic_schema(self) -> None:
        from training import selfplay

        legacy_basic = {
            "map_id": 212,
            "map_data": [
                [
                    {
                        "nb": [1, -1, -1, -1, -1, 100],
                        "c": 0,
                        "r": 0,
                        "e": 60,
                        "cd": 2,
                        "rd": [0, 0, 0, 0, 0, 0],
                        "rv": [0, 0, 0, 0, 0, 0],
                    }
                ]
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario_path = self._write_selfplay_case_fixture(root, map_id=212, basic_data=legacy_basic)
            with mock.patch.object(selfplay, "repo_root", return_value=root):
                loaded = selfplay._load_case_data(scenario_path, 212)

        cell = loaded["basic_data"]["map_data"][0][0]
        self.assertEqual(cell["neighbors"], legacy_basic["map_data"][0][0]["nb"])
        self.assertEqual(cell["roads"], legacy_basic["map_data"][0][0]["rd"])
        self.assertEqual(cell["rivers"], legacy_basic["map_data"][0][0]["rv"])
        self.assertEqual(cell["cond"], legacy_basic["map_data"][0][0]["cd"])
        self.assertEqual(cell["elev"], legacy_basic["map_data"][0][0]["e"])
        self.assertEqual(cell["nb"], legacy_basic["map_data"][0][0]["nb"])
        self.assertEqual(cell["rd"], legacy_basic["map_data"][0][0]["rd"])
        self.assertEqual(cell["rv"], legacy_basic["map_data"][0][0]["rv"])

    def test_selfplay_load_case_data_rejects_invalid_legacy_basic_schema(self) -> None:
        from training import selfplay

        bad_basic = {
            "map_id": 212,
            "map_data": [
                [
                    {
                        "c": 0,
                        "r": 0,
                        "e": 60,
                        "cd": 2,
                        "rv": [0, 0, 0, 0, 0, 0],
                    }
                ]
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario_path = self._write_selfplay_case_fixture(root, map_id=212, basic_data=bad_basic)
            with mock.patch.object(selfplay, "repo_root", return_value=root):
                with self.assertRaisesRegex(ValueError, r"map_id=212.*basic\.json.*neighbors"):
                    selfplay._load_case_data(scenario_path, 212)

    def test_selfplay_load_case_data_defaults_omitted_legacy_zero_fields(self) -> None:
        from training import selfplay

        legacy_basic = {
            "map_id": 212,
            "map_data": [
                [
                    {
                        "nb": [19, -1, -1, 17, 117, 118],
                        "c": 18,
                        "r": 0,
                        "e": 60,
                    }
                ]
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario_path = self._write_selfplay_case_fixture(root, map_id=212, basic_data=legacy_basic)
            with mock.patch.object(selfplay, "repo_root", return_value=root):
                loaded = selfplay._load_case_data(scenario_path, 212)

        cell = loaded["basic_data"]["map_data"][0][0]
        self.assertEqual(cell["neighbors"], legacy_basic["map_data"][0][0]["nb"])
        self.assertEqual(cell["roads"], [0, 0, 0, 0, 0, 0])
        self.assertEqual(cell["rivers"], [0, 0, 0, 0, 0, 0])
        self.assertEqual(cell["cond"], 0)
        self.assertEqual(cell["elev"], legacy_basic["map_data"][0][0]["e"])

    def test_selfplay_run_single_normalizes_basic_schema_for_env_and_agents(self) -> None:
        from training import selfplay

        legacy_basic = {
            "map_id": 212,
            "map_data": [
                [
                    {
                        "nb": [1, -1, -1, -1, -1, 100],
                        "c": 0,
                        "r": 0,
                        "e": 60,
                        "cd": 2,
                        "rd": [0, 0, 0, 0, 0, 0],
                        "rv": [0, 0, 0, 0, 0, 0],
                    }
                ]
            ],
        }

        class FakeAgent:
            setup_cells: list[dict[str, object]] = []

            def __init__(self, log_level: int) -> None:
                self.log_level = log_level

            def setup(self, payload: dict[str, object]) -> None:
                FakeAgent.setup_cells.append(dict(payload["basic_data"]["map_data"][0][0]))

            def step(self, observation: dict[str, object]) -> list[object]:
                return []

            def get_policy_stats(self) -> dict[str, object]:
                return {}

        class FakeEnv:
            setup_cells: list[dict[str, object]] = []

            def setup(self, env_info: dict[str, object]) -> list[dict[str, float]]:
                FakeEnv.setup_cells.append(dict(env_info["basic_data"]["map_data"][0][0]))
                return [{"reward": 0.0}, {"reward": 0.0}]

            def step(self, actions: list[object]) -> tuple[list[dict[str, float]], bool]:
                return [{"reward": 1.0}, {"reward": 0.0}], True

        red_profile = {
            "user_name": "CurrentAI",
            "profile": "challenger",
            "artifacts": "/tmp/candidate",
            "agent_class": FakeAgent,
        }
        blue_profile = {
            "user_name": "IncumbentAI",
            "profile": "baseline",
            "artifacts": "/tmp/incumbent",
            "agent_class": FakeAgent,
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario_path = self._write_selfplay_case_fixture(root, map_id=212, basic_data=legacy_basic)
            with (
                mock.patch.object(selfplay, "repo_root", return_value=root),
                mock.patch.object(selfplay, "_train_env_runtime", return_value=(FakeEnv, "train_env.TrainEnv")),
            ):
                red_score, blue_score, red_stats, blue_stats = selfplay._run_single(
                    scenario_path,
                    212,
                    red_profile,
                    blue_profile,
                    10,
                )

        self.assertEqual(red_score, 1.0)
        self.assertEqual(blue_score, 0.0)
        self.assertEqual(red_stats, {})
        self.assertEqual(blue_stats, {})
        self.assertEqual(len(FakeEnv.setup_cells), 1)
        self.assertEqual(len(FakeAgent.setup_cells), 2)
        for cell in FakeEnv.setup_cells + FakeAgent.setup_cells:
            self.assertEqual(cell["neighbors"], legacy_basic["map_data"][0][0]["nb"])
            self.assertEqual(cell["roads"], legacy_basic["map_data"][0][0]["rd"])
            self.assertEqual(cell["rivers"], legacy_basic["map_data"][0][0]["rv"])
            self.assertEqual(cell["cond"], legacy_basic["map_data"][0][0]["cd"])
            self.assertEqual(cell["elev"], legacy_basic["map_data"][0][0]["e"])

    def test_selfplay_run_single_emits_step_heartbeats(self) -> None:
        from training import selfplay

        class FakeAgent:
            def __init__(self, log_level: int) -> None:
                self.log_level = log_level

            def setup(self, payload: dict[str, object]) -> None:
                return None

            def step(self, observation: dict[str, object]) -> list[object]:
                return []

            def get_policy_stats(self) -> dict[str, object]:
                return {}

        class FakeEnv:
            def __init__(self) -> None:
                self.step_count = 0

            def setup(self, env_info: dict[str, object]) -> list[dict[str, float]]:
                return [{"reward": 0.0}, {"reward": 0.0}]

            def step(self, actions: list[object]) -> tuple[list[dict[str, float]], bool]:
                self.step_count += 1
                done = self.step_count >= 2
                return [{"reward": 1.0 if done else 0.0}, {"reward": 0.0}], done

        heartbeats: list[int] = []
        red_profile = {
            "user_name": "CurrentAI",
            "profile": "challenger",
            "artifacts": "/tmp/candidate",
            "agent_class": FakeAgent,
        }
        blue_profile = {
            "user_name": "IncumbentAI",
            "profile": "baseline",
            "artifacts": "/tmp/incumbent",
            "agent_class": FakeAgent,
        }

        with (
            mock.patch.object(selfplay, "_train_env_runtime", return_value=(FakeEnv, "train_env.TrainEnv")),
            mock.patch.object(selfplay, "_load_case_data", return_value={}),
        ):
            red_score, blue_score, red_stats, blue_stats = selfplay._run_single(
                Path("scenario_7.json"),
                7,
                red_profile,
                blue_profile,
                10,
                heartbeat=heartbeats.append,
            )

        self.assertEqual(heartbeats, [1, 2])
        self.assertEqual(red_score, 1.0)
        self.assertEqual(blue_score, 0.0)
        self.assertEqual(red_stats, {})
        self.assertEqual(blue_stats, {})

    def test_selfplay_progress_snapshot_persists_completed_pairs(self) -> None:
        from training.selfplay import _save_progress_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            candidate = {"source_commit_sha": "candsha"}
            incumbent = {"source_commit_sha": "incsha"}
            matches = [{"order_index": 0, "paired_score": 1.0}, {"order_index": 1, "paired_score": 0.5}]
            current_match = {
                "status": "in_progress",
                "scenario": "scenario_7.json",
                "bucket": "complex",
                "map_id": 7,
                "order_index": 3,
                "current_leg_index": 1,
                "leg_steps": 17,
                "first_leg": {"candidate": 1.0, "opponent": 0.0},
                "second_leg": None,
                "candidate_code_origin": "git:candsha",
                "opponent_code_origin": "git:incsha",
            }
            _save_progress_snapshot(
                run_dir,
                env_backend="train_env.TrainEnv",
                matches=matches,
                total_pairs=100,
                candidate=candidate,
                incumbent=incumbent,
                phase="formal_compare",
                current_match=current_match,
            )
            progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
            partial = json.loads((run_dir / "partial_match_results.json").read_text(encoding="utf-8"))
            participants_exists = (run_dir / "participants.json").exists()

        self.assertEqual(progress["completed_pairs"], 2)
        self.assertEqual(progress["total_pairs"], 100)
        self.assertEqual(progress["phase"], "formal_compare")
        self.assertEqual(progress["candidate_source_commit_sha"], "candsha")
        self.assertEqual(progress["incumbent_source_commit_sha"], "incsha")
        self.assertEqual(progress["current_order_index"], 3)
        self.assertEqual(progress["current_map_id"], 7)
        self.assertEqual(progress["current_scenario"], "scenario_7.json")
        self.assertEqual(progress["current_bucket"], "complex")
        self.assertEqual(progress["current_leg_index"], 1)
        self.assertEqual(progress["leg_steps"], 17)
        self.assertIsInstance(progress["heartbeat_ts"], str)
        self.assertEqual(partial["completed_pairs"], 2)
        self.assertEqual(partial["total_pairs"], 100)
        self.assertEqual(partial["phase"], "formal_compare")
        self.assertEqual(len(partial["matches"]), 2)
        self.assertEqual(partial["current_match"], current_match)
        self.assertFalse(participants_exists)

    def test_selfplay_main_writes_heartbeat_before_participants_sidecar(self) -> None:
        from training import selfplay

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario_root = root / "land_wargame_sdk" / "Data" / "Data" / "scenarios"
            scenario_root.mkdir(parents=True)
            (scenario_root / "scenario_7.json").write_text("{}", encoding="utf-8")

            candidate_dir = root / "candidate_dir"
            incumbent_dir = root / "incumbent_dir"
            fixed_tag = "20260424_120000"
            run_dir = root / "experiments" / f"league_{fixed_tag}"
            config = {
                "seed": 42,
                "output_root": "experiments",
                "league_dir": "league",
                "paired_matches": 1,
                "max_steps": 10,
                "promotion_gate": {
                    "overall_point_estimate_min": 0.5,
                    "overall_ci_lower_min": 0.4,
                    "complex_point_estimate_min": 0.5,
                    "complex_ci_lower_min": 0.4,
                    "consecutive_windows": 2,
                },
            }
            summary = {
                "overall": {"point_estimate": 1.0, "ci95": [1.0, 1.0]},
                "buckets": {"complex": {"point_estimate": 1.0, "ci95": [1.0, 1.0]}},
                "policy": {
                    "learned_path_coverage": 0.0,
                    "move_path_coverage": 0.0,
                    "shoot_path_coverage": 0.0,
                },
            }
            decision = {
                "accepted": False,
                "elo_delta_lower_bound": 0.0,
                "max_exploiter_win_rate": 0.0,
                "failures": [],
            }
            run_single_calls: list[int] = []

            def fake_build_participant_spec(
                artifacts: str,
                profile: str,
                fallback_commit_sha: str,
                user_name: str,
            ) -> dict[str, object]:
                source_commit_sha = "candsha" if user_name == "CurrentAI" else "incsha"
                return {
                    "artifacts": artifacts,
                    "profile": profile,
                    "user_name": user_name,
                    "source_commit_sha": source_commit_sha,
                    "code_origin": f"git:{source_commit_sha}",
                    "agent_class": object,
                }

            def fake_run_single(
                scenario_path: Path,
                map_id: int,
                red_profile: dict[str, object],
                blue_profile: dict[str, object],
                max_steps: int,
                heartbeat: object | None = None,
            ) -> tuple[float, float, dict[str, object], dict[str, object]]:
                leg_index = len(run_single_calls)
                progress_before = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
                partial_before = json.loads((run_dir / "partial_match_results.json").read_text(encoding="utf-8"))
                self.assertEqual(progress_before["phase"], "formal_compare")
                self.assertEqual(progress_before["current_order_index"], 0)
                self.assertEqual(progress_before["current_map_id"], 7)
                self.assertEqual(progress_before["current_scenario"], "scenario_7.json")
                self.assertEqual(progress_before["current_bucket"], "complex")
                self.assertEqual(progress_before["current_leg_index"], leg_index)
                self.assertEqual(progress_before["leg_steps"], 0)
                self.assertTrue(progress_before["heartbeat_ts"])
                self.assertFalse((run_dir / "participants.json").exists())
                self.assertEqual(partial_before["current_match"]["candidate_code_origin"], "git:candsha")
                self.assertEqual(partial_before["current_match"]["opponent_code_origin"], "git:incsha")
                if leg_index == 0:
                    self.assertIsNone(partial_before["current_match"]["first_leg"])
                else:
                    self.assertEqual(
                        partial_before["current_match"]["first_leg"],
                        {"candidate": 1.0, "opponent": 0.0},
                    )
                self.assertIsNotNone(heartbeat)
                assert callable(heartbeat)
                heartbeat(leg_index + 2)
                progress_after = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
                partial_after = json.loads((run_dir / "partial_match_results.json").read_text(encoding="utf-8"))
                self.assertEqual(progress_after["current_leg_index"], leg_index)
                self.assertEqual(progress_after["leg_steps"], leg_index + 2)
                self.assertEqual(partial_after["current_match"]["leg_steps"], leg_index + 2)
                run_single_calls.append(leg_index)
                if leg_index == 0:
                    return 1.0, 0.0, {}, {}
                return 0.0, 1.0, {}, {}

            with (
                mock.patch.object(selfplay, "repo_root", return_value=root),
                mock.patch.object(selfplay, "load_config", return_value=config),
                mock.patch.object(selfplay, "_configure_torch_cpu_inference_threads", return_value=None),
                mock.patch.object(selfplay, "set_seed", return_value=None),
                mock.patch.object(selfplay, "git_head_sha", return_value="headsha"),
                mock.patch.object(selfplay, "_resolve_incumbent_profile", return_value="baseline"),
                mock.patch.object(selfplay, "_build_participant_spec", side_effect=fake_build_participant_spec),
                mock.patch.object(selfplay, "_train_env_runtime", return_value=(object, "train_env.TrainEnv")),
                mock.patch.object(selfplay, "build_map_catalog", return_value=[]),
                mock.patch.object(selfplay, "map_bucket_lookup", return_value={7: "complex"}),
                mock.patch.object(
                    selfplay,
                    "_collect_scenario_candidates",
                    return_value=[{"scenario": "scenario_7.json", "map_id": 7, "bucket": "complex"}],
                ),
                mock.patch.object(
                    selfplay,
                    "_build_paired_map_list",
                    return_value={
                        "items": [{"scenario": "scenario_7.json", "map_id": 7, "bucket": "complex", "order_index": 0}],
                        "duplicates_used": False,
                    },
                ),
                mock.patch.object(
                    selfplay,
                    "_bootstrap_league",
                    return_value={
                        "champion": str(incumbent_dir),
                        "historical_mains": [],
                        "exploiters": [],
                        "specialists": [],
                        "consecutive_windows": 0,
                    },
                ),
                mock.patch.object(selfplay, "_run_single", side_effect=fake_run_single),
                mock.patch.object(selfplay, "summarize_matches", return_value=summary),
                mock.patch.object(selfplay, "promotion_decision", return_value=decision),
                mock.patch.object(selfplay, "now_tag", return_value=fixed_tag),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "training.selfplay",
                        "--config",
                        "ignored.json",
                        "--candidate",
                        str(candidate_dir),
                        "--incumbent",
                        str(incumbent_dir),
                    ],
                ),
            ):
                selfplay.main()

            participants = json.loads((run_dir / "participants.json").read_text(encoding="utf-8"))
            match_results = json.loads((run_dir / "match_results.json").read_text(encoding="utf-8"))
            progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))

        self.assertEqual(run_single_calls, [0, 1])
        self.assertEqual(participants["candidate"]["code_origin"], "git:candsha")
        self.assertEqual(participants["incumbent"]["code_origin"], "git:incsha")
        self.assertEqual(match_results["matches"][0]["candidate_code_origin"], "git:candsha")
        self.assertEqual(match_results["matches"][0]["opponent_code_origin"], "git:incsha")
        self.assertEqual(progress["completed_pairs"], 1)
        self.assertIsNone(progress["current_order_index"])

    def test_selfplay_main_writes_failed_sidecars_when_compare_aborts(self) -> None:
        from training import selfplay

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario_root = root / "land_wargame_sdk" / "Data" / "Data" / "scenarios"
            scenario_root.mkdir(parents=True)
            (scenario_root / "scenario_7.json").write_text("{}", encoding="utf-8")
            (scenario_root / "scenario_8.json").write_text("{}", encoding="utf-8")

            candidate_dir = root / "candidate_dir"
            incumbent_dir = root / "incumbent_dir"
            fixed_tag = "20260424_120100"
            run_dir = root / "experiments" / f"league_{fixed_tag}"
            config = {
                "seed": 42,
                "output_root": "experiments",
                "league_dir": "league",
                "paired_matches": 2,
                "max_steps": 10,
                "promotion_gate": {
                    "overall_point_estimate_min": 0.5,
                    "overall_ci_lower_min": 0.4,
                    "complex_point_estimate_min": 0.5,
                    "complex_ci_lower_min": 0.4,
                    "consecutive_windows": 2,
                },
            }
            run_single_calls: list[int] = []

            def fake_build_participant_spec(
                artifacts: str,
                profile: str,
                fallback_commit_sha: str,
                user_name: str,
            ) -> dict[str, object]:
                source_commit_sha = "candsha" if user_name == "CurrentAI" else "incsha"
                return {
                    "artifacts": artifacts,
                    "profile": profile,
                    "user_name": user_name,
                    "source_commit_sha": source_commit_sha,
                    "code_origin": f"git:{source_commit_sha}",
                    "agent_class": object,
                }

            def fake_run_single(
                scenario_path: Path,
                map_id: int,
                red_profile: dict[str, object],
                blue_profile: dict[str, object],
                max_steps: int,
                heartbeat: object | None = None,
            ) -> tuple[float, float, dict[str, object], dict[str, object]]:
                leg_index = len(run_single_calls)
                progress_before = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
                partial_before = json.loads((run_dir / "partial_match_results.json").read_text(encoding="utf-8"))
                self.assertEqual(progress_before["phase"], "formal_compare")
                self.assertEqual(partial_before["phase"], "formal_compare")
                if leg_index == 3:
                    self.assertEqual(progress_before["current_order_index"], 1)
                    self.assertEqual(progress_before["current_leg_index"], 1)
                    self.assertEqual(
                        partial_before["current_match"]["first_leg"],
                        {"candidate": 1.0, "opponent": 0.0},
                    )
                self.assertIsNotNone(heartbeat)
                assert callable(heartbeat)
                heartbeat(leg_index + 1)
                run_single_calls.append(leg_index)
                if leg_index == 0:
                    return 1.0, 0.0, {}, {}
                if leg_index == 1:
                    return 0.0, 1.0, {}, {}
                if leg_index == 2:
                    return 1.0, 0.0, {}, {}
                raise RuntimeError("compare exploded")

            with (
                mock.patch.object(selfplay, "repo_root", return_value=root),
                mock.patch.object(selfplay, "load_config", return_value=config),
                mock.patch.object(selfplay, "_configure_torch_cpu_inference_threads", return_value=None),
                mock.patch.object(selfplay, "set_seed", return_value=None),
                mock.patch.object(selfplay, "git_head_sha", return_value="headsha"),
                mock.patch.object(selfplay, "_resolve_incumbent_profile", return_value="baseline"),
                mock.patch.object(selfplay, "_build_participant_spec", side_effect=fake_build_participant_spec),
                mock.patch.object(selfplay, "_train_env_runtime", return_value=(object, "train_env.TrainEnv")),
                mock.patch.object(selfplay, "build_map_catalog", return_value=[]),
                mock.patch.object(selfplay, "map_bucket_lookup", return_value={7: "complex", 8: "complex"}),
                mock.patch.object(
                    selfplay,
                    "_collect_scenario_candidates",
                    return_value=[
                        {"scenario": "scenario_7.json", "map_id": 7, "bucket": "complex"},
                        {"scenario": "scenario_8.json", "map_id": 8, "bucket": "complex"},
                    ],
                ),
                mock.patch.object(
                    selfplay,
                    "_build_paired_map_list",
                    return_value={
                        "items": [
                            {"scenario": "scenario_7.json", "map_id": 7, "bucket": "complex", "order_index": 0},
                            {"scenario": "scenario_8.json", "map_id": 8, "bucket": "complex", "order_index": 1},
                        ],
                        "duplicates_used": False,
                    },
                ),
                mock.patch.object(
                    selfplay,
                    "_bootstrap_league",
                    return_value={
                        "champion": str(incumbent_dir),
                        "historical_mains": [],
                        "exploiters": [],
                        "specialists": [],
                        "consecutive_windows": 0,
                    },
                ),
                mock.patch.object(selfplay, "_run_single", side_effect=fake_run_single),
                mock.patch.object(selfplay, "now_tag", return_value=fixed_tag),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "training.selfplay",
                        "--config",
                        "ignored.json",
                        "--candidate",
                        str(candidate_dir),
                        "--incumbent",
                        str(incumbent_dir),
                    ],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "compare exploded"):
                    selfplay.main()

            progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
            partial = json.loads((run_dir / "partial_match_results.json").read_text(encoding="utf-8"))
            participants = json.loads((run_dir / "participants.json").read_text(encoding="utf-8"))
            match_results = json.loads((run_dir / "match_results.json").read_text(encoding="utf-8"))
            rating = json.loads((run_dir / "rating.json").read_text(encoding="utf-8"))
            promotion = json.loads((run_dir / "promotion.json").read_text(encoding="utf-8"))
            exploitability = json.loads((run_dir / "exploitability.json").read_text(encoding="utf-8"))
            summary_text = (run_dir / "summary.md").read_text(encoding="utf-8")

        self.assertEqual(run_single_calls, [0, 1, 2, 3])
        self.assertEqual(progress["phase"], "formal_compare_failed")
        self.assertEqual(progress["status"], "failed")
        self.assertEqual(progress["completed_pairs"], 1)
        self.assertEqual(progress["current_order_index"], 1)
        self.assertEqual(progress["current_leg_index"], 1)
        self.assertEqual(progress["leg_steps"], 4)
        self.assertEqual(progress["error"]["type"], "RuntimeError")
        self.assertEqual(progress["error"]["message"], "compare exploded")
        self.assertEqual(partial["status"], "failed")
        self.assertEqual(len(partial["matches"]), 1)
        self.assertEqual(partial["current_match"]["status"], "failed")
        self.assertEqual(partial["current_match"]["order_index"], 1)
        self.assertEqual(
            partial["current_match"]["first_leg"],
            {"candidate": 1.0, "opponent": 0.0},
        )
        self.assertEqual(partial["current_match"]["error_type"], "RuntimeError")
        self.assertEqual(partial["current_match"]["error_message"], "compare exploded")
        self.assertEqual(participants["candidate"]["code_origin"], "git:candsha")
        self.assertEqual(participants["incumbent"]["code_origin"], "git:incsha")
        self.assertEqual(match_results["status"], "failed")
        self.assertEqual(match_results["completed_pairs"], 1)
        self.assertEqual(len(match_results["matches"]), 1)
        self.assertEqual(match_results["matches"][0]["order_index"], 0)
        self.assertEqual(match_results["current_match"]["status"], "failed")
        self.assertEqual(rating["status"], "failed")
        self.assertEqual(promotion["status"], "failed")
        self.assertFalse(promotion["accepted"])
        self.assertEqual(promotion["failures"], ["compare_failed"])
        self.assertEqual(exploitability["status"], "failed")
        self.assertIn("formal_compare_failed", summary_text)
        self.assertIn("compare exploded", summary_text)

    def test_selfplay_runtime_uses_critical_log_level(self) -> None:
        import logging

        from training.selfplay import _AGENT_RUNTIME_LOG_LEVEL

        self.assertEqual(_AGENT_RUNTIME_LOG_LEVEL, logging.CRITICAL)

    def test_selfplay_caps_torch_cpu_inference_threads(self) -> None:
        from training.selfplay import _configure_torch_cpu_inference_threads

        class FakeTorch:
            def __init__(self) -> None:
                self.num_threads = 72
                self.num_interop_threads = 72

            def set_num_threads(self, value: int) -> None:
                self.num_threads = value

            def set_num_interop_threads(self, value: int) -> None:
                self.num_interop_threads = value

            def get_num_threads(self) -> int:
                return self.num_threads

            def get_num_interop_threads(self) -> int:
                return self.num_interop_threads

        fake_torch = FakeTorch()

        def fake_import(name: str) -> object:
            if name == "torch":
                return fake_torch
            raise AssertionError(f"unexpected import: {name}")

        configured = _configure_torch_cpu_inference_threads(fake_import)
        self.assertEqual(configured, {"num_threads": 1, "num_interop_threads": 1})

    def test_learned_policy_configure_moves_models_to_cuda_when_available(self) -> None:
        from ai import policy as policy_module

        class DummyModel:
            def __init__(self) -> None:
                self.devices: list[str] = []

            def to(self, device: str) -> "DummyModel":
                self.devices.append(str(device))
                return self

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            (artifacts / "model_manifest.json").write_text(
                json.dumps({"history_len": 4, "intent_lstm": "intent.pt", "situation_gcn": "situation.pt"}),
                encoding="utf-8",
            )
            (artifacts / "intent.pt").write_bytes(b"intent")
            (artifacts / "situation.pt").write_bytes(b"situation")
            intent_model = DummyModel()
            situation_model = DummyModel()
            with (
                mock.patch.object(policy_module.torch.cuda, "is_available", return_value=True),
                mock.patch.object(policy_module.torch, "load", side_effect=[{"model_state": {}}, {"model_state": {}}]),
                mock.patch.object(policy_module.LearnedPolicy, "_load_intent_model", return_value=intent_model),
                mock.patch.object(policy_module.LearnedPolicy, "_load_situation_model", return_value=situation_model),
            ):
                learned = policy_module.LearnedPolicy()
                learned.configure(0, str(artifacts), "challenger")

        self.assertEqual(learned.device, "cuda")
        self.assertEqual(intent_model.devices, ["cuda"])
        self.assertEqual(situation_model.devices, ["cuda"])
        self.assertEqual(learned.get_diagnostics()["device"], "cuda")
        self.assertTrue(learned.has_models())

    def test_learned_policy_configure_keeps_cpu_fallback_without_cuda(self) -> None:
        from ai import policy as policy_module

        class DummyModel:
            def __init__(self) -> None:
                self.devices: list[str] = []

            def to(self, device: str) -> "DummyModel":
                self.devices.append(str(device))
                return self

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            (artifacts / "model_manifest.json").write_text(
                json.dumps({"history_len": 4, "intent_lstm": "intent.pt", "situation_gcn": "situation.pt"}),
                encoding="utf-8",
            )
            (artifacts / "intent.pt").write_bytes(b"intent")
            (artifacts / "situation.pt").write_bytes(b"situation")
            intent_model = DummyModel()
            situation_model = DummyModel()
            with (
                mock.patch.object(policy_module.torch.cuda, "is_available", return_value=False),
                mock.patch.object(policy_module.torch, "load", side_effect=[{"model_state": {}}, {"model_state": {}}]),
                mock.patch.object(policy_module.LearnedPolicy, "_load_intent_model", return_value=intent_model),
                mock.patch.object(policy_module.LearnedPolicy, "_load_situation_model", return_value=situation_model),
            ):
                learned = policy_module.LearnedPolicy()
                learned.configure(0, str(artifacts), "challenger")

        self.assertEqual(learned.device, "cpu")
        self.assertEqual(intent_model.devices, ["cpu"])
        self.assertEqual(situation_model.devices, ["cpu"])
        self.assertEqual(learned.get_diagnostics()["device"], "cpu")
        self.assertTrue(learned.has_models())

    def test_learned_policy_configure_allows_explicit_cpu_override_with_cuda_visible(self) -> None:
        from ai import policy as policy_module

        class DummyModel:
            def __init__(self) -> None:
                self.devices: list[str] = []

            def to(self, device: str) -> "DummyModel":
                self.devices.append(str(device))
                return self

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            (artifacts / "model_manifest.json").write_text(
                json.dumps({"history_len": 4, "intent_lstm": "intent.pt", "situation_gcn": "situation.pt"}),
                encoding="utf-8",
            )
            (artifacts / "intent.pt").write_bytes(b"intent")
            (artifacts / "situation.pt").write_bytes(b"situation")
            intent_model = DummyModel()
            situation_model = DummyModel()
            with (
                mock.patch.dict(policy_module.os.environ, {"SHENJI_POLICY_DEVICE": "cpu"}, clear=False),
                mock.patch.object(policy_module.torch.cuda, "is_available", return_value=True),
                mock.patch.object(policy_module.torch, "load", side_effect=[{"model_state": {}}, {"model_state": {}}]),
                mock.patch.object(policy_module.LearnedPolicy, "_load_intent_model", return_value=intent_model),
                mock.patch.object(policy_module.LearnedPolicy, "_load_situation_model", return_value=situation_model),
            ):
                learned = policy_module.LearnedPolicy()
                learned.configure(0, str(artifacts), "challenger")

        self.assertEqual(learned.device, "cpu")
        self.assertEqual(learned.get_diagnostics()["requested_device"], "cpu")
        self.assertEqual(intent_model.devices, ["cpu"])
        self.assertEqual(situation_model.devices, ["cpu"])
        self.assertTrue(learned.has_models())

    def test_weapon_lock_ignores_invalid_cur_hex(self) -> None:
        from ai.agent import ActionType, Agent, Map

        agent = Agent(seed=0)
        agent.seat = 1
        agent.map = Map({"map_data": [[{"roads": [0], "neighbors": []}]]}, [[]], None)
        agent.map_data = agent.map.get_map_data()
        agent.observation = {
            "operators": [
                {
                    "obj_id": 7,
                    "cur_hex": 9900,
                }
            ]
        }

        self.assertIsNone(agent._generate_action(7, ActionType.WeaponLock, []))

    def test_tactical_dataset_uses_explicit_label_column(self) -> None:
        _install_fake_torch()
        from training.data import TacticalPositionDataset

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "toy.csv"
            csv_path.write_text(
                "ID,样本标签,整体移动方向,数值特征\n"
                "0,1,9,0.5\n"
                "1,0,8,1.5\n",
                encoding="utf-8",
            )
            dataset = TacticalPositionDataset(csv_path, label_column="样本标签")
        self.assertEqual(dataset.labels.tolist(), [1.0, 0.0])
        self.assertEqual(dataset.label_column, "样本标签")

    def test_map_id_resolution_can_be_scoped_to_filename_segment(self) -> None:
        _install_fake_torch()
        from training.data import resolve_map_id_by_strategy

        self.assertEqual(
            resolve_map_id_by_strategy(
                "20210109101817255-201033039601-Liquidator-DemoAI-t83lvwyW.zip",
                [83, 9601],
                "suffix",
                filename_segment_index=1,
            ),
            9601,
        )
        self.assertEqual(
            resolve_map_id_by_strategy(
                "20210730145902084-2110431553-zBv9bHRr.json",
                [53, 84],
                "suffix",
                filename_segment_index=1,
            ),
            53,
        )

    def test_selfplay_merge_policy_stats_keeps_model_load_diagnostics(self) -> None:
        from training.selfplay import _build_match_policy_diagnostics, _merge_policy_stats

        merged = _merge_policy_stats(
            {
                "policy_profile": "incumbent",
                "device": "cuda",
                "loaded_from": "/tmp/strongest",
                "intent_model_loaded": True,
                "situation_model_loaded": True,
                "counts": {
                    "move": {"total": 1, "learned": 1},
                    "shoot": {"total": 0, "learned": 0},
                },
                "score_means": {
                    "move": {"samples": 1, "intent_bias": 1.0, "situation_score": 0.4, "combined_score": 2.0},
                    "shoot": {"samples": 0, "intent_bias": 0.0, "situation_score": 0.0, "combined_score": 0.0},
                },
            },
            {
                "policy_profile": "incumbent",
                "device": "cuda",
                "loaded_from": "/tmp/strongest",
                "intent_model_loaded": True,
                "situation_model_loaded": True,
                "counts": {
                    "move": {"total": 1, "learned": 1},
                    "shoot": {"total": 1, "learned": 1},
                },
                "score_means": {
                    "move": {"samples": 1, "intent_bias": 3.0, "situation_score": 0.6, "combined_score": 4.0},
                    "shoot": {"samples": 1, "intent_bias": 2.0, "situation_score": 0.7, "combined_score": 5.0},
                },
            },
        )
        self.assertEqual(merged["policy_profiles"], ["incumbent"])
        self.assertEqual(merged["inference_devices"], ["cuda"])
        self.assertEqual(merged["loaded_from_dirs"], ["/tmp/strongest"])
        self.assertEqual(merged["intent_model_loaded_count"], 2)
        self.assertEqual(merged["situation_model_loaded_count"], 2)

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            (artifacts / "model_manifest.json").write_text(
                json.dumps(
                    {
                        "generated_at": "20260427_000000",
                        "source_commit_sha": "candsha",
                        "snapshot_role": "accepted_learned_mainline",
                        "managed_files": ["intent_lstm.pt", "situation_gcn.pt", "tactical_model.pt", "model_manifest.json"],
                    }
                ),
                encoding="utf-8",
            )
            diagnostics = _build_match_policy_diagnostics(
                artifacts=str(artifacts),
                participant={"user_name": "CurrentAI", "source_commit_sha": "candsha", "code_origin": "git:candsha"},
                profile="challenger",
                stats=merged,
            )

        self.assertEqual(diagnostics["user_name"], "CurrentAI")
        self.assertEqual(diagnostics["profile"], "challenger")
        self.assertTrue(diagnostics["manifest_loaded"])
        self.assertTrue(diagnostics["intent_model_loaded"])
        self.assertTrue(diagnostics["situation_model_loaded"])
        self.assertTrue(diagnostics["tactical_model_present"])
        self.assertFalse(diagnostics["tactical_model_loaded"])
        self.assertTrue(diagnostics["learned_policy_requested"])
        self.assertTrue(diagnostics["learned_policy_enabled"])
        self.assertEqual(diagnostics["fallback_reason"], [])

    def test_summarize_matches_reports_policy_coverage(self) -> None:
        _install_fake_torch()
        from training.evaluate import summarize_matches

        matches = [
            {
                "paired_score": 0.75,
                "bucket": "complex",
                "candidate_policy_stats": {
                    "counts": {
                        "move": {"total": 2, "learned": 1},
                        "shoot": {"total": 2, "learned": 2},
                    },
                    "score_means": {
                        "move": {
                            "samples": 1,
                            "intent_bias": 1.0,
                            "situation_score": 0.5,
                            "tactical_score": 0.2,
                            "objective_score": 4.0,
                            "combined_score": 3.0,
                        },
                        "shoot": {
                            "samples": 1,
                            "intent_bias": 2.0,
                            "situation_score": 0.7,
                            "tactical_score": 0.3,
                            "objective_score": 6.0,
                            "combined_score": 5.0,
                        },
                    },
                },
                "candidate_policy_diagnostics": {
                    "artifact_dir": "/tmp/candidate",
                    "manifest_path": "/tmp/candidate/model_manifest.json",
                    "manifest_loaded": True,
                    "intent_model_loaded": True,
                    "situation_model_loaded": True,
                    "tactical_model_loaded": False,
                    "learned_policy_enabled": True,
                    "profile": "challenger",
                    "fallback_reason": [],
                },
            }
        ]
        summary = summarize_matches(matches)
        self.assertAlmostEqual(summary["policy"]["learned_path_coverage"], 0.75)
        self.assertAlmostEqual(summary["policy"]["move_path_coverage"], 0.5)
        self.assertAlmostEqual(summary["policy"]["shoot_path_coverage"], 1.0)
        self.assertEqual(summary["policy"]["manifest_loaded_match_count"], 1)
        self.assertEqual(summary["policy"]["intent_model_loaded_match_count"], 1)
        self.assertEqual(summary["policy"]["situation_model_loaded_match_count"], 1)
        self.assertEqual(summary["policy"]["tactical_model_loaded_match_count"], 0)
        self.assertEqual(summary["policy"]["learned_policy_enabled_match_count"], 1)
        self.assertEqual(summary["policy"]["profiles"], ["challenger"])
        self.assertEqual(summary["policy"]["artifact_dirs"], ["/tmp/candidate"])
        self.assertEqual(summary["policy"]["manifest_paths"], ["/tmp/candidate/model_manifest.json"])
        self.assertAlmostEqual(summary["policy"]["score_means"]["move"]["objective_score"], 4.0)
        self.assertAlmostEqual(summary["policy"]["score_means"]["shoot"]["objective_score"], 6.0)

    def test_promotion_decision_enforces_sota_stop_guards(self) -> None:
        _install_fake_torch()
        from training.evaluate import promotion_decision

        summary = {
            "overall": {"point_estimate": 0.62, "ci95": (0.56, 0.67), "elo": 84.0, "count": 100},
            "buckets": {
                "complex": {"point_estimate": 0.61, "ci95": (0.56, 0.66), "count": 50},
                "medium": {"point_estimate": 0.58, "ci95": (0.52, 0.63), "count": 30},
                "simple": {"point_estimate": 0.57, "ci95": (0.50, 0.64), "count": 20},
            },
            "policy": {
                "learned_path_coverage": 0.34,
                "move_path_coverage": 0.24,
                "shoot_path_coverage": 0.21,
            },
        }
        decision = promotion_decision(
            summary,
            {
                "overall_point_estimate_min": 0.60,
                "overall_ci_lower_min": 0.55,
                "complex_point_estimate_min": 0.60,
                "complex_ci_lower_min": 0.55,
                "min_learned_path_coverage": 0.30,
                "min_move_path_coverage": 0.20,
                "min_shoot_path_coverage": 0.20,
                "elo_delta_lower_min": 35.0,
                "max_bucket_regression": 0.03,
                "max_exploiter_win_rate": 0.45,
            },
            {
                "aggressive_exploiter": {"win_rate_vs_candidate": 0.42},
                "occupy_exploiter": {"win_rate_vs_candidate": 0.40},
            },
        )
        self.assertTrue(decision["accepted"])
        self.assertGreaterEqual(decision["elo_delta_lower_bound"], 35.0)
        self.assertAlmostEqual(decision["max_exploiter_win_rate"], 0.42)

        blocked = promotion_decision(
            summary,
            {
                "overall_point_estimate_min": 0.60,
                "overall_ci_lower_min": 0.55,
                "complex_point_estimate_min": 0.60,
                "complex_ci_lower_min": 0.55,
                "min_learned_path_coverage": 0.30,
                "min_move_path_coverage": 0.20,
                "min_shoot_path_coverage": 0.20,
                "elo_delta_lower_min": 35.0,
                "max_bucket_regression": 0.03,
                "max_exploiter_win_rate": 0.45,
            },
            {
                "aggressive_exploiter": {"win_rate_vs_candidate": 0.48},
            },
        )
        self.assertFalse(blocked["accepted"])
        self.assertIn("exploiter_win_rate", blocked["failures"])

    def test_selfplay_rejects_same_candidate_and_incumbent(self) -> None:
        from training import selfplay

        with mock.patch.object(
            selfplay,
            "load_config",
            return_value={
                "seed": 42,
                "output_root": "experiments/shenji_agent",
                "league_dir": "league",
                "paired_matches": 2,
                "max_steps": 10,
                "promotion_gate": {
                    "overall_point_estimate_min": 0.5,
                    "overall_ci_lower_min": 0.4,
                    "complex_point_estimate_min": 0.5,
                    "complex_ci_lower_min": 0.4,
                },
            },
        ), mock.patch.object(
            sys,
            "argv",
            [
                "training.selfplay",
                "--config",
                "ignored.json",
                "--candidate",
                "ai/weights",
                "--incumbent",
                "ai/weights",
            ],
        ):
            with self.assertRaises(SystemExit) as ctx:
                selfplay.main()
        self.assertIn("candidate and incumbent must point to different artifact directories", str(ctx.exception))

    def test_report_progress_returns_none_for_missing_root(self) -> None:
        missing_root = REPO_ROOT / "experiments" / "missing_for_test"
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/report_progress.py",
                "--experiments-root",
                str(missing_root),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["event"], "none")
        self.assertIsNone(payload["run_dir"])

    def test_learned_move_prefers_reachable_city_for_non_empty_route(self) -> None:
        from ai.agent import ActionType, Agent, BopType, MoveType

        class FakeMap:
            def __init__(self) -> None:
                self.calls: list[tuple[int, int, int]] = []

            def gen_move_route(self, begin: int, end: int, mode: int) -> list[int]:
                self.calls.append((begin, end, mode))
                if end == 102:
                    return [12, 102]
                return []

            def get_distance(self, pos1: int, pos2: int) -> int:
                return abs(pos1 - pos2) // 100

            def get_neighbors(self, pos: int) -> list[int]:
                return [pos - 100, pos + 100, pos - 1, pos + 1]

        class FakePolicy:
            def __init__(self) -> None:
                self.choose_city_calls: list[list[int]] = []
                self.note_calls: list[tuple[str, bool, object]] = []

            def choose_city(self, bop, cities, observation, tactical_map):
                coords = [int(city["coord"]) for city in cities]
                self.choose_city_calls.append(coords)
                city = cities[0] if cities else None
                trace = {"obj_id": int(bop["obj_id"]), "candidate_id": int(city["coord"]), "combined_score": 1.0}
                return city, trace

            def note_action(self, action_type: str, used_learned: bool, trace=None) -> None:
                self.note_calls.append((action_type, used_learned, trace))

            def record_move_route(self, diagnostic: dict) -> None:
                pass

        fake_map = FakeMap()
        fake_policy = FakePolicy()
        agent = Agent(seed=0)
        agent.use_ai_optimization = True
        agent.seat = 1
        agent.map = fake_map
        agent.policy = fake_policy
        agent.observation = {
            "operators": [
                {
                    "obj_id": 7,
                    "cur_hex": 11,
                    "sub_type": 2,
                    "type": BopType.Vehicle,
                    "move_state": MoveType.Maneuver,
                }
            ],
            "cities": [
                {"coord": 101, "value": 100.0},
                {"coord": 102, "value": 1.0},
            ],
        }

        action = agent._generate_action(7, ActionType.Move, [{}])

        self.assertEqual(
            fake_map.calls,
            [
                (11, 101, MoveType.Maneuver),
                (11, 102, MoveType.Maneuver),
                (11, 101, MoveType.March),
                (11, 102, MoveType.March),
                (11, 101, MoveType.Walk),
                (11, 102, MoveType.Walk),
                (11, 101, MoveType.Fly),
                (11, 102, MoveType.Fly),
            ],
        )
        self.assertEqual(fake_policy.choose_city_calls, [[102]])
        self.assertEqual(
            action,
            {"actor": 1, "obj_id": 7, "type": ActionType.Move, "move_path": [12, 102]},
        )
        self.assertEqual(
            fake_policy.note_calls,
            [("move", True, {"obj_id": 7, "candidate_id": 102, "combined_score": 1.0})],
        )

    def test_learned_move_with_no_reachable_city_does_not_claim_learned(self) -> None:
        from ai.agent import ActionType, Agent, BopType, MoveType

        class FakeMap:
            def __init__(self) -> None:
                self.calls: list[tuple[int, int, int]] = []

            def gen_move_route(self, begin: int, end: int, mode: int) -> list[int]:
                self.calls.append((begin, end, mode))
                return []

            def get_distance(self, pos1: int, pos2: int) -> int:
                return abs(pos1 - pos2) // 100

            def get_neighbors(self, pos: int) -> list[int]:
                return [pos - 100, pos + 100, pos - 1, pos + 1]

        class FakePolicy:
            def __init__(self) -> None:
                self.choose_city_calls: list[list[int]] = []
                self.note_calls: list[tuple[str, bool, object]] = []

            def choose_city(self, bop, cities, observation, tactical_map):
                coords = [int(city["coord"]) for city in cities]
                self.choose_city_calls.append(coords)
                return None, None

            def note_action(self, action_type: str, used_learned: bool, trace=None) -> None:
                self.note_calls.append((action_type, used_learned, trace))

            def record_move_route(self, diagnostic: dict) -> None:
                pass

        fake_map = FakeMap()
        fake_policy = FakePolicy()
        agent = Agent(seed=0)
        agent.use_ai_optimization = True
        agent.seat = 1
        agent.map = fake_map
        agent.policy = fake_policy
        agent.observation = {
            "operators": [
                {
                    "obj_id": 7,
                    "cur_hex": 11,
                    "sub_type": 2,
                    "type": BopType.Vehicle,
                    "move_state": MoveType.Maneuver,
                }
            ],
            "cities": [
                {"coord": 101, "value": 100.0},
                {"coord": 102, "value": 1.0},
            ],
        }

        action = agent._generate_action(7, ActionType.Move, [{}])

        self.assertIsNone(action)
        self.assertEqual(
            fake_map.calls,
            [
                (11, 101, MoveType.Maneuver),
                (11, 102, MoveType.Maneuver),
                (11, 101, MoveType.March),
                (11, 102, MoveType.March),
                (11, 101, MoveType.Walk),
                (11, 102, MoveType.Walk),
                (11, 101, MoveType.Fly),
                (11, 102, MoveType.Fly),
            ],
        )
        self.assertEqual(fake_policy.choose_city_calls, [])
        self.assertEqual(len(fake_policy.note_calls), 1)
        action_type, used_learned, trace = fake_policy.note_calls[0]
        self.assertEqual(action_type, "move")
        self.assertFalse(used_learned)
        self.assertEqual(trace, {"obj_id": 7, "reason": "empty_route"})

    def test_learned_move_with_no_valid_candidate_tracks_reason(self) -> None:
        from ai.agent import ActionType, Agent, BopType, MoveType

        class FakeMap:
            def __init__(self) -> None:
                self.calls: list[tuple[int, int, int]] = []

            def gen_move_route(self, begin: int, end: int, mode: int) -> list[int]:
                self.calls.append((begin, end, mode))
                return []

            def get_distance(self, pos1: int, pos2: int) -> int:
                return abs(pos1 - pos2) // 100

            def get_neighbors(self, pos: int) -> list[int]:
                return [pos - 100, pos + 100, pos - 1, pos + 1]

        class FakePolicy:
            def __init__(self) -> None:
                self.choose_city_calls: list[list[int]] = []
                self.note_calls: list[tuple[str, bool, object]] = []

            def choose_city(self, bop, cities, observation, tactical_map):
                coords = [int(city["coord"]) for city in cities]
                self.choose_city_calls.append(coords)
                return None, None

            def note_action(self, action_type: str, used_learned: bool, trace=None) -> None:
                self.note_calls.append((action_type, used_learned, trace))

            def record_move_route(self, diagnostic: dict) -> None:
                pass

        fake_map = FakeMap()
        fake_policy = FakePolicy()
        agent = Agent(seed=0)
        agent.use_ai_optimization = True
        agent.seat = 1
        agent.map = fake_map
        agent.policy = fake_policy
        agent.observation = {
            "operators": [
                {
                    "obj_id": 7,
                    "cur_hex": 11,
                    "sub_type": 2,
                    "type": BopType.Vehicle,
                    "move_state": MoveType.Maneuver,
                }
            ],
            "cities": [
                {"coord": 101, "value": 100.0},
                {"coord": 102, "value": 1.0},
            ],
        }

        action = agent._generate_action(7, ActionType.Move, None)

        self.assertIsNone(action)
        self.assertEqual(
            fake_map.calls,
            [
                (11, 101, MoveType.Maneuver),
                (11, 102, MoveType.Maneuver),
                (11, 101, MoveType.March),
                (11, 102, MoveType.March),
                (11, 101, MoveType.Walk),
                (11, 102, MoveType.Walk),
                (11, 101, MoveType.Fly),
                (11, 102, MoveType.Fly),
            ],
        )
        self.assertEqual(fake_policy.choose_city_calls, [])
        self.assertEqual(
            fake_policy.note_calls,
            [("move", False, {"obj_id": 7, "reason": "no_valid_move_candidate"})],
        )


if __name__ == "__main__":
    unittest.main()
