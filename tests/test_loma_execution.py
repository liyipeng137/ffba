"""CPU contracts; real LoMa numerical checks live in scripts/check_loma_execution.py."""

import ast
from contextlib import nullcontext
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from utils import loma_prior as loma
from utils.loma_execution import (
    GeometryQueue,
    bucket_batches,
    image_shape,
    prepare_image,
    prefetch_batches,
    validate_execution,
)


class ArrayTensor:
    """Only the tensor conversions used by the actual native image loaders."""

    def __init__(self, array):
        self.array = array

    def permute(self, *dims):
        return ArrayTensor(self.array.transpose(dims))

    def float(self):
        return ArrayTensor(self.array.astype(np.float32))

    def to(self, device):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def __getitem__(self, key):
        return ArrayTensor(self.array[key])


def native_loader(relative, class_name, method):
    path = Path(__file__).resolve().parents[1] / "third_party/LoMa/src/loma" / relative
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    node = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method
    )
    ns = {
        "torch": SimpleNamespace(from_numpy=ArrayTensor, Tensor=ArrayTensor),
        "np": np,
        "Image": Image,
        "device": "cpu",
        "check_not_i16": lambda image: None,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns[method]


@pytest.mark.parametrize(
    "size,mode,aspect",
    [
        ((127, 83), "RGB", True),
        ((731, 1297), "RGB", True),
        ((1433, 891), "L", True),
        ((93, 107), "RGBA", False),
    ],
)
def test_preprocessing_matches_vendored_loaders_exactly(tmp_path, size, mode, aspect):
    channels = {"RGB": 3, "RGBA": 4}.get(mode)
    shape = (size[1], size[0]) + ((channels,) if channels else ())
    pixels = np.random.default_rng(24).integers(0, 256, shape, dtype=np.uint8)
    path = tmp_path / "image.png"
    Image.fromarray(pixels).save(path)
    cfg = SimpleNamespace(resize=1024, keep_aspect_ratio=aspect)
    native_detector = native_loader("detector/dad.py", "DaD", "load_image")(
        cfg, path
    ).array[0]
    native_descriptor = native_loader(
        "descriptor/dedode.py", "DeDoDeDescriptor", "read_image"
    )(None, path).array[0]
    prepared = prepare_image(path, keep_aspect_ratio=aspect)
    np.testing.assert_array_equal(prepared["detector"], native_detector)
    np.testing.assert_array_equal(prepared["descriptor"], native_descriptor)
    assert prepared["image_size_hw"] == size[::-1]
    assert image_shape(path, keep_aspect_ratio=aspect) == native_detector.shape[:0:-1]


def test_preprocessing_errors_identify_input(tmp_path):
    path = tmp_path / "missing.png"
    with pytest.raises(RuntimeError, match="missing.png"):
        prepare_image(path)
    path = tmp_path / "depth.png"
    Image.fromarray(np.zeros((16, 16), dtype=np.uint16)).save(path)
    with pytest.raises(RuntimeError, match="16 bit"):
        prepare_image(path)


def test_buckets_preserve_ids_and_tails_without_padding():
    assert list(bucket_batches(range(7), lambda i: i % 2, 3)) == [
        [0, 2, 4],
        [6],
        [1, 3, 5],
    ]
    assert list(bucket_batches([], lambda i: i, 2)) == []


def test_adapter_unpacks_every_match_batch_element_and_preserves_feature_ids():
    backend = object.__new__(loma.LoMaBackend)
    backend.torch = SimpleNamespace(inference_mode=nullcontext)
    backend._measure = lambda *_: nullcontext()
    backend.collect_events = lambda: None
    backend._upload = lambda tensors: np.concatenate(tensors, axis=0)
    features = [
        {
            "keypoints": np.zeros((3, 2)),
            "normalized": np.full((1, 3, 2), i),
            "descriptors": np.full((1, 3, 4), i + 10),
        }
        for i in range(3)
    ]
    calls = []

    def forward(*inputs):
        calls.append(inputs)
        return {"scores": "compact result fixture"}

    backend.model = SimpleNamespace(cfg=SimpleNamespace(filter_threshold=0.1))

    # SimpleNamespace is not callable; retain the real adapter's cfg + forward API.
    class Model:
        cfg = backend.model.cfg

        def __call__(self, *inputs):
            return forward(*inputs)

    backend.model = Model()
    backend.filter_matches = lambda result, threshold: (
        ArrayTensor(np.array([[1, -1, 0], [-1, 2, -1]])),
        None,
        ArrayTensor(np.array([[0.9, 0.0, 0.8], [0.0, 0.7, 0.0]])),
        None,
    )
    outputs = backend.match_batch(
        [(features[0], features[1]), (features[1], features[2])]
    )
    np.testing.assert_array_equal(outputs[0][0], [[0, 1], [2, 0]])
    np.testing.assert_array_equal(outputs[1][0], [[1, 2]])
    np.testing.assert_allclose(outputs[0][1], [0.9, 0.8])
    np.testing.assert_allclose(outputs[1][1], [0.7])
    for array, expected in zip(calls[0], ([0, 1], [1, 2], [10, 11], [11, 12])):
        np.testing.assert_array_equal(array[:, 0, 0], expected)
    assert backend.match_batch([]) == []
    empty = {"keypoints": np.empty((0, 2))}
    assert all(
        len(matches) == 0
        for matches, _ in backend.match_batch([(empty, features[0])] * 2)
    )
    assert len(calls) == 1


def test_prefetch_bound_order_and_early_close():
    started = []
    lock = threading.Lock()

    def prepare(i):
        with lock:
            started.append(i)
        return i * 10

    stats = {}
    iterator = prefetch_batches([[0, 1], [2, 3], [4]], prepare, 4, stats)
    assert next(iterator) == [(0, 0), (1, 10)]
    # The consumer still owns batch 1; no third batch can be scheduled yet.
    assert set(started) <= {0, 1, 2, 3}
    assert list(iterator) == [[(2, 20), (3, 30)], [(4, 40)]]
    assert stats["peak_pending_images"] == 4
    iterator = prefetch_batches([[0], [1], [2]], prepare, 2, {})
    next(iterator)
    iterator.close()
    assert not any(t.name.startswith("loma-preprocess") for t in threading.enumerate())


def test_prefetch_propagates_failure_and_joins_workers():
    def prepare(i):
        if i == 1:
            raise RuntimeError("image 1 broken")
        return i

    with pytest.raises(RuntimeError, match="image 1 broken"):
        list(prefetch_batches([[0, 1], [2]], prepare, 2, {}))
    assert not any(t.name.startswith("loma-preprocess") for t in threading.enumerate())


def test_geometry_runs_concurrently_with_bounded_queue_and_main_thread_consumer():
    barrier = threading.Barrier(2, timeout=5)
    received = {}
    consumer_thread = threading.get_ident()

    def work(i):
        barrier.wait()
        return i * 10

    def consume(i, value):
        assert threading.get_ident() == consumer_thread
        received[i] = value

    with GeometryQueue(2, 2, consume) as queue:
        for i in range(8):
            queue.submit(i, work, i)
    assert received == {i: i * 10 for i in range(8)}
    assert queue.stats["peak_pending_pairs"] == 2
    assert not queue.pending


@pytest.mark.parametrize("workers", [1, 2])
def test_geometry_errors_are_not_zero_matches(workers):
    def fail():
        raise ValueError("pair (2, 3) invalid")

    with pytest.raises(RuntimeError, match=r"task 17.*pair \(2, 3\)"):
        with GeometryQueue(workers, 2, lambda *_: None) as queue:
            queue.submit(17, fail)
    assert not any(t.name.startswith("loma-geometry") for t in threading.enumerate())


@pytest.mark.parametrize(
    "override,error",
    [
        ({"match_batch_size": 0}, "match_batch_size"),
        ({"extract_batch_size": -1}, "extract_batch_size"),
        ({"preprocess_workers": -1}, "preprocess_workers"),
        ({"geometry_workers": 0}, "geometry_workers"),
        ({"feature_cache": "disk"}, "feature_cache"),
        ({"feature_cache": "cuda"}, "CUDA device"),
    ],
)
def test_invalid_execution_settings(override, error):
    values = dict(
        device="cpu",
        match_batch_size=1,
        extract_batch_size=1,
        preprocess_workers=0,
        geometry_workers=1,
        feature_cache="cpu",
    )
    values.update(override)
    with pytest.raises(ValueError, match=error):
        validate_execution(**values)


class BatchBackend:
    metadata = {"test_backend": True}

    def __init__(self):
        rng = np.random.default_rng(19)
        points = rng.uniform([-1.2, -0.8, 4], [1.2, 0.8, 8], (100, 3))
        self.intrinsic = np.array(
            [[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float
        )
        self.features = []
        for i, count in enumerate((100, 75, 100, 100, 0)):
            projected = (points[:count] - [0.35 * i, 0, 0]) @ self.intrinsic.T
            self.features.append(
                {
                    "keypoints": (projected[:, :2] / projected[:, 2:]).astype(
                        np.float32
                    ),
                    "image_size_hw": (480, 640),
                }
            )
        self.batches = []
        self.closed = False

    def synchronize(self):
        pass

    def extract(self, path):
        return self.features[int(Path(path).stem)]

    def extract_batched(self, paths, batch_size, workers):
        for i in (2, 0, 1, 4, 3):
            yield i, self.features[i]

    def match(self, left, right):
        ids = np.arange(
            min(len(left["keypoints"]), len(right["keypoints"])), dtype=np.uint32
        )
        return np.column_stack((ids, ids)), np.ones(len(ids))

    def match_batch(self, pairs):
        sizes = [(len(a["keypoints"]), len(b["keypoints"])) for a, b in pairs]
        assert len(set(sizes)) == 1  # no padding or heterogeneous feature counts
        self.batches.append(len(pairs))
        return [self.match(a, b) for a, b in pairs]

    def close(self):
        self.closed = True


def run_backend(backend, **settings):
    return loma.run_loma_prior(
        [f"{i}.png" for i in range(5)],
        (480, 640),
        [backend.intrinsic] * 5,
        [
            {"pair": [i, j], "sift_support": "untried"}
            for i in range(5)
            for j in range(i + 1, 5)
        ],
        device="cpu",
        backend=backend,
        **settings,
    )


def test_batched_execution_real_geometry_restores_ids_empty_pairs_and_tails():
    pytest.importorskip("pycolmap")
    serial = run_backend(BatchBackend())
    backend = BatchBackend()
    parallel = run_backend(
        backend,
        match_batch_size=2,
        extract_batch_size=2,
        preprocess_workers=2,
        geometry_workers=2,
    )
    assert 1 in backend.batches and 2 in backend.batches
    assert serial.pair_records == parallel.pair_records
    assert list(serial.geometries) == list(parallel.geometries)
    for pair in serial.geometries:
        np.testing.assert_array_equal(
            serial.geometries[pair].inlier_matches,
            parallel.geometries[pair].inlier_matches,
        )
    for a, b in zip(serial.keypoints, parallel.keypoints):
        np.testing.assert_array_equal(a, b)
    assert parallel.stats["execution"]["geometry"]["peak_pending_pairs"] <= 4
    assert parallel.stats["timing"]["schema_version"] == 2
    assert len([r for r in parallel.pair_records if r["raw_matches"] == 0]) == 4
    json.dumps(parallel.stats)
    assert not backend.closed  # injected backend belongs to the caller


def test_matching_failure_closes_owned_backend_without_retry(monkeypatch):
    pytest.importorskip("pycolmap")
    backend = BatchBackend()
    monkeypatch.setattr(loma, "LoMaBackend", lambda *_, **__: backend)

    def fail(_):
        backend.batches.append("failed")
        raise RuntimeError("CUDA out of memory")

    backend.match_batch = fail
    with pytest.raises(RuntimeError, match="cache_bytes=.*CUDA out of memory"):
        loma.run_loma_prior(
            ["0.png", "1.png"],
            (480, 640),
            [backend.intrinsic] * 2,
            [{"pair": [0, 1], "sift_support": "untried"}],
            device="cpu",
            match_batch_size=2,
        )
    assert backend.closed and backend.batches == ["failed"]
