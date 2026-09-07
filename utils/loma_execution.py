"""CPU-only scheduling and preprocessing for the LoMa inference adapter."""

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import time

import numpy as np
from PIL import Image


def validate_execution(
    device,
    match_batch_size,
    extract_batch_size,
    preprocess_workers,
    geometry_workers,
    feature_cache,
):
    for name, value in (
        ("match_batch_size", match_batch_size),
        ("extract_batch_size", extract_batch_size),
        ("geometry_workers", geometry_workers),
    ):
        if value < 1:
            raise ValueError(f"loma_{name} must be >= 1")
    if preprocess_workers < 0:
        raise ValueError("loma_preprocess_workers must be >= 0")
    if feature_cache not in {"cpu", "cuda"}:
        raise ValueError("loma_feature_cache must be cpu or cuda")
    if feature_cache == "cuda" and str(device).split(":")[0] != "cuda":
        raise ValueError("loma_feature_cache=cuda requires a CUDA device")


def bucket_batches(items, key, batch_size):
    """Stable shape buckets; IDs travel with items, with no padded tail."""
    buckets = {}
    for item in items:
        buckets.setdefault(key(item), []).append(item)
    for values in buckets.values():
        for offset in range(0, len(values), batch_size):
            yield values[offset : offset + batch_size]


def detector_size(size_wh, resize=1024, keep_aspect_ratio=True):
    w, h = size_wh
    if not keep_aspect_ratio:
        return resize, resize
    scale = resize / max(w, h)
    return int((scale * w) // 8 * 8), int((scale * h) // 8 * 8)


def image_shape(path, resize=1024, keep_aspect_ratio=True):
    with Image.open(path) as image:
        return detector_size(image.size, resize, keep_aspect_ratio)


def prepare_image(path, resize=1024, keep_aspect_ratio=True):
    """Match native PIL path APIs including float64 division before float32."""
    start = time.perf_counter()
    try:
        with Image.open(path) as source:
            if source.mode == "I;16":
                raise NotImplementedError("Can't handle 16 bit images")
            rgb = source.convert("RGB")
            w, h = rgb.size
            detector = rgb.resize(detector_size(rgb.size, resize, keep_aspect_ratio))
            descriptor = rgb.resize((784, 784))
            return {
                "detector": (np.asarray(detector) / 255.0)
                .transpose(2, 0, 1)
                .astype(np.float32),
                "descriptor": (np.asarray(descriptor) / 255.0)
                .transpose(2, 0, 1)
                .astype(np.float32),
                "image_size_hw": (h, w),
                "seconds": time.perf_counter() - start,
            }
    except Exception as exc:
        raise RuntimeError(f"LoMa preprocessing failed for {path}: {exc}") from exc


def prefetch_batches(batches, prepare, workers, stats):
    """At most two batches in flight, including current CPU/GPU consumption."""
    stats.update(peak_pending_images=0, wait_seconds=0.0)
    if workers == 0:
        for batch in batches:
            stats["peak_pending_images"] = max(stats["peak_pending_images"], len(batch))
            yield [(item, prepare(item)) for item in batch]
        return
    executor = ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="loma-preprocess"
    )
    pending = deque()
    iterator = iter(batches)

    def fill():
        while len(pending) < 2:
            batch = next(iterator, None)
            if batch is None:
                break
            pending.append([(item, executor.submit(prepare, item)) for item in batch])
        stats["peak_pending_images"] = max(
            stats["peak_pending_images"], sum(map(len, pending))
        )

    try:
        fill()
        while pending:
            start = time.perf_counter()
            batch = pending[0]
            ready = [(item, future.result()) for item, future in batch]
            stats["wait_seconds"] += time.perf_counter() - start
            yield ready
            # Consumer has finished this batch before we schedule its replacement.
            pending.popleft()
            del ready, batch
            fill()
    finally:
        for batch in pending:
            for _, future in batch:
                future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


class GeometryQueue:
    """Bounded independent jobs; only the calling thread consumes results."""

    def __init__(self, workers, limit, consume):
        self.executor = (
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix="loma-geometry")
            if workers > 1
            else None
        )
        self.limit = limit
        self.consume = consume
        self.pending = {}
        self.stats = {
            "workers": workers,
            "executor": "thread" if workers > 1 else "serial",
            "max_pending_pairs": limit,
            "peak_pending_pairs": 0,
            "queue_wait_seconds": 0.0,
        }

    def __enter__(self):
        return self

    def collect(self, block=False):
        if not self.pending:
            return
        start = time.perf_counter()
        done = {future for future in self.pending if future.done()}
        if block and not done:
            done, _ = wait(self.pending, return_when=FIRST_COMPLETED)
        self.stats["queue_wait_seconds"] += time.perf_counter() - start
        for future in sorted(done, key=lambda f: self.pending[f]):
            key = self.pending.pop(future)
            try:
                self.consume(key, future.result())
            except Exception as exc:
                raise RuntimeError(f"LoMa geometry task {key} failed: {exc}") from exc

    def submit(self, key, function, *args):
        if self.executor is None:
            self.stats["peak_pending_pairs"] = max(self.stats["peak_pending_pairs"], 1)
            try:
                self.consume(key, function(*args))
            except Exception as exc:
                raise RuntimeError(f"LoMa geometry task {key} failed: {exc}") from exc
            return
        self.collect()
        while len(self.pending) >= self.limit:
            self.collect(block=True)
        self.pending[self.executor.submit(function, *args)] = key
        self.stats["peak_pending_pairs"] = max(
            self.stats["peak_pending_pairs"], len(self.pending)
        )

    def finish(self):
        while self.pending:
            self.collect(block=True)

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                self.finish()
        finally:
            for future in self.pending:
                future.cancel()
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
