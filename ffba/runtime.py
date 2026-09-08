"""runtime for the formal SIFT + prior + BAE pipeline."""

import contextlib
import os
import sys
from pathlib import Path


def _lazy_import_pycolmap():
    try:
        import pycolmap  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pycolmap is required. Run this script in the Gluemap environment."
        ) from exc
    return pycolmap


def _ensure_gluemap_imports():
    repo_root = Path(__file__).resolve().parents[1] / "third_party" / "gluemap"
    if not (repo_root / "gluemap").is_dir():
        raise FileNotFoundError(f"GlueMap directory not found: {repo_root}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import thirdparty.path_to_thirdparty  # noqa: F401, PLC0415


def debug(args, message):
    if args.debug_print:
        print(f"[MERG3R-REFINE] {message}", flush=True)


@contextlib.contextmanager
def suppress_native_stdio(enabled=True):
    if not enabled:
        yield
        return
    saved_fds = [os.dup(1), os.dup(2)]
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_fds[0], 1)
        os.dup2(saved_fds[1], 2)
        os.close(devnull)
        os.close(saved_fds[0])
        os.close(saved_fds[1])
