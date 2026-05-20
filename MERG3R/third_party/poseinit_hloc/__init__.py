def run_poseinit(*args, **kwargs):
    from .pipeline import run_poseinit as _run_poseinit

    return _run_poseinit(*args, **kwargs)


def __getattr__(name):
    if name == "PoseInitResult":
        from .pipeline import PoseInitResult

        return PoseInitResult
    raise AttributeError(name)


__all__ = ["PoseInitResult", "run_poseinit"]
