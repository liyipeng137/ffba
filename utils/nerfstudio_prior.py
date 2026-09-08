"""Compatibility import; the implementation lives in ffba.initialization.prior_pose."""
import sys
import ffba.initialization.prior_pose as implementation
sys.modules[__name__] = implementation
