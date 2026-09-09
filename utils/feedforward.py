"""Compatibility import; the implementation lives in ffba.initialization.feedforward."""
import sys
import ffba.initialization.feedforward as implementation
sys.modules[__name__] = implementation
