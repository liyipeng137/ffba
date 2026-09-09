"""Compatibility import; the implementation lives in ffba.initialization.images."""
import sys
import ffba.initialization.images as implementation
sys.modules[__name__] = implementation
