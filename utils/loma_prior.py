"""Compatibility import; the implementation lives in ffba.matching.loma."""
import sys
import ffba.matching.loma as implementation
sys.modules[__name__] = implementation
