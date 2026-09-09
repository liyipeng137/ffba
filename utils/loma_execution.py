"""Compatibility import; the implementation lives in ffba.matching.loma_execution."""
import sys
import ffba.matching.loma_execution as implementation
sys.modules[__name__] = implementation
