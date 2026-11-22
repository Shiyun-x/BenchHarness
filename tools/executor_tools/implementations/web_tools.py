"""Backward-compatible import path for structured web evidence retrieval."""
try:
    from tools.executor_tools.implementations.web_evidence import web_search
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools.executor_tools.implementations.web_evidence import web_search

__all__ = ['web_search']
