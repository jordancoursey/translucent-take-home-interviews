"""Observability: append-only tracing of pipeline calls."""

from .tracing import TRACE_PATH, log_trace, read_traces

__all__ = ["TRACE_PATH", "log_trace", "read_traces"]
