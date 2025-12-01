"""Lightweight settings for optional features."""
from __future__ import annotations

import os

PIPELINE_METRICS_ENABLED: bool = str(
    os.getenv("PIPELINE_METRICS_ENABLED", "true")
).strip().lower() in {"1", "true", "yes", "on"}

BIOTECH_PEER_REQUIRED_FOR_VALIDATION: bool = str(
    os.getenv("BIOTECH_PEER_REQUIRED_FOR_VALIDATION", "false")
).strip().lower() in {"1", "true", "yes", "on"}

__all__ = ["PIPELINE_METRICS_ENABLED", "BIOTECH_PEER_REQUIRED_FOR_VALIDATION"]
