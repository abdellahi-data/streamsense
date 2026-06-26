"""Data models for streamsense."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class SentimentLabel(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Event(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    text: str = Field(min_length=1)
    source: str = "unknown"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    label: SentimentLabel | None = None  # ground truth when replaying a labelled set


class SentimentResult(BaseModel):
    # Field descriptions are sent to the model as the output schema, so keep them clear.
    sentiment: SentimentLabel = Field(description="The predicted sentiment class.")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence, 0 to 1.")