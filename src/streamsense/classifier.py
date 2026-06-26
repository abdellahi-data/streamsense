"""Sentiment classification: Event in, SentimentResult out."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from streamsense.schemas import Event, SentimentLabel, SentimentResult

SYSTEM_PROMPT = (
    "You classify the sentiment of a short piece of text. "
    "Return positive, negative, or neutral, with a confidence from 0 to 1."
)


class Classifier(Protocol):
    """Anything that can turn an Event into a SentimentResult."""

    def predict(self, event: Event) -> SentimentResult: ...


class StubClassifier:
    """Offline classifier for local dev and tests. No model call."""

    def predict(self, event: Event) -> SentimentResult:
        text = event.text.lower()
        if any(w in text for w in ("love", "great", "good", "excellent")):
            return SentimentResult(sentiment=SentimentLabel.POSITIVE, confidence=0.9)
        if any(w in text for w in ("hate", "bad", "terrible", "awful")):
            return SentimentResult(sentiment=SentimentLabel.NEGATIVE, confidence=0.9)
        return SentimentResult(sentiment=SentimentLabel.NEUTRAL, confidence=0.5)


class BedrockClassifier:
    """Real classifier backed by Claude on Bedrock via LangChain."""

    def __init__(self, model: str, region: str = "us-east-1") -> None:
        from langchain_aws import ChatBedrockConverse

        llm = ChatBedrockConverse(model=model, region_name=region, temperature=0)
        self._model = llm.with_structured_output(SentimentResult)

    def predict(self, event: Event) -> SentimentResult:
        messages = [("system", SYSTEM_PROMPT), ("human", event.text)]
        return self._model.invoke(messages)


@lru_cache
def _default() -> Classifier:
    return StubClassifier()


def classify(event: Event) -> SentimentResult:
    """Classify one event with the default classifier."""
    return _default().predict(event)