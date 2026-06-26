"""Tests for the stub classifier."""

from streamsense.classifier import StubClassifier
from streamsense.schemas import Event, SentimentLabel


def test_positive():
    result = StubClassifier().predict(Event(text="I love it, works great"))
    assert result.sentiment is SentimentLabel.POSITIVE


def test_negative():
    result = StubClassifier().predict(Event(text="this is terrible"))
    assert result.sentiment is SentimentLabel.NEGATIVE


def test_neutral():
    result = StubClassifier().predict(Event(text="it arrived on tuesday"))
    assert result.sentiment is SentimentLabel.NEUTRAL


def test_confidence_in_range():
    result = StubClassifier().predict(Event(text="anything"))
    assert 0.0 <= result.confidence <= 1.0