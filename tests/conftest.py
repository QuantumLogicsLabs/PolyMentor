"""Pytest collection config for PolyMentor unit tests."""

# Manual smoke scripts that hit localhost on import — not unit tests.
collect_ignore = [
    "test_api.py",
    "test_learning_comprehensive.py",
]
