from src.data_pipeline.prompt_exporter import clean_prompt_document, dedupe_samples


def test_clean_prompt_document_keeps_training_fields_only():
    document = {
        "_id": "ignored",
        "userMessage": " hello ",
        "assistantMessage": " Hi there. ",
        "liked": False,
        "context": {"page": "general"},
    }

    assert clean_prompt_document(document) == {
        "userMessage": "hello",
        "assistantMessage": "Hi there.",
        "liked": False,
    }


def test_clean_prompt_document_rejects_empty_messages():
    assert clean_prompt_document({"userMessage": "", "assistantMessage": "hello"}) is None
    assert clean_prompt_document({"userMessage": "hello", "assistantMessage": ""}) is None


def test_dedupe_samples_removes_duplicate_conversations():
    samples = [
        {"userMessage": "hello", "assistantMessage": "hi", "liked": False},
        {"userMessage": "hello", "assistantMessage": "hi", "liked": True},
        {"userMessage": "next", "assistantMessage": "answer", "liked": False},
    ]

    assert dedupe_samples(samples) == [
        {"userMessage": "hello", "assistantMessage": "hi", "liked": False},
        {"userMessage": "next", "assistantMessage": "answer", "liked": False},
    ]
