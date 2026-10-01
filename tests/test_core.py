from datetime import date

import pytest

from campus_compass import rag
from campus_compass.planner import apply_plan_change, build_study_plan
from campus_compass.tools import calculator
from campus_compass import workflow
from campus_compass.workflow import answer_question


def test_retrieval_finds_relevant_college_passage(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(rag, "DATA_DIR", tmp_path)
    monkeypatch.setattr(rag, "DATABASE_PATH", tmp_path / "test.sqlite3")
    rag.ingest_text(
        "Students must maintain at least 75 percent attendance in every course. "
        "Applications for examination accommodations go to the registrar.",
        "academic-regulations.txt",
    )

    results = rag.retrieve("minimum attendance requirement")

    assert results
    assert "75 percent attendance" in results[0].page_content
    assert results[0].metadata["source"] == "academic-regulations.txt"
    answer = answer_question("What is the attendance policy?")["answer"]
    assert "75 percent" in str(answer)
    assert "registrar confirms arrangements" not in str(answer).lower()


def test_unknown_question_is_honest(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(rag, "DATA_DIR", tmp_path)
    monkeypatch.setattr(rag, "DATABASE_PATH", tmp_path / "empty.sqlite3")

    result = answer_question("What is the campus observatory's opening date?")

    assert "couldn't find relevant information" in str(result["answer"]).lower()
    assert result["reviewed"] is True


def test_calculator_accepts_arithmetic_and_rejects_code():
    assert calculator.invoke({"expression": "(18 * 4) / 3"}) == "24"
    assert "couldn't evaluate" in calculator.invoke({"expression": "__import__('os').system('id')"})
    assert calculator.invoke({"expression": "1 / 0"}).startswith("I couldn't")


def test_study_plan_tracks_exam_dates_and_can_be_modified():
    today = date(2026, 10, 1)
    plan = build_study_plan(
        [{"name": "Biology", "exam_date": "2026-10-03", "priority": 3}],
        hours_per_day=1.5,
        start_date=today,
    )

    assert {entry["date"] for entry in plan} == {"2026-10-01", "2026-10-02", "2026-10-03"}
    changed = apply_plan_change(plan, "replace", 0, {**plan[0], "duration_min": 25})
    assert any(entry["duration_min"] == 25 for entry in changed)
    assert len(apply_plan_change(changed, "remove", 0)) == len(changed) - 1


def test_invalid_exam_dates_are_skipped():
    assert build_study_plan([{"name": "Unknown", "exam_date": "not-a-date"}], 2, date(2026, 10, 1)) == []


def test_only_openrouter_free_model_and_langgraph_nodes(monkeypatch):
    class RecordingChatModel:
        def __init__(self, **options):
            self.options = options

    monkeypatch.setattr(workflow, "ChatOpenAI", RecordingChatModel)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://must-not-be-used.invalid/v1")
    monkeypatch.setenv("CAMPUS_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

    model = workflow._model()

    assert model.options == {
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "api_key": "sk-or-v1-test-key",
        "temperature": 0.2,
        "base_url": "https://openrouter.ai/api/v1",
    }
    assert workflow.provider_status() == "OpenRouter · Free models"
    assert {
        "question_analysis",
        "information_retrieval",
        "response_generation",
        "response_review",
    }.issubset(workflow.assistant_graph.get_graph().nodes)


def test_paid_openrouter_model_is_rejected(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-key")
    monkeypatch.setenv("CAMPUS_MODEL", "provider/paid-model")

    assert workflow.provider_status() == "OpenRouter · Free models only"
    with pytest.raises(workflow.ProviderConfigurationError, match="Paid models are disabled"):
        workflow._model()


def test_openai_credentials_are_not_used_as_provider(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-enable-a-model")

    assert workflow._model() is None
    assert workflow.provider_status() == "OpenRouter · Add API key"


def test_malformed_openrouter_key_is_reported(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "invalid-short-key")

    assert workflow.provider_status() == "OpenRouter · Check API key"
    with pytest.raises(workflow.ProviderConfigurationError):
        workflow._model()


def test_malformed_key_returns_document_fallback_without_traceback(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "DATA_DIR", tmp_path)
    monkeypatch.setattr(rag, "DATABASE_PATH", tmp_path / "fallback.sqlite3")
    monkeypatch.setenv("OPENROUTER_API_KEY", "invalid-short-key")
    monkeypatch.setattr(
        workflow,
        "retrieve",
        lambda query, limit=5: [
            rag.Document(
                page_content="Research is a systematic inquiry into facts.",
                metadata={"source": "research-notes.pdf", "score": 1.0},
            )
        ],
    )

    response = answer_question("Define research")

    assert "Replace OPENROUTER_API_KEY" in str(response["answer"])
    assert "systematic inquiry" in str(response["answer"])
    assert response["reviewed"] is True
