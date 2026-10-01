# Campus Compass

Campus Compass is a student-facing academic assistant built with Streamlit, LangChain, and LangGraph. It retrieves college guidance from uploaded documents, maintains short conversational context for follow-up questions, generates editable study schedules, and compares general-model answers with document-grounded responses.

## Run locally

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app starts in local demo mode without credentials and automatically indexes `data/sample_academic_regulations.txt`. Upload your own PDF, DOCX, TXT, or Markdown resources in the Documents workspace. PDF files must contain selectable text; scanned PDFs need OCR first. The local demo uses deterministic hashed text vectors stored in SQLite and a lightweight extractive answer path; it does not download a model.

For live responses through OpenRouter, copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` to a valid OpenRouter key. Set `CAMPUS_MODEL` to `openrouter/free` or a model ID ending in `:free` (for example, `nvidia/nemotron-3-ultra-550b-a55b:free`). Paid model IDs are rejected. Requests use OpenRouter's endpoint at `https://openrouter.ai/api/v1`; the app does not read OpenAI credentials or custom endpoints. A valid OpenRouter API key is required, and free model availability or rate limits are controlled by OpenRouter. Keep credentials private and verify college policies against current official sources.

## Workspaces

- **Ask Compass**: academic questions, cited answers, conversation follow-ups, summaries, and a safe arithmetic calculator.
- **Study Plan**: create a schedule from subjects, daily availability, priority, and examination dates; edit or remove sessions and export CSV.
- **Documents**: ingest, inspect, and remove local college knowledge-base files.
- **Compare answers**: compare a general chat-model response with a RAG-grounded response and its retrieved sources. In local demo mode, the basic-model column says a model is not configured.

The LangGraph workflow in `campus_compass/workflow.py` separates question analysis, information retrieval, response generation, and response review. Its plan prompt is available for multi-step study-plan requests, while the editable schedule is generated deterministically from the student's entered dates and time budget.

## Checks

```bash
python -m pytest -q
```

Tests cover local retrieval, unknown-answer behavior, calculator restrictions, exam-date scheduling, and plan changes. The sample handbook is fictional demo material and must not be treated as a college policy.
