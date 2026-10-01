"""LangChain prompts and the LangGraph assistant workflow."""

from __future__ import annotations

import os
import re
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import OpenAIAuthenticationError
from langgraph.graph import END, START, StateGraph

from campus_compass.rag import STOP_WORDS, retrieve
from campus_compass.tools import calculator


ACADEMIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """You are Campus Compass, a careful college academic assistant.
Answer using only the supplied college context. If it does not contain the answer, say that the college knowledge base does not have that information and suggest the relevant office or document to check. Cite sources inline as [source filename]. Never invent policy, deadlines, or course facts. Be concise and student-friendly."""),
        ("human", "Conversation so far:\n{history}\n\nCollege context:\n{context}\n\nStudent question: {question}"),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Summarize the supplied college material faithfully. Preserve dates, requirements, exceptions, and source names. Do not add facts."),
        ("human", "Material:\n{context}\n\nStudent request: {question}"),
    ]
)

STUDY_PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Create practical study-plan guidance based on the student's subjects, available time, and exam dates. Prefer active recall, spaced repetition, and practice questions. Do not invent exam requirements."),
        ("human", "Student details:\n{profile}\n\nCollege context:\n{context}"),
    ]
)


class AssistantState(TypedDict, total=False):
    question: str
    history: str
    mode: str
    profile: str
    intent: str
    documents: list[Document]
    answer: str
    sources: list[str]
    reviewed: bool


class ProviderConfigurationError(ValueError):
    """Raised when configured model credentials cannot be used safely."""


def _free_model_name() -> str:
    model_name = os.getenv("CAMPUS_MODEL", "openrouter/free").strip()
    if model_name != "openrouter/free" and not model_name.endswith(":free"):
        raise ProviderConfigurationError(
            "CAMPUS_MODEL must be `openrouter/free` or an OpenRouter model ending in `:free`. Paid models are disabled."
        )
    return model_name


def provider_status() -> str:
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not openrouter_key:
        return "OpenRouter · Add API key"
    if not openrouter_key.startswith("sk-or-v1-") or len(openrouter_key) <= len("sk-or-v1-"):
        return "OpenRouter · Check API key"
    try:
        _free_model_name()
    except ProviderConfigurationError:
        return "OpenRouter · Free models only"
    return "OpenRouter · Free models"


def _model() -> ChatOpenAI | None:
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not openrouter_key:
        return None
    if (
        not openrouter_key.startswith("sk-or-v1-")
        or len(openrouter_key) <= len("sk-or-v1-")
    ):
        raise ProviderConfigurationError(
            "The configured OpenRouter key does not look like an OpenRouter API key. Replace OPENROUTER_API_KEY in .env with a key beginning `sk-or-v1-`."
        )
    model_name = _free_model_name()
    options: dict[str, str | float] = {
        "model": model_name,
        "api_key": openrouter_key,
        "temperature": 0.2,
        "base_url": "https://openrouter.ai/api/v1",
    }
    return ChatOpenAI(**options)


def _history_text(history: list[dict[str, str]]) -> str:
    return "\n".join(f"{item['role'].title()}: {item['content']}" for item in history[-6:])


def _analyze(state: AssistantState) -> AssistantState:
    question = state["question"].strip()
    lowered = question.lower()
    if any(word in lowered for word in ("calculate", "what is", "compute")) and re.search(r"\d.*[+*/%-]|[+*/%-].*\d", lowered):
        intent = "calculation"
    elif any(word in lowered for word in ("summarize", "summary", "in brief")):
        intent = "summary"
    elif any(word in lowered for word in ("study plan", "study schedule", "revise for", "revision plan")):
        intent = "study_plan"
    else:
        intent = "academic_question"
    return {**state, "question": question, "intent": intent}


def _retrieve(state: AssistantState) -> AssistantState:
    previous_questions = re.findall(r"(?:^|\n)User: ([^\n]+)", state.get("history", ""))
    retrieval_query = " ".join([*previous_questions[-1:], state["question"]])
    documents = retrieve(retrieval_query)
    return {**state, "documents": documents, "sources": list(dict.fromkeys(doc.metadata["source"] for doc in documents))}


def _local_answer(state: AssistantState) -> str:
    documents = state.get("documents", [])
    if not documents:
        return "I couldn't find relevant information in the college knowledge base. Try uploading the relevant syllabus or guideline, or contact the appropriate college office."
    question_terms = {
        word
        for word in re.findall(r"[a-z0-9]+", state["question"].lower())
        if word not in STOP_WORDS
    }
    useful_lines: list[str] = []
    for document in documents[:3]:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", document.page_content)
        ranked = sorted(
            ((len(question_terms.intersection(re.findall(r"[a-z0-9]+", sentence.lower()))), sentence.strip()) for sentence in sentences),
            reverse=True,
        )
        useful_lines.extend(sentence for score, sentence in ranked[:2] if score > 0 and sentence)
    if not useful_lines:
        return "I found potentially related material, but it does not clearly answer this question. Please check the source document or ask the relevant college office."
    citations = ", ".join(f"[{source}]" for source in state.get("sources", []))
    return "\n\n".join(dict.fromkeys(useful_lines[:2])) + (f"\n\nSources: {citations}" if citations else "")


def _generate(state: AssistantState) -> AssistantState:
    if state.get("intent") == "calculation":
        expression = re.sub(r"^(what is|calculate|compute)\s+", "", state["question"], flags=re.IGNORECASE).rstrip("? ")
        return {**state, "answer": f"The result is **{calculator.invoke({'expression': expression})}**."}

    try:
        model = _model()
    except ProviderConfigurationError as error:
        return {
            **state,
            "answer": f"{error}\n\nHere is the available document text instead:\n\n{_local_answer(state)}",
        }
    documents = state.get("documents", [])
    if model is None:
        return {**state, "answer": _local_answer(state)}

    context = "\n\n".join(
        f"[{document.metadata['source']}]\n{document.page_content}" for document in documents
    ) or "No relevant college documents were found."
    if state.get("intent") == "study_plan":
        if not state.get("profile"):
            return {
                **state,
                "answer": "I can help shape a study plan. Add your subjects, available study time, and exam dates in the Study Plan workspace first.",
            }
        prompt = STUDY_PLAN_PROMPT
    else:
        prompt = SUMMARY_PROMPT if state.get("intent") == "summary" else ACADEMIC_PROMPT
    messages = prompt.format_messages(
        history=state.get("history", ""),
        context=context,
        question=state["question"],
        profile=state.get("profile", ""),
    )
    try:
        response = model.invoke(messages)
    except OpenAIAuthenticationError:
        local_answer = _local_answer(state)
        return {
            **state,
            "answer": "OpenRouter rejected the configured API key. Replace OPENROUTER_API_KEY in .env with an active OpenRouter key. The free model could not be reached; here is the matching document text instead.\n\n" + local_answer,
        }
    return {**state, "answer": str(response.content)}


def _review(state: AssistantState) -> AssistantState:
    answer = state.get("answer", "")
    if not answer.strip():
        answer = "I couldn't prepare a response. Please try rephrasing your question."
    if state.get("intent") != "calculation" and not state.get("documents") and "couldn't find" not in answer.lower():
        answer = "I couldn't verify this against the college knowledge base. Please confirm with the appropriate college office before relying on it.\n\n" + answer
    return {**state, "answer": answer, "reviewed": True}


def build_graph():
    builder = StateGraph(AssistantState)
    builder.add_node("question_analysis", _analyze)
    builder.add_node("information_retrieval", _retrieve)
    builder.add_node("response_generation", _generate)
    builder.add_node("response_review", _review)
    builder.add_edge(START, "question_analysis")
    builder.add_edge("question_analysis", "information_retrieval")
    builder.add_edge("information_retrieval", "response_generation")
    builder.add_edge("response_generation", "response_review")
    builder.add_edge("response_review", END)
    return builder.compile()


assistant_graph = build_graph()


def answer_question(
    question: str,
    history: list[dict[str, str]] | None = None,
    profile: str = "",
) -> dict[str, object]:
    result = assistant_graph.invoke(
        {
            "question": question,
            "history": _history_text(history or []),
            "profile": profile,
        }
    )
    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "intent": result.get("intent", "academic_question"),
        "reviewed": result.get("reviewed", False),
    }


def compare_responses(question: str) -> dict[str, object]:
    documents = retrieve(question)
    try:
        model = _model()
    except ProviderConfigurationError as error:
        model = None
        basic = f"{error} A live model comparison is unavailable until the key is corrected."
        rag_answer = _local_answer({"question": question, "documents": documents, "sources": list(dict.fromkeys(doc.metadata["source"] for doc in documents))})
    else:
        basic = ""
        rag_answer = ""
    if model is None and not basic:
        basic = "No OpenRouter key is configured. Add OPENROUTER_API_KEY to use the free-model router."
        rag_answer = _local_answer({"question": question, "documents": documents, "sources": list(dict.fromkeys(doc.metadata["source"] for doc in documents))})
    else:
        if model is not None:
            try:
                basic = str(model.invoke(question).content)
                rag_answer = answer_question(question)["answer"]
            except OpenAIAuthenticationError:
                basic = "OpenRouter rejected the configured API key. Replace OPENROUTER_API_KEY in .env with an active OpenRouter key."
                rag_answer = _local_answer({"question": question, "documents": documents, "sources": list(dict.fromkeys(doc.metadata["source"] for doc in documents))})
    return {
        "basic": basic,
        "rag": rag_answer,
        "sources": list(dict.fromkeys(doc.metadata["source"] for doc in documents)),
    }