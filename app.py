from __future__ import annotations

import html
import hashlib
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from campus_compass.planner import apply_plan_change, build_study_plan
from campus_compass.rag import delete_source, ingest_file, ingest_text, list_sources, source_preview
from campus_compass.workflow import answer_question, compare_responses, provider_status

load_dotenv(override=True)
sample_handbook = Path(__file__).parent / "data" / "sample_academic_regulations.txt"
if not list_sources() and sample_handbook.exists():
    ingest_text(sample_handbook.read_text(encoding="utf-8"), sample_handbook.name)
st.set_page_config(page_title="Campus Compass", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { color-scheme:light; --ink:#172a2a; --muted:#6e7e79; --paper:#f5f6f1; --line:#e2e7de; --green:#1d6b59; --deep:#123f36; --lime:#c7e26b; --coral:#ed765a; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
    .stApp, [data-testid="stAppViewContainer"] { background:var(--paper); color:var(--ink); }
    [data-testid="stAppViewContainer"] .main .block-container { max-width:1180px; padding:2.1rem 2.35rem 5rem; }
    header[data-testid="stHeader"] { background:var(--paper); }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stMarkdownContainer"] strong,
    .stApp label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color:var(--ink)!important; }
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p { color:var(--muted)!important; }
    [data-testid="stSidebar"] { background:#edf0e8; border-right:1px solid var(--line); color:var(--ink); min-width:260px!important; }
    [data-testid="stSidebar"] > div:first-child { padding:1.35rem .8rem 1.5rem; }
    h1,h2,h3 { font-family:'Manrope',sans-serif!important; letter-spacing:0!important; color:var(--ink); }
    .brand { display:flex; align-items:center; font:800 1.14rem 'Manrope',sans-serif; letter-spacing:0; color:var(--ink); }
    .brand-mark { display:inline-grid; place-items:center; width:31px; height:31px; margin-right:9px; flex:none; color:#fff; background:var(--green); border-radius:8px; }
    .sidebar-section { margin:1.1rem 0 .35rem; font:500 .68rem 'DM Mono',monospace; letter-spacing:0; color:var(--muted); text-transform:uppercase; }
    [data-testid="stRadio"] [role="radiogroup"] { gap:.18rem; }
    [data-testid="stRadio"] label { padding:.35rem .48rem; border-radius:5px; }
    [data-testid="stRadio"] label:has(input:checked) { background:#dfe9df; }
    [data-testid="stRadio"] label p { font-weight:500; }
    .eyebrow { font:500 .72rem 'DM Mono',monospace; letter-spacing:0; text-transform:uppercase; color:var(--green); }
    .page-title { font:800 2rem 'Manrope',sans-serif; letter-spacing:0; line-height:1.15; margin:.35rem 0 .35rem; }
    .page-sub { color:var(--muted); font-size:.93rem; margin-bottom:1.4rem; }
    .welcome-band { position:relative; overflow:hidden; display:grid; grid-template-columns:minmax(0,1fr) auto; align-items:end; gap:1.5rem; padding:1.7rem 1.85rem; margin:.5rem 0 1.15rem; border-radius:8px; background:var(--deep); color:#f5f6f1; }
    .welcome-band:after { position:absolute; z-index:0; right:12%; top:-42px; width:190px; height:190px; border:1px solid rgba(199,226,107,.18); border-radius:50%; content:""; }
    .welcome-band > * { position:relative; z-index:1; }
    .welcome-kicker { color:var(--lime); font:500 .7rem 'DM Mono',monospace; text-transform:uppercase; }
    .welcome-title { max-width:650px; margin:.6rem 0 .5rem; color:#fff; font:700 1.75rem 'Manrope',sans-serif; line-height:1.2; }
    .stApp [data-testid="stMarkdownContainer"] .welcome-band .welcome-copy { max-width:640px; margin:0; color:#d9e5df!important; font-size:.92rem; }
    .stApp [data-testid="stMarkdownContainer"] .welcome-band .welcome-stat strong { color:var(--lime)!important; }
    .welcome-stat { position:relative; z-index:1; min-width:140px; padding:.8rem 0 .15rem 1.2rem; border-left:1px solid rgba(255,255,255,.22); }
    .welcome-stat strong { display:block; color:var(--lime); font:700 1.7rem 'Manrope',sans-serif; }
    .welcome-stat span { color:#d9e5df; font-size:.76rem; }
    .section-title { margin:1.15rem 0 .15rem; font:700 1.05rem 'Manrope',sans-serif; }
    .section-note { margin:.1rem 0 .7rem; color:var(--muted); font-size:.82rem; }
    .exam-row,.resource-row { display:grid; grid-template-columns:minmax(0,1fr) auto; align-items:center; gap:.5rem; padding:.72rem 0; border-bottom:1px solid var(--line); }
    .exam-name,.resource-name { color:var(--ink); font-weight:600; font-size:.87rem; overflow-wrap:anywhere; }
    .exam-date,.resource-meta { margin-top:.15rem; color:var(--muted); font-size:.76rem; }
    .exam-countdown { white-space:nowrap; color:var(--green); font:500 .7rem 'DM Mono',monospace; }
    .resource-count { white-space:nowrap; color:var(--green); font:500 .72rem 'DM Mono',monospace; }
    .dashboard-actions [data-testid="stButton"] button { min-height:3.2rem; text-align:left; }
    .source-chip { display:inline-block; padding:4px 8px; margin:3px 4px 3px 0; border:1px solid var(--line); border-radius:5px; font: .71rem 'DM Mono',monospace; color:var(--green); background:#fff; }
    .stButton button[kind="primary"] { background:var(--green); border-color:var(--green); color:#fff!important; }
    .stButton button[kind="secondary"] { background:#fff; border-color:var(--line); color:var(--ink)!important; }
    .stButton button { border-radius:5px; }
    [data-testid="stChatInput"] { background:#fff; border-color:var(--line); }
    [data-testid="stChatInput"] textarea { background:#fff; color:var(--ink)!important; }
    [data-testid="stChatInputSubmitButton"] { color:#fff!important; background:var(--green)!important; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:6px; padding:14px; }
    [data-testid="stChatMessage"] { border:1px solid var(--line); border-radius:6px; background:#fff; }
    div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:6px; }
    .footnote { font-size:.76rem; color:var(--muted); }
        @media (max-width:700px) {
            [data-testid="stAppViewContainer"] .main .block-container { padding:1.2rem 1rem 4rem; }
            .welcome-band { grid-template-columns:1fr; padding:1.3rem; }
            .welcome-title { font-size:1.4rem; }
            .welcome-stat { min-width:0; padding:.65rem 0 0; border-left:0; border-top:1px solid rgba(255,255,255,.22); }
            .page-title { font-size:1.65rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "workspace" not in st.session_state:
    st.session_state.workspace = "Overview"
if "indexed_uploads" not in st.session_state:
    st.session_state.indexed_uploads = set()
if "study_plan" not in st.session_state:
    st.session_state.study_plan = []
if "subjects" not in st.session_state:
    st.session_state.subjects = pd.DataFrame(
        [
            {"Subject": "Data Structures", "Exam date": date.today() + timedelta(days=14), "Priority": 3},
            {"Subject": "Database Systems", "Exam date": date.today() + timedelta(days=21), "Priority": 2},
        ]
    )


def page_heading(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="eyebrow">{eyebrow}</div><div class="page-title">{title}</div><div class="page-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def route_to(workspace: str, question: str | None = None) -> None:
    st.session_state.workspace = workspace
    if question:
        st.session_state.pending_question = question


with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">◈</span>Campus Compass</div>', unsafe_allow_html=True)
    st.caption("STUDENT WORKSPACE")
    st.divider()
    mode = st.radio("Workspace", ["Overview", "Ask Compass", "Study Plan", "Documents", "Compare answers"], key="workspace", label_visibility="collapsed")
    st.divider()
    sources = list_sources()
    chunk_count = sum(int(item["chunks"]) for item in sources)
    st.markdown('<div class="sidebar-section">Knowledge base</div>', unsafe_allow_html=True)
    st.markdown(f"`{len(sources):02}` documents · `{chunk_count:02}` passages")
    st.markdown('<div class="sidebar-section">Answer engine</div>', unsafe_allow_html=True)
    status = provider_status()
    st.markdown(f"● {status}" if status.endswith("Free models") else f"○ {status}")
    st.markdown('<div class="footnote">Answers are checked against your college documents when available.</div>', unsafe_allow_html=True)
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if mode == "Overview":
    today = date.today()
    subject_rows = []
    for _, subject in st.session_state.subjects.dropna(subset=["Subject", "Exam date"]).iterrows():
        exam_day = pd.to_datetime(subject["Exam date"]).date()
        if exam_day >= today:
            subject_rows.append((str(subject["Subject"]), exam_day, (exam_day - today).days))
    subject_rows.sort(key=lambda item: item[1])
    next_exam = subject_rows[0] if subject_rows else None
    page_heading(today.strftime("%A, %B %d").upper() + " · STUDENT WORKSPACE", "Your academic workspace.", "Your college resources and study plan, together in one place.")
    st.markdown(
        f'<section class="welcome-band"><div><div class="welcome-kicker">CAMPUS COMPASS · READY WHEN YOU ARE</div><div class="welcome-title">Get a clear answer, grounded in your college resources.</div><p class="welcome-copy">Ask a question, find the source, and keep moving with a plan that fits your deadlines.</p></div><div class="welcome-stat"><strong>{len(sources):02}</strong><span>indexed resources</span></div></section>',
        unsafe_allow_html=True,
    )
    metrics = st.columns(4)
    metrics[0].metric("Resources", len(sources), help="Documents indexed in your private workspace")
    metrics[1].metric("Searchable passages", chunk_count)
    metrics[2].metric("Subjects tracked", len(subject_rows))
    metrics[3].metric("Next exam", f"{next_exam[2]} days" if next_exam else "Not set", help=next_exam[0] if next_exam else "Add an exam date in Study Plan")

    left, right = st.columns([1.08, .92], gap="large")
    with left:
        st.markdown('<div class="section-title">Pick up where you need help</div><div class="section-note">Common starting points from your academic workspace.</div>', unsafe_allow_html=True)
        action_col, action_col_two = st.columns(2, gap="small")
        with action_col:
            st.button("Ask about a college policy", key="home-policy", use_container_width=True, on_click=route_to, args=("Ask Compass", "What does my college policy say about "))
            st.button("Build an exam study plan", key="home-plan", use_container_width=True, on_click=route_to, args=("Study Plan",))
        with action_col_two:
            st.button("Study a topic from my notes", key="home-topic", use_container_width=True, on_click=route_to, args=("Ask Compass", "Explain this topic from my uploaded course material: "))
            st.button("Manage college resources", key="home-documents", use_container_width=True, on_click=route_to, args=("Documents",))
        st.markdown('<div class="section-title">Upcoming exams</div><div class="section-note">Your next deadlines, based on the dates you entered.</div>', unsafe_allow_html=True)
        if subject_rows:
            for subject_name, exam_day, days_left in subject_rows[:4]:
                st.markdown(
                    f'<div class="exam-row"><div><div class="exam-name">{html.escape(subject_name)}</div><div class="exam-date">{exam_day.strftime("%A, %B %d")}</div></div><div class="exam-countdown">{"TODAY" if days_left == 0 else f"IN {days_left} DAYS"}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Add your subjects and exam dates in Study Plan to see your next deadlines here.")
    with right:
        st.markdown('<div class="section-title">Your indexed resources</div><div class="section-note">Files are ready to retrieve in answers.</div>', unsafe_allow_html=True)
        if sources:
            for item in sources[:5]:
                source_name = str(item["source"])
                st.markdown(
                    f'<div class="resource-row"><div><div class="resource-name">{html.escape(source_name)}</div><div class="resource-meta">Available in Ask Compass</div></div><div class="resource-count">{int(item["chunks"])} PASSAGES</div></div>',
                    unsafe_allow_html=True,
                )
            if len(sources) > 5:
                st.caption(f"And {len(sources) - 5} more resources")
        else:
            st.info("Add your first syllabus, policy, or course guide in Documents.")
        st.button("Open document library", key="home-library", use_container_width=True, on_click=route_to, args=("Documents",))

elif mode == "Ask Compass":
    page_heading("ACADEMIC ASSISTANT · 01", "Ask Compass", "Answers are grounded in your indexed college documents and include source names when relevant.")
    if not st.session_state.messages:
        left, right = st.columns([1.15, .85], gap="large")
        with left:
            st.markdown('<div class="section-title">Ask about your college material</div><div class="section-note">Search across indexed syllabi, handbooks, and course notes. Follow up naturally in the same conversation.</div>', unsafe_allow_html=True)
            if sources:
                st.caption("Available sources: " + " · ".join(str(item["source"]) for item in sources[:3]))
            else:
                st.warning("No college resources are indexed yet. Add documents in the Documents workspace to get source-grounded answers.")
        with right:
            st.markdown('<div class="section-title">Suggested questions</div><div class="section-note">Choose one to start.</div>', unsafe_allow_html=True)
            for prompt in ["Define research and its objectives", "Summarize the examination guidelines", "Explain research methods vs methodology"]:
                if st.button(prompt, key=f"starter-{prompt}", use_container_width=True):
                    st.session_state.pending_question = prompt
                    st.rerun()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption("From your documents: " + " · ".join(message["sources"]))
    question = st.chat_input("Ask about a course, policy, or college process…")
    if st.session_state.get("pending_question"):
        question = st.session_state.pop("pending_question")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Checking your college resources…"):
                result = answer_question(question, st.session_state.messages[:-1])
            st.markdown(result["answer"])
            if result["sources"]:
                st.caption("From your documents: " + " · ".join(result["sources"]))
            if result["intent"] == "study_plan":
                st.info("Open Study Plan to enter your subjects, study time, and exam dates.")
        st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})

elif mode == "Study Plan":
    page_heading("PERSONAL STUDY PLANNER · 02", "Make time count.", "Build a realistic revision rhythm around your subjects, available hours, and exam dates. Then adjust every session to fit your week.")
    with st.form("study_profile"):
        st.markdown("### Your study window")
        hours = st.slider("Available study time per day", min_value=0.5, max_value=8.0, value=2.0, step=0.5, format="%.1f hr")
        edited_subjects = st.data_editor(
            st.session_state.subjects,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Subject": st.column_config.TextColumn(required=True),
                "Exam date": st.column_config.DateColumn(required=True, min_value=date.today()),
                "Priority": st.column_config.NumberColumn(min_value=1, max_value=3, step=1, help="1 = lower focus, 3 = highest focus"),
            },
            key="subjects_editor",
        )
        generate = st.form_submit_button("Build my study plan", type="primary")
    if generate:
        st.session_state.subjects = edited_subjects
        subjects = [
            {"name": row["Subject"], "exam_date": row["Exam date"].isoformat(), "priority": row["Priority"]}
            for _, row in edited_subjects.dropna(subset=["Subject", "Exam date"]).iterrows()
        ]
        st.session_state.study_plan = build_study_plan(subjects, hours)
    if st.session_state.study_plan:
        st.markdown("### Your plan")
        frame = pd.DataFrame(st.session_state.study_plan)
        frame = frame.rename(columns={"date": "Date", "subject": "Subject", "duration_min": "Minutes", "focus": "Focus", "status": "Status"})
        st.dataframe(frame, use_container_width=True, hide_index=True, column_config={"Date": st.column_config.DateColumn(format="ddd, MMM D")})
        with st.expander("Adjust a session"):
            plan = st.session_state.study_plan
            choice = st.selectbox("Session", range(len(plan)), format_func=lambda index: f"{plan[index]['date']} · {plan[index]['subject']} ({plan[index]['duration_min']} min)")
            change_col, action_col = st.columns([2, 1])
            with change_col:
                new_minutes = st.number_input("New session length", min_value=15, max_value=180, value=int(plan[choice]["duration_min"]), step=5)
            with action_col:
                st.write("")
                st.write("")
                if st.button("Update session", type="primary"):
                    replacement = {**plan[choice], "duration_min": new_minutes}
                    st.session_state.study_plan = apply_plan_change(plan, "replace", choice, replacement)
                    st.rerun()
            if st.button("Remove selected session"):
                st.session_state.study_plan = apply_plan_change(plan, "remove", choice)
                st.rerun()
        if st.button("Download plan as CSV"):
            st.download_button("Save CSV", pd.DataFrame(st.session_state.study_plan).to_csv(index=False), "campus-compass-study-plan.csv", "text/csv")
    else:
        st.info("Add at least one subject with a future exam date, then build your plan. Sundays are reserved for weekly recap when they fall in your study window.")

elif mode == "Documents":
    page_heading("COLLEGE KNOWLEDGE BASE · 03", "Your document library", "Add course and college resources. Each file is extracted, split into searchable passages, and stored locally for cited answers.")
    upload_col, tips_col = st.columns([1.2, 1], gap="large")
    with upload_col:
        st.markdown('<div class="section-title">Add resources</div><div class="section-note">Files are indexed as soon as you select them.</div>', unsafe_allow_html=True)
        uploads = st.file_uploader("Choose PDF, DOCX, TXT, or Markdown files", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True, key="resource_upload")
        if uploads:
            indexed_results = []
            errors = []
            pending_uploads = []
            for upload in uploads:
                content = upload.getvalue()
                signature = hashlib.sha256(upload.name.encode("utf-8") + content).hexdigest()
                if signature not in st.session_state.indexed_uploads:
                    pending_uploads.append((upload, content, signature))
            progress = st.progress(0, text="Preparing document indexing…") if pending_uploads else None
            for index, (upload, content, signature) in enumerate(pending_uploads, start=1):
                try:
                    chunk_total = ingest_file(upload.name, content)
                    indexed_results.append((upload.name, chunk_total))
                    st.session_state.indexed_uploads.add(signature)
                except Exception as error:
                    errors.append((upload.name, str(error)))
                progress.progress(index / len(pending_uploads), text=f"Processed {upload.name}")
            if progress:
                progress.empty()
            if indexed_results:
                summary = " · ".join(f"{name}: {count} passages" for name, count in indexed_results)
                st.success(f"Indexed {len(indexed_results)} file(s). {summary}")
                st.session_state.last_ingested_names = [name for name, _ in indexed_results]
            for name, error in errors:
                st.error(f"{name}: {error}")
        for source_name in st.session_state.get("last_ingested_names", []):
            with st.expander(f"Extraction preview · {source_name}", expanded=False):
                preview = source_preview(source_name)
                if preview:
                    st.write(preview[:1800])
                    st.caption("Preview from indexed text, not the original file view.")
                else:
                    st.info("Indexed passages are present, but no preview passage is available.")
    with tips_col:
        st.markdown("### Good documents to start with")
        st.markdown("Academic regulations\n\nCourse syllabi\n\nExamination guidelines\n\nInternship handbook\n\nStudent FAQs")
        st.caption("PDFs need a selectable text layer. Scanned image PDFs require OCR before their text can be indexed.")
    st.markdown("### Indexed resources")
    current_sources = list_sources()
    if current_sources:
        for item in current_sources:
            left, middle, right = st.columns([4, 1, 1])
            left.write(item["source"])
            middle.caption(f"{item['chunks']} passages")
            if right.button("Remove", key=f"remove-{item['source']}"):
                delete_source(str(item["source"]))
                st.rerun()
    else:
        st.info("No resources indexed yet. Add a college document above or load the sample pack from the project data folder.")

else:
    page_heading("RETRIEVAL CHECK · 04", "What changes when context is added?", "Compare a general model response with an answer grounded in your indexed college documents.")
    with st.form("compare_form"):
        comparison_question = st.text_input("Question", placeholder="For example: How many days before an exam must a student apply for accommodations?")
        compare = st.form_submit_button("Compare responses", type="primary")
    if compare and comparison_question:
        with st.spinner("Generating both responses…"):
            result = compare_responses(comparison_question)
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("### Basic LLM")
            st.markdown(result["basic"])
            st.caption("No college documents supplied to this response.")
        with right:
            st.markdown("### RAG assistant")
            st.markdown(result["rag"])
            if result["sources"]:
                st.caption("Retrieved sources: " + " · ".join(result["sources"]))
            else:
                st.caption("No matching college sources found.")
        st.markdown('<div class="footnote">Check any college policy with the current official document or the responsible office.</div>', unsafe_allow_html=True)
