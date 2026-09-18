import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from html import escape


# -------------------------------------------------------------------
# Project import path
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import streamlit as st

from app.api.dependencies import (
    get_document_registry,
    get_ingestion_service,
    get_rag_service,
    get_services,
)

from app.services.evaluation_service import EvaluationService


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# Custom CSS
# -------------------------------------------------------------------

st.markdown(
    """
    <style>

    /* ============================================================
       GLOBAL
       ============================================================ */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(99, 102, 241, 0.12),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(14, 165, 233, 0.10),
                transparent 28%
            ),
            linear-gradient(
                135deg,
                #070b14 0%,
                #0b1020 45%,
                #080d18 100%
            );
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ============================================================
       ANIMATIONS
       ============================================================ */

    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(18px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
        }

        to {
            opacity: 1;
        }
    }

    @keyframes glow {
        0% {
            box-shadow:
                0 0 0 rgba(99, 102, 241, 0);
        }

        50% {
            box-shadow:
                0 0 30px rgba(99, 102, 241, 0.12);
        }

        100% {
            box-shadow:
                0 0 0 rgba(99, 102, 241, 0);
        }
    }


    /* ============================================================
       HERO
       ============================================================ */

    .hero {
        padding: 2.3rem 2.5rem;
        border-radius: 24px;
        margin-bottom: 2rem;

        background:
            linear-gradient(
                135deg,
                rgba(30, 41, 75, 0.82),
                rgba(15, 23, 42, 0.76)
            );

        border:
            1px solid rgba(148, 163, 184, 0.16);

        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.28),
            inset 0 1px 0 rgba(255, 255, 255, 0.04);

        backdrop-filter: blur(18px);

        animation:
            fadeUp 0.8s ease-out;
    }

    .hero-badge {
        display: inline-block;

        padding: 0.35rem 0.8rem;

        border-radius: 999px;

        background:
            rgba(99, 102, 241, 0.13);

        border:
            1px solid rgba(129, 140, 248, 0.24);

        color: #c7d2fe;

        font-size: 0.78rem;

        font-weight: 600;

        letter-spacing: 0.04em;

        margin-bottom: 0.9rem;
    }

    .hero-title {
        font-size: 3rem;

        font-weight: 800;

        line-height: 1.05;

        margin: 0;

        background:
            linear-gradient(
                90deg,
                #f8fafc,
                #c7d2fe,
                #bae6fd
            );

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        margin-top: 0.9rem;

        font-size: 1.05rem;

        line-height: 1.7;

        color: #94a3b8;

        max-width: 760px;
    }


    /* ============================================================
       SECTION HEADERS
       ============================================================ */

    .section-label {
        font-size: 0.78rem;

        text-transform: uppercase;

        letter-spacing: 0.12em;

        color: #818cf8;

        font-weight: 700;

        margin-bottom: 0.35rem;
    }

    .section-title {
        font-size: 1.65rem;

        font-weight: 750;

        color: #f8fafc;

        margin-bottom: 0.25rem;
    }

    .section-description {
        color: #94a3b8;

        margin-bottom: 1.2rem;
    }


    /* ============================================================
       METRIC CARDS
       ============================================================ */

    .metric-card {
        padding: 1.15rem;

        border-radius: 16px;

        background:
            rgba(15, 23, 42, 0.65);

        border:
            1px solid rgba(148, 163, 184, 0.13);

        text-align: center;

        animation:
            fadeUp 0.6s ease-out;
    }

    .metric-value {
        font-size: 1.7rem;

        font-weight: 750;

        color: #f8fafc;
    }

    .metric-label {
        font-size: 0.76rem;

        color: #64748b;

        margin-top: 0.2rem;

        text-transform: uppercase;

        letter-spacing: 0.08em;
    }


    /* ============================================================
       ANSWER CARD
       ============================================================ */

    .answer-card {
        padding: 1.6rem;

        border-radius: 18px;

        background:
            linear-gradient(
                135deg,
                rgba(30, 41, 75, 0.72),
                rgba(15, 23, 42, 0.78)
            );

        border:
            1px solid rgba(129, 140, 248, 0.18);

        box-shadow:
            0 18px 55px rgba(0, 0, 0, 0.24);

        animation:
            fadeUp 0.65s ease-out,
            glow 3s ease-in-out infinite;

        margin-top: 0.8rem;

        margin-bottom: 1.5rem;
    }

    .answer-label {
        color: #a5b4fc;

        font-size: 0.78rem;

        text-transform: uppercase;

        letter-spacing: 0.1em;

        font-weight: 700;

        margin-bottom: 1rem;
    }

    .answer-text {
        color: #e2e8f0;

        font-size: 1.03rem;

        line-height: 1.8;

        white-space: pre-wrap;
    }


    /* ============================================================
       SOURCE CARDS
       ============================================================ */

    .source-card {
        padding: 1rem 1.1rem;

        border-radius: 14px;

        background:
            rgba(15, 23, 42, 0.72);

        border:
            1px solid rgba(148, 163, 184, 0.13);

        margin-bottom: 0.7rem;

        transition:
            transform 0.2s ease,
            border-color 0.2s ease;

        animation:
            fadeUp 0.5s ease-out;
    }

    .source-card:hover {
        transform: translateY(-2px);

        border-color:
            rgba(129, 140, 248, 0.35);
    }

    .source-name {
        color: #e2e8f0;

        font-weight: 650;
    }

    .source-page {
        color: #94a3b8;

        font-size: 0.9rem;

        margin-top: 0.2rem;
    }


    /* ============================================================
       DOCUMENT CARDS
       ============================================================ */

    .document-card {
        padding: 0.9rem;

        border-radius: 13px;

        background:
            rgba(30, 41, 59, 0.52);

        border:
            1px solid rgba(148, 163, 184, 0.11);

        margin-bottom: 0.45rem;

        transition:
            transform 0.2s ease,
            background 0.2s ease;
    }

    .document-card:hover {
        transform: translateX(3px);

        background:
            rgba(51, 65, 85, 0.62);
    }

    .document-name {
        font-size: 0.88rem;

        font-weight: 650;

        color: #e2e8f0;

        word-break: break-word;
    }

    .document-meta {
        font-size: 0.76rem;

        color: #64748b;

        margin-top: 0.2rem;
    }


    /* ============================================================
       SCORECARD
       ============================================================ */

    .scorecard {
        padding: 1.2rem;

        border-radius: 18px;

        background:
            rgba(15, 23, 42, 0.70);

        border:
            1px solid rgba(148, 163, 184, 0.14);

        animation:
            fadeUp 0.6s ease-out;

        margin-bottom: 1.5rem;
    }

    .score-title {
        font-size: 0.78rem;

        color: #818cf8;

        text-transform: uppercase;

        letter-spacing: 0.1em;

        font-weight: 700;

        margin-bottom: 0.8rem;
    }


    /* ============================================================
       BUTTONS
       ============================================================ */

    .stButton > button {
        border-radius: 11px;

        font-weight: 650;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 8px 24px rgba(0, 0, 0, 0.25);
    }


    /* ============================================================
       INPUTS
       ============================================================ */

    textarea,
    input {
        border-radius: 12px !important;
    }

    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
    }


    /* ============================================================
       SIDEBAR
       ============================================================ */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #080c16,
                #0b1020
            );

        border-right:
            1px solid rgba(148, 163, 184, 0.10);
    }


    /* ============================================================
       FOOTER
       ============================================================ */

    .footer {
        text-align: center;

        color: #475569;

        font-size: 0.78rem;

        padding-top: 1.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Services
# -------------------------------------------------------------------

registry = get_document_registry()

documents = registry.list_documents()

evaluation_service = EvaluationService()


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:1.35rem;
            font-weight:750;
            margin-bottom:1rem;
        ">
            🔬 Research Library
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.metric(
        "Indexed papers",
        len(documents),
    )

    st.divider()

    if documents:

        st.markdown(
            "### 📚 Your Papers"
        )

        for index, document in enumerate(
            documents
        ):

            source = document["source"]

            pages = document["pages"]

            chunks = document["chunks"]

            st.markdown(
                f"""
                <div class="document-card">

                    <div class="document-name">
                        📄 {escape(source)}
                    </div>

                    <div class="document-meta">
                        {pages} pages · {chunks} chunks
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                "🗑️ Remove",
                key=f"remove_{index}",
                use_container_width=True,
            ):

                try:

                    services = get_services()

                    removed_chunks = (
                        services.vector_store.remove_source(
                            source
                        )
                    )

                    registry.remove(
                        source
                    )

                    st.success(
                        f"Removed {source}"
                    )

                    st.caption(
                        f"{removed_chunks} indexed passages removed."
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        f"Failed to remove document: {exc}"
                    )

    else:

        st.info(
            "Your research library is empty."
        )

    st.divider()

    st.caption(
        "Local AI · FAISS · Ollama"
    )


# -------------------------------------------------------------------
# Hero
# -------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            ✦ PRIVATE LOCAL RESEARCH ASSISTANT
        </div>

        <div class="hero-title">
            AI Research Agent
        </div>

        <div class="hero-subtitle">
            Search your research papers with semantic retrieval
            and generate grounded answers using a local AI model.
            Every answer can be traced back to the source document
            and page.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Dashboard metrics
# -------------------------------------------------------------------

total_pages = sum(
    document["pages"]
    for document in documents
)

total_chunks = sum(
    document["chunks"]
    for document in documents
)

metric_columns = st.columns(3)


with metric_columns[0]:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-value">
                {len(documents)}
            </div>

            <div class="metric-label">
                Papers
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with metric_columns[1]:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-value">
                {total_pages}
            </div>

            <div class="metric-label">
                Pages
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with metric_columns[2]:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-value">
                {total_chunks}
            </div>

            <div class="metric-label">
                Indexed passages
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    "<br>",
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Upload section
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-label">Knowledge Base</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Add Research Papers</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Upload PDFs and build your searchable research library."
    "</div>",
    unsafe_allow_html=True,
)


upload_column, info_column = st.columns(
    [2, 1]
)


with upload_column:

    uploaded_file = st.file_uploader(
        "Choose a research paper",
        type=["pdf"],
        label_visibility="collapsed",
    )


with info_column:

    st.info(
        "PDF files are processed locally and indexed "
        "for semantic search."
    )


if uploaded_file is not None:

    filename = uploaded_file.name

    if registry.exists(filename):

        st.warning(
            f"'{filename}' is already indexed."
        )

    elif st.button(
        "🚀 Index Research Paper",
        type="primary",
        use_container_width=True,
    ):

        temporary_path = None

        try:

            with NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as temporary_file:

                temporary_file.write(
                    uploaded_file.getbuffer()
                )

                temporary_path = temporary_file.name

            ingestion_service = (
                get_ingestion_service()
            )

            with st.spinner(
                "Analyzing paper · creating embeddings · indexing..."
            ):

                chunks_indexed = (
                    ingestion_service.ingest(
                        temporary_path,
                        source_name=filename,
                    )
                )

            pages = (
                ingestion_service.document_loader.load(
                    temporary_path
                )
            )

            pages_processed = sum(
                1
                for page in pages
                if page.get(
                    "text",
                    "",
                ).strip()
            )

            registry.add(
                source=filename,
                pages=pages_processed,
                chunks=chunks_indexed,
            )

            st.success(
                f"✓ '{filename}' has been added to your research library."
            )

            st.info(
                f"{pages_processed} pages · "
                f"{chunks_indexed} passages indexed"
            )

            st.rerun()

        except ValueError as exc:

            st.error(
                str(exc)
            )

        except Exception as exc:

            st.error(
                f"Failed to index document: {exc}"
            )

        finally:

            if temporary_path:

                Path(
                    temporary_path
                ).unlink(
                    missing_ok=True
                )


# -------------------------------------------------------------------
# Research workspace
# -------------------------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-label">Research Workspace</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Ask Your Research</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Ask questions about your indexed research papers."
    "</div>",
    unsafe_allow_html=True,
)


if not documents:

    st.warning(
        "Upload a research paper above to start asking questions."
    )

else:

    selected_documents = st.multiselect(
        "Research scope",
        options=[
            document["source"]
            for document in documents
        ],
        default=[
            document["source"]
            for document in documents
        ],
        help=(
            "Choose one or more papers. "
            "Multiple selections allow cross-paper research."
        ),
    )

    question = st.text_area(
        "Research question",
        placeholder=(
            "Ask something about your research papers..."
        ),
        height=130,
    )

    settings_column, action_column = st.columns(
        [2, 1]
    )


    with settings_column:

        top_k = st.slider(
            "Retrieved passages",
            min_value=1,
            max_value=10,
            value=5,
        )


    with action_column:

        st.markdown(
            "<br>",
            unsafe_allow_html=True,
        )

        ask_button = st.button(
            "🔎 Ask Research Agent",
            type="primary",
            use_container_width=True,
        )


    # ----------------------------------------------------------------
    # Question processing
    # ----------------------------------------------------------------

    if ask_button:

        if not selected_documents:

            st.warning(
                "Please select at least one paper."
            )

        elif not question.strip():

            st.warning(
                "Please enter a research question."
            )

        else:

            try:

                rag_service = (
                    get_rag_service()
                )

                retrieval_k = top_k

                if len(selected_documents) > 1:

                    retrieval_k = min(
                        top_k * 3,
                        30,
                    )


                # ----------------------------------------------------
                # Retrieval
                # ----------------------------------------------------

                with st.spinner(
                    "Searching your research library..."
                ):

                    result = rag_service.answer(
                        question=question,
                        top_k=retrieval_k,
                    )


                # ----------------------------------------------------
                # Filter results to selected papers
                # ----------------------------------------------------

                filtered_results = [
                    item
                    for item in result.get(
                        "results",
                        [],
                    )
                    if item.get(
                        "metadata",
                        {},
                    ).get(
                        "source"
                    ) in selected_documents
                ]

                filtered_results = (
                    filtered_results[:top_k]
                )


                # ----------------------------------------------------
                # Rebuild answer if filtering changed the context
                # ----------------------------------------------------

                if (
                    len(selected_documents) > 1
                    and filtered_results
                    and len(filtered_results)
                    != len(
                        result.get(
                            "results",
                            [],
                        )
                    )
                ):

                    context = (
                        rag_service.context_builder.build_context(
                            filtered_results
                        )
                    )

                    prompt = (
                        rag_service.prompt_builder.build(
                            question=question,
                            context=context,
                        )
                    )

                    with st.spinner(
                        "Synthesizing evidence..."
                    ):

                        answer = (
                            rag_service.llm_service.generate(
                                prompt
                            )
                        )

                    citations = (
                        rag_service.citation_service.build_citations(
                            filtered_results
                        )
                    )

                    result = {
                        "answer": answer,
                        "results": filtered_results,
                        "context": context,
                        "prompt": prompt,
                        "citations": citations,
                    }

                else:

                    result["results"] = (
                        filtered_results
                    )

                    result["citations"] = (
                        rag_service.citation_service.build_citations(
                            filtered_results
                        )
                    )


                # ----------------------------------------------------
                # Research mode
                # ----------------------------------------------------

                if len(selected_documents) == 1:

                    st.caption(
                        f"📄 Researching "
                        f"**{selected_documents[0]}**"
                    )

                else:

                    st.caption(
                        f"🔬 Cross-paper research · "
                        f"**{len(selected_documents)} papers**"
                    )


                # ----------------------------------------------------
                # Answer
                # ----------------------------------------------------

                st.markdown(
                    '<div class="section-label">'
                    "Generated Research"
                    "</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="section-title">'
                    "Answer"
                    "</div>",
                    unsafe_allow_html=True,
                )


                answer_text = escape(
                    str(
                        result.get(
                            "answer",
                            "",
                        )
                    )
                )

                answer_html = f"""
                <div class="answer-card">

                    <div class="answer-label">
                        AI Research Synthesis
                    </div>

                    <div class="answer-text">
                        {answer_text}
                    </div>

                </div>
                """

                st.markdown(
                    answer_html,
                    unsafe_allow_html=True,
                )


                # ----------------------------------------------------
                # Evaluation
                # ----------------------------------------------------

                evaluation = (
                    evaluation_service.evaluate(
                        results=result.get(
                            "results",
                            [],
                        ),
                        citations=result.get(
                            "citations",
                            [],
                        ),
                        answer=result.get(
                            "answer",
                            "",
                        ),
                        context=result.get(
                            "context",
                            "",
                        ),
                    )
                )


                st.markdown(
                    '<div class="section-title">'
                    "📊 RAG Quality"
                    "</div>",
                    unsafe_allow_html=True,
                )

                retrieval_score = (
                    evaluation["retrieval"][
                        "relevance_rate"
                    ] * 100
                )

                citation_score = (
                    evaluation["citation"][
                        "citation_accuracy"
                    ] * 100
                )

                groundedness_passed = (
                    evaluation["groundedness"][
                        "passed"
                    ]
                )

                overall_score = (
                    evaluation["overall_score"]
                    * 100
                )

                st.markdown(
                    '<div class="scorecard">',
                    unsafe_allow_html=True,
                )

                score_columns = st.columns(4)

                with score_columns[0]:

                    st.metric(
                        "Retrieval",
                        f"{retrieval_score:.0f}%",
                    )

                with score_columns[1]:

                    st.metric(
                        "Citations",
                        f"{citation_score:.0f}%",
                    )

                with score_columns[2]:

                    st.metric(
                        "Groundedness",
                        (
                            "PASS"
                            if groundedness_passed
                            else "FAIL"
                        ),
                    )

                with score_columns[3]:

                    st.metric(
                        "Overall",
                        f"{overall_score:.0f}%",
                    )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )

                if evaluation["passed"]:

                    st.success(
                        "✓ All deterministic RAG quality checks passed."
                    )

                else:

                    st.warning(
                        "Some RAG quality checks did not pass."
                    )


                # ----------------------------------------------------
                # Sources
                # ----------------------------------------------------

                citations = result.get(
                    "citations",
                    [],
                )

                if citations:

                    st.markdown(
                        '<div class="section-title">'
                        "📑 Sources"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    for citation in citations:

                        source = escape(
                            str(
                                citation.get(
                                    "source",
                                    "Unknown source",
                                )
                            )
                        )

                        page = escape(
                            str(
                                citation.get(
                                    "page",
                                    "Unknown",
                                )
                            )
                        )

                        st.markdown(
                            f"""
                            <div class="source-card">

                                <div class="source-name">
                                    📄 {source}
                                </div>

                                <div class="source-page">
                                    Page {page}
                                </div>

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


                # ----------------------------------------------------
                # Paper contribution statistics
                # ----------------------------------------------------

                results = result.get(
                    "results",
                    [],
                )

                if results:

                    contributions = {}

                    for item in results:

                        source = item.get(
                            "metadata",
                            {},
                        ).get(
                            "source",
                            "Unknown",
                        )

                        contributions[source] = (
                            contributions.get(
                                source,
                                0,
                            ) + 1
                        )


                    if len(contributions) > 1:

                        st.markdown(
                            '<div class="section-title">'
                            "📊 Evidence Distribution"
                            "</div>",
                            unsafe_allow_html=True,
                        )

                        columns = st.columns(
                            len(contributions)
                        )

                        for column, (
                            source,
                            count,
                        ) in zip(
                            columns,
                            contributions.items(),
                        ):

                            with column:

                                st.metric(
                                    source,
                                    f"{count} passage"
                                    + (
                                        ""
                                        if count == 1
                                        else "s"
                                    ),
                                )


                # ----------------------------------------------------
                # Retrieved evidence
                # ----------------------------------------------------

                if results:

                    with st.expander(
                        "🔍 View retrieved evidence"
                    ):

                        for index, item in enumerate(
                            results,
                            start=1,
                        ):

                            metadata = item.get(
                                "metadata",
                                {},
                            )

                            source = metadata.get(
                                "source",
                                "Unknown",
                            )

                            page = metadata.get(
                                "page",
                                "Unknown",
                            )

                            distance = item.get(
                                "distance",
                                0,
                            )

                            st.markdown(
                                f"**Evidence {index}**  \n"
                                f"📄 {source} · "
                                f"Page {page}"
                            )

                            st.caption(
                                f"Semantic distance: "
                                f"{distance:.4f}"
                            )

                            st.write(
                                item.get(
                                    "chunk",
                                    "",
                                )
                            )

                            st.divider()


            except Exception as exc:

                st.error(
                    f"Failed to generate answer: {exc}"
                )


# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        AI Research Agent ·
        RAG + FAISS + Sentence Transformers + Ollama
        <br>
        Private local research intelligence
    </div>
    """,
    unsafe_allow_html=True,
)