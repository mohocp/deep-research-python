import streamlit as st
import asyncio
import nest_asyncio
import os
import dotenv
from deep_research import feedback_questions, deep_research, write_final_report


# Make the page layout wide to avoid overlap in narrow layouts.
st.set_page_config(page_title="Deep Research App", layout="wide")

# Apply nest_asyncio to allow nested event loops in Streamlit.
nest_asyncio.apply()

dotenv.load_dotenv()


DEFAULT_BREADTH = 4
DEFAULT_DEPTH = 2

def main():
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "breadth" not in st.session_state:
        st.session_state.breadth = DEFAULT_BREADTH
    if "depth" not in st.session_state:
        st.session_state.depth = DEFAULT_DEPTH
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "answers" not in st.session_state:
        st.session_state.answers = {}

    st.title("Deep Research App")

    # Step 1: Enter Query & Parameters
    if st.session_state.step == 1:
        st.header("Step 1: Enter Your Query & Parameters")
        st.session_state.query = st.text_input("Enter your query:", key="query_input")
        st.session_state.breadth = st.number_input(
            "Breadth (how many branches to search):",
            min_value=1, value=DEFAULT_BREADTH, key="breadth_input"
        )
        st.session_state.depth = st.number_input(
            "Depth (how many levels deep to search):",
            min_value=1, value=DEFAULT_DEPTH, key="depth_input"
        )
        
        if st.button("Submit Query"):
            if st.session_state.query:
                with st.spinner("Generating follow up questions..."):
                    try:
                        questions = asyncio.run(feedback_questions(st.session_state.query))
                    except Exception as e:
                        st.error(f"Error generating questions: {e}")
                        questions = []
                st.session_state.questions = questions
                st.session_state.step = 2
            else:
                st.error("Please enter a query before submitting.")

    # Step 2: Answer Follow Up Questions
    if st.session_state.step == 2:
        st.header("Step 2: Answer the Follow Up Questions")
        st.write("Please provide answers for the following questions:")
        for idx, question in enumerate(st.session_state.questions):
            key = f"question_{idx}"
            if key not in st.session_state.answers:
                st.session_state.answers[key] = ""
            st.session_state.answers[key] = st.text_input(question, key=key)
        
        if st.button("Proceed to Research"):
            unanswered = [
                q for i, q in enumerate(st.session_state.questions)
                if not st.session_state.answers.get(f"question_{i}", "").strip()
            ]
            if unanswered:
                st.error("Please answer all follow-up questions before proceeding.")
            else:
                st.session_state.step = 3

    # Step 3: Run Deep Research & Show Progress
    if st.session_state.step == 3:
        st.header("Step 3: Running Deep Research")

        answers_list = [
            st.session_state.answers[f"question_{i}"] for i in range(len(st.session_state.questions))
        ]
        joined_qna = "\n".join([
            f"- Question: {q}\n  - Answer: {a}"
            for q, a in zip(st.session_state.questions, answers_list)
        ])
        combined_query = (
            f"User query: {st.session_state.query}\n"
            f"Follow up questions and answers:\n{joined_qna}"
        )
        
        status_placeholder = st.empty()
        progress_cols_placeholder = st.empty()
        report_placeholder = st.empty()
        
        progress = {
            "total_urls": [],
            "total_queries": [],
            "current_queries": [],
            "status": "initializing"
        }
        
        # Callback to update progress in the UI
        def streamlit_progress_callback(progress_update):
            status = progress_update.get("status", "")
            total_urls = progress_update.get("total_urls", [])
            total_queries = progress_update.get("total_queries", [])
            current_queries = progress_update.get("current_queries", [])

            status_placeholder.markdown(f"**Status:** {status}")

            # Display columns within a container
            with progress_cols_placeholder.container():
                # Custom CSS to handle wrapping
                st.markdown(
                    """
                    <style>
                        .wrap-container {
                            width: 90%;
                            margin: auto;
                            /* Optional: reduce font-size if columns are still too wide */
                            /* font-size: 14px; */
                        }
                        .wrap-container a {
                            /* Ensure very long links wrap */
                            word-wrap: break-word;
                            overflow-wrap: anywhere;
                            display: inline-block;
                            text-decoration: none;
                            color: #1a73e8;
                        }
                        .wrap-container ul {
                            margin-top: 0;
                            margin-bottom: 0.5em;
                            padding-left: 1.2em;
                            word-wrap: break-word;
                            overflow-wrap: anywhere;
                        }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # Use bigger ratio for the first column (2) to accommodate long URLs
                col1, col2, col3 = st.columns([2, 1, 1], gap="large")

                with col1:
                    st.markdown(f"**Total URLs ({len(total_urls)})**")
                    if total_urls:
                        items = "\n".join([
                            f"<li><a href='{url}' target='_blank'>{url}</a></li>"
                            for url in total_urls
                        ])
                        st.markdown(f"<ul>{items}</ul>", unsafe_allow_html=True)
                    else:
                        st.write("None")

                with col2:
                    st.markdown(f"**Total Queries ({len(total_queries)})**")
                    if total_queries:
                        items = "\n".join([f"<li>{q}</li>" for q in total_queries])
                        st.markdown(f"<ul>{items}</ul>", unsafe_allow_html=True)
                    else:
                        st.write("None")

                with col3:
                    st.markdown(f"**Current Queries ({len(current_queries)})**")
                    if current_queries:
                        items = "\n".join([f"<li>{q}</li>" for q in current_queries])
                        st.markdown(f"<ul>{items}</ul>", unsafe_allow_html=True)
                    else:
                        st.write("None")

        # Perform the deep research
        with st.spinner("Deep research in progress..."):
            try:
                result = asyncio.run(
                    deep_research(
                        combined_query,
                        breadth=int(st.session_state.breadth),
                        depth=int(st.session_state.depth),
                        progress=progress,
                        on_progress=streamlit_progress_callback
                    )
                )
                report = asyncio.run(
                    write_final_report(
                        st.session_state.query,
                        result.get("learnings", ""),
                        result.get("visited_urls", [])
                    )
                )
            except Exception as e:
                st.error(f"An error occurred during deep research: {e}")
                report = ""
        
        # Render the final report as Markdown
        if report:
            st.markdown("### Final Report")
            st.markdown(report)  
            st.download_button("Download Report", report, file_name="output.md")
            st.success("Deep research completed!")

if __name__ == "__main__":
    main()
