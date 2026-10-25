"""
app.py
-------
SMAAI - Smart Meeting AI. This is the Streamlit user interface.

Run with:  streamlit run app.py

This file only draws pages and buttons. The real work happens in
meeting_manager.py (and the files it uses).
"""

import csv
import io
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # read settings (MongoDB address, API key...) from the .env file

import database  # noqa: E402  (imported after load_dotenv on purpose)
import llm_service  # noqa: E402
import meeting_manager  # noqa: E402
import text_to_speech  # noqa: E402

st.set_page_config(page_title="SMAAI - Smart Meeting AI", page_icon="🎙️", layout="wide")


# ---------------------------------------------------------------------------
# Small helper functions used by several pages
# ---------------------------------------------------------------------------

def format_duration(seconds):
    """85.3 -> '1m 25s'"""
    if not seconds:
        return "unknown"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s"


def format_date(date_value):
    return date_value.strftime("%d %b %Y, %H:%M") if date_value else "unknown date"


def run_action(function, *args, spinner_text="Working..."):
    """
    Run function(*args) with a spinner. If it fails, show a friendly error message
    and return None. (Every function used with run_action returns a non-None value
    when it succeeds.)
    """
    try:
        with st.spinner(spinner_text):
            return function(*args)
    except (ValueError, llm_service.LLMError) as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Something went wrong: {error}")
    return None


def pick_meeting(key):
    """Show a drop-down of saved meetings and return the chosen meeting id (or None)."""
    meetings = meeting_manager.list_meetings()
    if not meetings:
        st.info("No meetings saved yet. Go to 'Upload & Transcribe' to add one.")
        return None
    labels = {m["id"]: f"{m['title']}  -  {format_date(m['created_at'])}" for m in meetings}
    return st.selectbox("Select a meeting", options=list(labels), format_func=labels.get, key=key)


def read_aloud(text, key, language="en"):
    """A 'Read aloud' button that plays the text as speech."""
    if st.button("🔊 Read aloud", key=key):
        audio = run_action(text_to_speech.text_to_mp3, text, language, spinner_text="Generating speech...")
        if audio:
            st.audio(audio, format="audio/mp3", autoplay=False)
            if len(text) > text_to_speech.MAX_CHARACTERS:
                st.caption(f"Only the first {text_to_speech.MAX_CHARACTERS} characters are read aloud.")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_dashboard():
    st.subheader("Dashboard")
    stats = meeting_manager.dashboard_stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Meetings", stats["total_meetings"])
    col2.metric("Total audio (minutes)", f"{stats['total_minutes']:.1f}")
    col3.metric("Meetings summarised", stats["summarised"])
    col4.metric("Documents (RAG)", stats["total_documents"])

    st.markdown("#### Recent meetings")
    if not stats["recent"]:
        st.info("No meetings yet. Open 'Upload & Transcribe' to process your first recording.")
    for meeting in stats["recent"]:
        st.write(f"**{meeting['title']}** - {format_date(meeting['created_at'])} "
                 f"- {format_duration(meeting.get('duration_seconds'))}")

    with st.expander("System status"):
        st.write("✅ MongoDB connected")
        if os.getenv("LLM_API_KEY"):
            st.write(f"✅ LLM API key found (model: {os.getenv('LLM_MODEL', 'llama-3.3-70b-versatile')})")
        else:
            st.write("⚠️ LLM_API_KEY is missing - summaries, action items and Q&A will not work until it is set in .env")


def transcribe_and_show(file_name, file_bytes, title):
    """Run the whole upload pipeline and show the transcript."""
    result = run_action(meeting_manager.process_meeting, file_name, file_bytes, title,
                        spinner_text="Transcribing... this can take a few minutes for long recordings.")
    if result is not None:
        meeting_id, transcript = result
        st.success(f"Meeting '{title.strip()}' saved. You can now summarise it or ask questions about it.")
        st.text_area("Transcript", transcript, height=300, key=f"new_transcript_{meeting_id}")


def page_upload():
    st.subheader("Upload & Transcribe")
    title = st.text_input("Meeting title")
    upload_tab, microphone_tab = st.tabs(["Upload a file", "Record with microphone"])

    with upload_tab:
        uploaded = st.file_uploader("Meeting recording (WAV, MP3 or M4A)", type=["wav", "mp3", "m4a"])
        if st.button("Transcribe", type="primary", key="transcribe_upload"):
            if uploaded is None:
                st.error("No audio file was uploaded.")
            else:
                transcribe_and_show(uploaded.name, uploaded.getvalue(), title)

    with microphone_tab:
        recording = st.audio_input("Click the microphone, speak, then stop")
        if st.button("Transcribe recording", type="primary", key="transcribe_mic"):
            if recording is None:
                st.error("Please record something first.")
            else:
                transcribe_and_show("microphone_recording.wav", recording.getvalue(), title)


def page_history():
    st.subheader("Meeting History")
    meeting_id = pick_meeting("history_pick")
    if meeting_id is None:
        return
    meeting = meeting_manager.get_meeting(meeting_id)
    if meeting is None:
        st.error("This meeting no longer exists.")
        return

    st.markdown(f"### {meeting['title']}")
    st.caption(f"{format_date(meeting['created_at'])}  |  duration {format_duration(meeting.get('duration_seconds'))}"
               f"  |  language: {meeting.get('language') or 'unknown'}")

    audio_file = meeting_manager.audio_path(meeting)
    if audio_file:
        st.audio(audio_file)
    st.text_area("Transcript", meeting["transcript"], height=300, key=f"history_transcript_{meeting_id}")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️ Download transcript", meeting["transcript"],
                           file_name=f"{meeting['title']}.txt", key=f"download_{meeting_id}")
    with col2:
        read_aloud(meeting["transcript"], f"read_transcript_{meeting_id}", meeting.get("language") or "en")

    st.markdown("---")
    st.markdown("**Delete this meeting**")
    confirmed = st.checkbox("Yes, delete this meeting and its audio file permanently", key=f"confirm_{meeting_id}")
    if st.button("🗑️ Delete meeting", disabled=not confirmed, key=f"delete_{meeting_id}"):
        if run_action(meeting_manager.delete_meeting, meeting_id) is not None:
            st.session_state["flash"] = f"Deleted '{meeting['title']}'."
            st.rerun()


def page_search():
    st.subheader("Search Meetings")
    query = st.text_input("Search meeting titles and transcripts (e.g. 'database', 'deadline')")
    if not query.strip():
        return

    results = run_action(meeting_manager.search, query, spinner_text="Searching...")
    if results is None:
        return
    if not results:
        st.warning(f"No meetings found for '{query}'.")
        return

    st.write(f"Found **{len(results)}** meeting(s):")
    for meeting in results:
        with st.expander(f"{meeting['title']}  -  {format_date(meeting['created_at'])}"):
            if meeting["snippet"]:
                st.markdown(f"Mentioned **{meeting['match_count']}** time(s). First match:")
                st.markdown(f"> {meeting['snippet']}")
            else:
                st.write("The match is in the meeting title.")
            st.text_area("Full transcript", meeting["transcript"], height=200, key=f"search_{meeting['id']}")


def page_summaries():
    st.subheader("Meeting Summaries")
    meeting_id = pick_meeting("summary_pick")
    if meeting_id is None:
        return
    meeting = meeting_manager.get_meeting(meeting_id)
    summary = meeting["summary"]

    button_label = "🔄 Regenerate summary" if summary else "✨ Generate summary"
    if st.button(button_label, type="primary", key=f"summarise_{meeting_id}"):
        new_summary = run_action(meeting_manager.get_summary, meeting_id, True,
                                 spinner_text="AI is summarising the meeting...")
        if new_summary is not None:
            summary = new_summary

    if summary:
        st.markdown(summary)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Download summary", summary,
                               file_name=f"{meeting['title']}_summary.txt", key=f"dl_summary_{meeting_id}")
        with col2:
            read_aloud(summary, f"read_summary_{meeting_id}")
    else:
        st.info("No summary yet. Click 'Generate summary'.")


def page_action_items():
    st.subheader("Action Items")
    meeting_id = pick_meeting("action_pick")
    if meeting_id is None:
        return
    meeting = meeting_manager.get_meeting(meeting_id)
    items = meeting["action_items"]  # None = never generated, [] = generated but none found

    button_label = "🔄 Re-extract action items" if items is not None else "✨ Extract action items"
    if st.button(button_label, type="primary", key=f"extract_{meeting_id}"):
        new_items = run_action(meeting_manager.get_action_items, meeting_id, True,
                               spinner_text="AI is looking for action items...")
        if new_items is not None:
            items = new_items

    if items is None:
        st.info("Action items have not been extracted yet.")
    elif not items:
        st.success("No action items were found in this meeting.")
    else:
        rows = [{
            "Task": item["task"],
            "Owner": item["owner"] or "Not mentioned",
            "Deadline": item["deadline"] or "Not mentioned",
            "Evidence from transcript": item["quote"] or "",
        } for item in items]
        st.dataframe(rows)
        st.caption("Owner and deadline are shown only when they were actually said in the meeting.")

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        st.download_button("⬇️ Download as CSV", buffer.getvalue(),
                           file_name=f"{meeting['title']}_action_items.csv", key=f"dl_actions_{meeting_id}")


def show_chat_history(history_key):
    """Display all previous questions and answers stored in session_state."""
    for question, answer, sources in st.session_state.get(history_key, []):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
            if sources:
                with st.expander("Text used to answer"):
                    for source in sources:
                        st.write(source)
                        st.markdown("---")


def page_ask_questions():
    st.subheader("Ask Questions")
    meeting_tab, document_tab = st.tabs(["Ask about a meeting", "Ask about documents (RAG)"])

    with meeting_tab:
        meeting_id = pick_meeting("qa_pick")
        if meeting_id is not None:
            history_key = f"qa_history_{meeting_id}"
            show_chat_history(history_key)
            with st.form(f"meeting_question_form_{meeting_id}", clear_on_submit=True):
                question = st.text_input("Your question about this meeting")
                asked = st.form_submit_button("Ask")
            if asked:
                result = run_action(meeting_manager.ask_about_meeting, meeting_id, question,
                                    spinner_text="Reading the transcript...")
                if result is not None:
                    answer, passages = result
                    st.session_state.setdefault(history_key, []).append((question, answer, passages))
                    st.rerun()

    with document_tab:
        documents = meeting_manager.list_documents()
        if not documents:
            st.info("No documents uploaded yet. Add a PDF, DOCX or TXT file on the 'Documents' page.")
        else:
            names = {d["id"]: d["name"] for d in documents}
            chosen = st.multiselect("Documents to search", options=list(names), default=list(names),
                                    format_func=names.get)
            show_chat_history("document_qa_history")
            with st.form("document_question_form", clear_on_submit=True):
                question = st.text_input("Your question about the selected documents")
                asked = st.form_submit_button("Ask")
            if asked:
                result = run_action(meeting_manager.ask_about_documents, question, chosen,
                                    spinner_text="Searching the documents...")
                if result is not None:
                    answer, sources = result
                    shown_sources = [f"**{name}**: {text}" for name, text in sources]
                    st.session_state.setdefault("document_qa_history", []).append((question, answer, shown_sources))
                    st.rerun()


def page_documents():
    st.subheader("Documents")
    st.write("Upload reference documents (agendas, reports, notes). You can then ask questions "
             "about them on the 'Ask Questions' page.")

    uploaded = st.file_uploader("Choose a document", type=["pdf", "docx", "txt"])
    if st.button("Add document", type="primary"):
        if uploaded is None:
            st.error("No document was selected.")
        else:
            result = run_action(meeting_manager.add_document, uploaded.name, uploaded.getvalue(),
                                spinner_text="Reading and indexing the document...")
            if result is not None:
                st.success(f"Added '{uploaded.name}' ({result[1]} text chunks).")

    st.markdown("#### Uploaded documents")
    documents = meeting_manager.list_documents()
    if not documents:
        st.info("No documents uploaded yet.")
    for document in documents:
        col1, col2 = st.columns([4, 1])
        col1.write(f"📄 **{document['name']}** - {len(document['chunks'])} chunks - "
                   f"uploaded {format_date(document['uploaded_at'])}")
        if col2.button("Delete", key=f"delete_doc_{document['id']}"):
            if run_action(meeting_manager.delete_document, document["id"]) is not None:
                st.session_state["flash"] = f"Deleted '{document['name']}'."
                st.rerun()


# ---------------------------------------------------------------------------
# Main: connect to the database, show the sidebar and open the chosen page
# ---------------------------------------------------------------------------

st.title("🎙️ SMAAI - Smart Meeting AI")

try:
    database.check_connection()
except Exception as error:
    st.error("Cannot connect to MongoDB. Start MongoDB and check MONGODB_URI in your .env file.")
    st.caption(f"Technical details: {error}")
    st.stop()

PAGES = {
    "Dashboard": page_dashboard,
    "Upload & Transcribe": page_upload,
    "Meeting History": page_history,
    "Search": page_search,
    "Summaries": page_summaries,
    "Action Items": page_action_items,
    "Ask Questions": page_ask_questions,
    "Documents": page_documents,
}
chosen_page = st.sidebar.radio("Navigate", list(PAGES))

# A message saved before st.rerun() (for example after deleting something).
if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

PAGES[chosen_page]()
