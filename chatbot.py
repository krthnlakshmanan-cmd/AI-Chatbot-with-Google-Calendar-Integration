
import streamlit as st
import tempfile
import asyncio
import json
import re
import traceback
import os
import hashlib
import fitz
import pytesseract
from PIL import Image
from io import BytesIO
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from google import genai
from tavily import TavilyClient
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.document_loaders import (
    Docx2txtLoader,
    BSHTMLLoader,
    TextLoader
)
# Google Calendar MCP integration
from mcp_integration import call_calendar_tool
import pytesseract
import os
from mcp_integration import call_calendar_tool

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

print("Tesseract Version:", pytesseract.get_tesseract_version())


# ==================================================
# 1. CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

# Initialize Gemini only when an API key exists
client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)

# Initialize Tavily only when an API key exists
tavily = (
    TavilyClient(api_key=TAVILY_API_KEY)
    if TAVILY_API_KEY
    else None
)

# Windows Tesseract configuration
# ==================================================
# 1. CONFIGURATION
# ==================================================

import os
import pytesseract

TESSERACT_PATH = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Configure Tesseract OCR
if os.path.isfile(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    print("Tesseract OCR configured successfully!")
    print("Version:", pytesseract.get_tesseract_version())
else:
    print("ERROR: Tesseract executable not found:", TESSERACT_PATH)

# ==================================================
# 2. LOAD EMBEDDINGS
# ==================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ==================================================
# 3. PDF TEXT EXTRACTION WITH OCR
# ==================================================

def extract_pdf_content(pdf_path):

    documents = []

    if not os.path.exists(TESSERACT_PATH):
        raise FileNotFoundError(
            "Tesseract OCR was not found. Install Tesseract "
            "or check the TESSERACT_CMD path."
        )

    with fitz.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf):

            # Extract selectable PDF text
            normal_text = page.get_text("text").strip()

            # Render the complete page as an image
            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False
            )

            image = Image.open(
                BytesIO(pix.tobytes("png"))
            )

            # OCR the complete page
            ocr_text = pytesseract.image_to_string(
                image,
                lang="eng"
            ).strip()

            # Keep normal text and add OCR text when useful.
            # This helps with pages containing both selectable
            # text and specifications inside images.
            if normal_text and ocr_text:

                normal_clean = re.sub(
                    r"\s+",
                    " ",
                    normal_text
                ).strip().lower()

                ocr_clean = re.sub(
                    r"\s+",
                    " ",
                    ocr_text
                ).strip().lower()

                if normal_clean in ocr_clean:
                    combined_text = ocr_text
                else:
                    combined_text = (
                        normal_text + "\n\n" + ocr_text
                    )

            else:
                combined_text = normal_text or ocr_text

            if combined_text.strip():

                documents.append(
                    Document(
                        page_content=combined_text,
                        metadata={
                            "page": page_number + 1,
                            "source": os.path.basename(pdf_path)
                        }
                    )
                )

    return documents


# ==================================================
# 4. LOAD OTHER DOCUMENT TYPES
# ==================================================

def load_document(file_path, file_extension):

    if file_extension == "pdf":

        return extract_pdf_content(file_path)

    elif file_extension == "docx":

        loader = Docx2txtLoader(file_path)
        return loader.load()

    elif file_extension in ["html", "htm"]:

        loader = BSHTMLLoader(file_path)
        return loader.load()

    elif file_extension == "txt":

        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )

        return loader.load()

    else:

        raise ValueError("Unsupported file format.")
# ==================================================
# 5. CALENDAR QUESTION DETECTION
# ==================================================

def detect_calendar_request(question):

    if not isinstance(question, str) or not question.strip():
        return False

    q = question.lower()

    calendar_keywords = [
        "calendar",
        "meeting",
        "meetings",
        "event",
        "events",
        "schedule",
        "appointment",
        "appointments",
        "standup",
        "stand-up",
        "upcoming events",
        "upcoming meetings",
        "my next event",
        "my next meeting"
    ]

    return any(
        word in q for word in calendar_keywords
    )


# ==================================================
# 6. GET GOOGLE CALENDAR EVENTS THROUGH MCP
# ==================================================

def get_calendar_events(question):

    if not isinstance(question, str) or not question.strip():
        return "Please enter a valid calendar question."

    q = question.lower()

    max_results = 10

    number_match = re.search(
        r"(?:next|show|last|upcoming)\s+(\d+)",
        q
    )

    if number_match:

        max_results = min(
            max(int(number_match.group(1)), 1),
            50
        )

    try:

        result = asyncio.run(
            call_calendar_tool(
                "get_events",
                {"max_results": max_results}
            )
        )

        events = (
            json.loads(result)
            if isinstance(result, str)
            else result
        )

        if isinstance(events, dict):
            events = events.get("events", [])

        india_tz = ZoneInfo("Asia/Kolkata")

        today = datetime.now(india_tz).date()

        target_date = None

        if "tomorrow" in q:

            target_date = today + timedelta(days=1)

        elif "today" in q:

            target_date = today

        # Filter events by date
        if target_date:

            filtered_events = []

            for event in events:

                start_info = event.get("start", {})

                start_value = (
                    start_info.get("dateTime")
                    or start_info.get("date")
                )

                if not start_value:
                    continue

                if "T" in start_value:

                    event_date = datetime.fromisoformat(
                        start_value.replace("Z", "+00:00")
                    ).astimezone(india_tz).date()

                else:

                    event_date = datetime.fromisoformat(
                        start_value
                    ).date()

                if event_date == target_date:
                    filtered_events.append(event)

            events = filtered_events

        if not events:
            return "No upcoming calendar events found."

        output = []

        for event in events:

            summary = event.get(
                "summary",
                "Untitled event"
            )

            start_info = event.get("start", {})
            end_info = event.get("end", {})

            start_value = (
                start_info.get("dateTime")
                or start_info.get("date")
            )

            end_value = (
                end_info.get("dateTime")
                or end_info.get("date")
            )

            # Format start time
            if start_value and "T" in start_value:

                start_dt = datetime.fromisoformat(
                    start_value.replace("Z", "+00:00")
                )

                start_text = start_dt.astimezone(
                    india_tz
                ).strftime("%d-%m-%Y %I:%M %p")

            elif start_value:

                start_text = f"{start_value} (All day)"

            else:

                start_text = "Not available"

            # Format end time
            if end_value and "T" in end_value:

                end_dt = datetime.fromisoformat(
                    end_value.replace("Z", "+00:00")
                )

                end_text = end_dt.astimezone(
                    india_tz
                ).strftime("%I:%M %p")

            else:

                end_text = "Not available"

            location = event.get(
                "location",
                "Not specified"
            )

            meet_link = ""

            for entry in event.get(
                "conferenceData", {}
            ).get("entryPoints", []):

                if entry.get("entryPointType") == "video":
                    meet_link = entry.get("uri", "")

            output.append(
                f"""
📅 **{summary}**

- Start: {start_text}
- End: {end_text}
- Location: {location}
- Google Meet: {meet_link or 'Not available'}
"""
            )

        return "\n".join(output)

    except Exception as e:

        return f"Google Calendar error: {str(e)}"


# ==================================================
# 7. UNDERSTAND MEETING REQUEST
# ==================================================


def understand_meeting_request(question):

    try:
        # Set Indian timezone
        india_timezone = ZoneInfo("Asia/Kolkata")
        current_date = datetime.now(india_timezone).date()

        # --------------------------------
        # 1. EXTRACT MEETING TITLE
        # --------------------------------

        title_match = re.search(
            r"(?:called|named|of)\s+(.+?)(?=\s+(?:on|today|tomorrow|at)\b|$)",
            question,
            re.IGNORECASE
        )

        title = (
            title_match.group(1).strip(" .,!?")
            if title_match
            else "Meeting Schedule"
        )

        # --------------------------------
        # 2. EXTRACT MEETING DATE
        # --------------------------------

        if re.search(r"\btomorrow\b", question, re.IGNORECASE):

            meeting_date = current_date + timedelta(days=1)

        elif re.search(r"\btoday\b", question, re.IGNORECASE):

            meeting_date = current_date

        else:

            date_match = re.search(
                r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})\b",
                question,
                re.IGNORECASE
            )

            if not date_match:
                return None

            date_text = date_match.group(1)

            if re.fullmatch(
                r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}",
                date_text
            ):

                date_format = (
                    "%d.%m.%y" if "." in date_text and len(date_text.split(".")[-1]) == 2
                    else "%d.%m.%Y" if "." in date_text
                    else "%d/%m/%y" if "/" in date_text and len(date_text.split("/")[-1]) == 2
                    else "%d/%m/%Y" if "/" in date_text
                    else "%d-%m-%y" if len(date_text.split("-")[-1]) == 2
                    else "%d-%m-%Y"
                )

                meeting_date = datetime.strptime(
                    date_text,
                    date_format
                ).date()

            else:

                try:
                    meeting_date = datetime.strptime(
                        date_text,
                        "%d %B %Y"
                    ).date()
                except ValueError:
                    meeting_date = datetime.strptime(
                        date_text,
                        "%d %b %Y"
                    ).date()

        # --------------------------------
        # 3. EXTRACT MEETING TIME
        # --------------------------------

        time_match = re.search(
            r"\bat\s+(\d{1,2})(?:(:|\.)(\d{2}))?\s*(AM|PM)\b",
            question,
            re.IGNORECASE
        )

        if not time_match:
            return None

        hour = int(time_match.group(1))
        minute = int(time_match.group(3) or 0)
        meridian = time_match.group(4).upper()

        if hour < 1 or hour > 12 or minute > 59:
            return None

        if meridian == "PM" and hour != 12:
            hour += 12

        elif meridian == "AM" and hour == 12:
            hour = 0

        # --------------------------------
        # 4. CREATE START DATETIME
        # --------------------------------

        start = datetime(
            meeting_date.year,
            meeting_date.month,
            meeting_date.day,
            hour,
            minute,
            tzinfo=india_timezone
        )

        # --------------------------------
        # 5. EXTRACT MEETING DURATION
        # --------------------------------

        duration_match = re.search(
            r"for\s+(\d+)\s+(minutes?|mins?|hours?|hrs?)\b",
            question,
            re.IGNORECASE
        )

        if duration_match:

            duration_value = int(duration_match.group(1))
            duration_unit = duration_match.group(2).lower()

            if duration_unit.startswith(("hour", "hr")):
                duration = duration_value * 60
            else:
                duration = duration_value
        else:
            duration = 30  # Default duration
        if duration <= 0:
            return None

        # --------------------------------
        # 6. CALCULATE END DATETIME
        # --------------------------------

        end = start + timedelta(minutes=duration)

        # --------------------------------
        # 7. RETURN MEETING DETAILS
        # --------------------------------

        return {
            "summary": title,
            "start_datetime": start.isoformat(),
            "end_datetime": end.isoformat(),
            "description": "Created using AI PDF Chatbot",
            "location": "",
            "attendees": [],
            "create_meet_link": True
        }

    except Exception as e:

        print("Meeting parsing error:", str(e))
        return None

# ==================================================
# 8. CREATE GOOGLE CALENDAR MEETING THROUGH MCP
# ==================================================

def create_calendar_event(question):

    try:

        details = understand_meeting_request(question)

        if not details:
           return (
        "I can help schedule your meeting! "
        "Please provide the missing details. "
        "For example: What time should I schedule it today, "
        "and what should I call the meeting?"
    )

        title = details["summary"]

        start_datetime = datetime.fromisoformat(
            details["start_datetime"]
        )

        end_datetime = datetime.fromisoformat(
            details["end_datetime"]
        )

        result = asyncio.run(
            call_calendar_tool(
                "create_event",
                {
                    "summary": title,
                    "start_datetime": start_datetime.isoformat(),
                    "end_datetime": end_datetime.isoformat(),
                    "description": details.get("description", ""),
                    "location": details.get("location", ""),
                    "attendees": details.get("attendees", []),
                    "create_meet_link": True
                }
            )
        )

        event = (
            json.loads(result)
            if isinstance(result, str)
            else result
        )

        if event.get("status") != "success":

            return f"Meeting creation failed: {event}"

        return f"""
### Meeting created successfully!

**Title:** {title}

**Start:** {start_datetime.strftime('%d-%m-%Y %I:%M %p')}

**End:** {end_datetime.strftime('%I:%M %p')}

**Google Meet:** {event.get('google_meet_link', 'Not available')}

**Google Calendar:** {event.get('htmlLink', event.get('calendar_link', 'Not available'))}
"""

    except Exception as e:

        traceback.print_exc()

        return f"Meeting creation error: {repr(e)}"


# ==================================================
# 9. SIDEBAR
# ==================================================

with st.sidebar:

    st.title("⚙️ Settings")

    uploaded_file = st.file_uploader(
        "📄 Upload Documents",
        type=["pdf", "docx", "html", "htm", "txt"]
    )

    web_toggle = st.toggle(
        "🌐 Enable Web Search"
    )

    st.markdown("---")

    st.markdown(
        "🚀 Built by Keerthana Lakshmanan"
    )


# ==================================================
# 10. MAIN UI
# ==================================================

st.title("🤖 AI  Chatbot")

st.caption(
    "Ask questions from your Documents, Web, Google Calendar or general knowledge."
)


# ==================================================
# 11. DOCUMENT PROCESSING AND FAISS
# ==================================================

retriever = None

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    # Generate a unique signature for the uploaded file
    file_signature = hashlib.sha256(
        file_bytes
    ).hexdigest()

    # Process only when a new file is uploaded
    if (
        st.session_state.get("file_signature")
        != file_signature
    ):

        st.session_state.file_signature = None
        st.session_state.retriever = None

        file_extension = (
            uploaded_file.name.split(".")[-1].lower()
        )

        file_path = None

        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix="." + file_extension
            ) as tmp_file:

                tmp_file.write(file_bytes)
                file_path = tmp_file.name

            with st.spinner(
                "Reading document and creating search index..."
            ):

                # Load document
                documents = load_document(
                    file_path,
                    file_extension
                )

                if not documents:

                    st.error(
                        "No readable text found in the document."
                    )

                    st.stop()

                # Split document into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(
                    documents
                )

                if not chunks:

                    st.error(
                        "Could not create document chunks."
                    )

                    st.stop()

                # Generate embeddings
                embeddings = load_embeddings()

                # Create FAISS vector database
                vectorstore = FAISS.from_documents(
                    chunks,
                    embeddings
                )

                # Create retriever
                st.session_state.retriever = (
                    vectorstore.as_retriever(
                        search_kwargs={"k": 5}
                    )
                )

                st.session_state.file_signature = (
                    file_signature
                )

                st.session_state.file_name = (
                    uploaded_file.name
                )

                st.success(
                    f"Document uploaded successfully! "
                    f"Type: {file_extension.upper()} | "
                    f"Pages/Documents: {len(documents)} | "
                    f"Chunks: {len(chunks)}"
                )

        except Exception as e:

            st.error(
                f"Document processing error: {e}"
            )

        finally:

            if (
                file_path
                and os.path.exists(file_path)
            ):

                os.remove(file_path)

    else:

        st.success(
            f"Ready: {uploaded_file.name}"
        )

    retriever = st.session_state.get("retriever")


# ==================================================
# 12. CHAT INPUT
# ==================================================

query = st.chat_input(
    "Ask your question..."
)


# ==================================================
# 13. PROCESS USER QUESTION
# ==================================================

if query:

    with st.chat_message("user"):
        st.write(query)

    answer = ""

    # --------------------------------------------------
    # ROUTE 1: GOOGLE CALENDAR MCP
    # --------------------------------------------------

    if detect_calendar_request(query):

        with st.chat_message("assistant"):

            if re.search(
                r"\b(create|schedule|book|add)\b",
                query,
                re.IGNORECASE
            ):

                with st.spinner(
                    "Understanding and creating meeting..."
                ):

                    answer = create_calendar_event(query)

            else:

                with st.spinner(
                    "Fetching Google Calendar events..."
                ):

                    answer = get_calendar_events(query)

            st.markdown(answer)

    # --------------------------------------------------
    # ROUTE 2: PDF RAG, WEB SEARCH AND GENERAL CHAT
    # --------------------------------------------------

    else:

        pdf_context = ""

        # Search uploaded document
        if retriever is not None:

            with st.spinner(
                "Searching uploaded document..."
            ):

                docs = retriever.invoke(query)

            context_parts = []

            for doc in docs:

                page_number = doc.metadata.get(
                    "page",
                    "Unknown"
                )

                source = doc.metadata.get(
                    "source",
                    "Uploaded document"
                )

                context_parts.append(
                    f"""
Source: {source}
Page: {page_number}

Content:
{doc.page_content}
"""
                )

            pdf_context = "\n\n".join(
                context_parts
            )

        # --------------------------------------------------
        # ROUTE 3: TAVILY WEB SEARCH
        # --------------------------------------------------

        web_context = ""

        if web_toggle:

            if tavily is None:

                st.warning(
                    "Tavily API key is missing. "
                    "Please check your .env file."
                )

            else:

                try:

                    with st.spinner(
                        "Searching the web..."
                    ):

                        web_results = tavily.search(
                            query=query,
                            max_results=3
                        )

                    for result in web_results.get(
                        "results",
                        []
                    ):

                        content = result.get(
                            "content",
                            ""
                        )

                        url = result.get(
                            "url",
                            ""
                        )

                        if content.strip():

                            web_context += (
                                f"\nSource URL: {url}\n"
                                f"{content[:1000]}\n"
                            )

                except Exception as e:

                    st.warning(
                        f"Web search error: {e}"
                    )

        # --------------------------------------------------
        # ROUTE 4: GEMINI + OLLAMA
        # --------------------------------------------------

        final_context = (
            "UPLOADED DOCUMENT CONTEXT:\n"
            + pdf_context
            + "\n\nWEB SEARCH CONTEXT:\n"
            + web_context
        )

        final_context = final_context[:12000]

        system_prompt = """
You are an AI assistant and document chatbot.

Follow these instructions:

1. Answer the user's question clearly and simply.
2. If uploaded document context is relevant, use it as the main source.
3. If web search context is available and relevant, use it as additional information.
4. Do not invent product specifications, vehicle compatibility,
   screen sizes, Android versions, resolutions, or configurations.
5. If the uploaded document does not contain the requested information,
   clearly say that it was not found in the uploaded document.
6. If you use document information, mention the page number when available.
7. If the user asks a general knowledge question and no document
   information is relevant, answer using general knowledge.
8. Do not confuse information from the web with information from the PDF.
9. Use headings and bullet points when they improve readability.
"""

        prompt = f"""
{system_prompt}

{final_context}

USER QUESTION:
{query}

ANSWER:
"""

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    if client is None:

                        raise ValueError(
                            "Gemini API key is missing. "
                            "Check GEMINI_API_KEY in your .env file."
                        )

                    response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=prompt
                    )

                    answer = (
                        response.text
                        or "No response received from Gemini."
                    )

                except Exception as gemini_error:

                    st.warning(
                        "Gemini unavailable. "
                        "Trying local Ollama model."
                    )

                    try:

                        ollama_model = Ollama(
                            model="mistral"
                        )

                        answer = ollama_model.invoke(
                            prompt
                        )

                    except Exception as ollama_error:

                        answer = (
                            f"Gemini error: {gemini_error}\n\n"
                            f"Ollama error: {ollama_error}"
                        )

            st.markdown(answer)


# ==================================================
# 14. FOOTER
# ==================================================

#st.markdown("---")

#st.caption(
    #"Powered by Gemini + FAISS + OCR + Tavily + "
    #"Ollama + Google Calendar MCP"
#)
