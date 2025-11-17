from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from gtts import gTTS
import google.generativeai as genai

import qdrant_manager

# --- App bootstrap --------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent
PROMPT_PATH = PROJECT_ROOT / "prompts" / "suggestion_prompt.txt"

load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="EchoMind – Assistive Communication",
    page_icon="🗣️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Force light theme
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #f9fbff 0%, #f4f7fb 100%);
    }
    </style>
""", unsafe_allow_html=True)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-pro")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is missing. Set it in your .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# Initialize model lazily with error handling
@st.cache_resource
def get_gemini_model():
    try:
        return genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        st.error(f"Failed to initialize Gemini model '{MODEL_NAME}': {e}")
        st.info("Try setting GEMINI_MODEL in .env to 'gemini-pro' or 'gemini-1.5-pro'")
        st.stop()
        return None

CHILD_ID = "demo_child"

CATEGORY_CONFIG: Dict[str, str] = {
    "Body & Needs": "🍎",
    "Feelings & Sensory": "💛",
    "Activities & People": "🎨",
    "Help & Safety": "🆘",
}


# --- Helper functions ------------------------------------------------------ #


def init_session_state() -> None:
    defaults = {
        "stage": "intro",
        "selected_category": None,
        "latitude": None,
        "longitude": None,
        "location_name": None,
        "gps_requested": False,
        "options": [],
        "last_phrase": None,
        "audio_file": None,
        "play_triggered": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_current_datetime() -> Dict[str, str]:
    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "time_of_day": "morning" if now.hour < 12 else "afternoon" if now.hour < 17 else "evening",
    }


def render_gps_location() -> None:
    """Render GPS location component to request location from browser"""
    if not st.session_state.gps_requested:
        return
    
    # JavaScript to get GPS and store in sessionStorage
    # Since iframe is sandboxed, we rely on sessionStorage and parent window checking
    js_code = """
    <script>
    (function() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                function(position) {
                    const lat = position.coords.latitude;
                    const lng = position.coords.longitude;
                    
                    // Store in sessionStorage as strings
                    sessionStorage.setItem('gps_lat', String(lat));
                    sessionStorage.setItem('gps_lng', String(lng));
                    sessionStorage.setItem('gps_timestamp', String(Date.now()));
                    
                    console.log('GPS coordinates stored in sessionStorage:', lat, lng);
                    
                    // Try to send message to parent window
                    try {
                        window.parent.postMessage({
                            type: 'gps_coordinates',
                            lat: lat,
                            lng: lng
                        }, '*');
                    } catch (e) {
                        console.log('postMessage failed, coordinates in sessionStorage');
                    }
                },
                function(error) {
                    console.error('Geolocation error:', error);
                    alert('Unable to get location: ' + error.message + '. Please check your browser permissions.');
                },
                {
                    enableHighAccuracy: true,
                    timeout: 15000,
                    maximumAge: 0
                }
            );
        } else {
            alert('Geolocation is not supported by your browser.');
        }
    })();
    </script>
    """
    components.html(js_code, height=0)


def build_context(category: str) -> Dict[str, str]:
    datetime_info = get_current_datetime()
    location_str = ""
    if st.session_state.latitude and st.session_state.longitude:
        location_str = f"GPS coordinates: {st.session_state.latitude:.6f}, {st.session_state.longitude:.6f}"
        if st.session_state.location_name:
            location_str += f" ({st.session_state.location_name})"
    
    return {
        "child_id": CHILD_ID,
        "category": category,
        "date": datetime_info["date"],
        "time": datetime_info["time"],
        "day_of_week": datetime_info["day_of_week"],
        "time_of_day": datetime_info["time_of_day"],
        "location": location_str if location_str else "Location not available",
        "latitude": str(st.session_state.latitude) if st.session_state.latitude else None,
        "longitude": str(st.session_state.longitude) if st.session_state.longitude else None,
        "last_phrase": st.session_state.get("last_phrase"),
    }


def load_prompt_template() -> str:
    if PROMPT_PATH.exists():
        text = PROMPT_PATH.read_text(encoding="utf-8").strip()
        if text:
            return text
    return (
        "You help a non-verbal autistic child communicate using very short first-person "
        "phrases.\n"
        "Context:\n{context}\n\n"
        "Return a JSON object with a key `phrases` that maps to an array of exactly three objects. "
        "Each object must have two keys: `text` (a short literal phrase suitable for text-to-speech) "
        "and `emoji` (a single relevant emoji). The phrases must be literal, concrete, and avoid "
        "question marks, metaphors, or figurative language. Choose emojis that clearly represent "
        "the meaning of each phrase."
    )


def parse_model_output(raw_text: str) -> List[Dict[str, str]]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[-1]
    
    data = json.loads(cleaned)

    # Handle two possible formats:
    # Format 1: {"phrases": [...]} - dict with phrases key
    # Format 2: [...] - direct list
    if isinstance(data, dict):
        phrases = data.get("phrases", [])
    elif isinstance(data, list):
        phrases = data
    else:
        raise ValueError(f"Expected dict or list, got {type(data).__name__}")

    if not isinstance(phrases, list):
        raise ValueError("Expected 'phrases' to be a list")
    
    if len(phrases) != 3:
        raise ValueError(f"Expected exactly 3 phrases, got {len(phrases)}")
    
    result = []
    for item in phrases:
        # Handle both formats: dict with "text"/"emoji" or list [text, emoji]
        if isinstance(item, dict):
            text = item.get("text", "").strip()
            emoji = item.get("emoji", "").strip()
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            text = str(item[0]).strip()
            emoji = str(item[1]).strip()
        else:
            raise ValueError(f"Expected dict or [text, emoji] list, got {type(item).__name__}")
        
        if not text:
            raise ValueError("Phrase 'text' field is required and cannot be empty")
        if not emoji:
            raise ValueError("Phrase 'emoji' field is required and cannot be empty")
        
        result.append({"text": text, "emoji": emoji})
    
    return result


def generate_ai_options(category: str, context: Dict[str, str]) -> List[Dict[str, str]]:
    model = get_gemini_model()
    if not model:
        st.error("Gemini model is not available.")
        st.stop()
        return []

    prompt_template = load_prompt_template()
    context_lines = [
        f"Child ID: {context['child_id']}",
        f"Category: {context['category']}",
        f"Date: {context['date']}",
        f"Time: {context['time']}",
        f"Day of week: {context['day_of_week']}",
        f"Time of day: {context['time_of_day']}",
        f"Location: {context['location']}",
    ]
    if context.get("latitude") and context.get("longitude"):
        context_lines.append(f"GPS coordinates: {context['latitude']}, {context['longitude']}")
    if context.get("last_phrase"):
        context_lines.append(f"Last phrase spoken: {context['last_phrase']}")

    # Add personalization context from Qdrant if available
    try:
        if st.session_state.get("qdrant_initialized"):
            personalization = qdrant_manager.get_personalization_context(
                child_id=context["child_id"],
                category=category,
                context=context,
            )
            if personalization:
                context_lines.append(f"Personalization: {personalization}")
    except Exception as e:
        print(f"Warning: Could not get personalization context: {e}")

    prompt = prompt_template.format(context="\n".join(context_lines))

    try:
        response = model.generate_content(prompt)
        if not getattr(response, "text", "").strip():
            st.error("Gemini returned an empty response. Please try again.")
            st.stop()
            return []
        phrases = parse_model_output(response.text)
        return phrases
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse Gemini response as JSON: {e}")
        st.info("Gemini must return valid JSON with exactly 3 phrases, each containing 'text' and 'emoji' fields.")
        st.stop()
        return []
    except ValueError as e:
        st.error(f"Invalid response format from Gemini: {e}")
        st.info("Gemini must return exactly 3 phrases, each with 'text' and 'emoji' fields.")
        st.stop()
        return []
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        st.info("Please check your API key and model name in .env file.")
        st.stop()
        return []


def build_option_payload(category: str, phrases: List[Dict[str, str]]) -> List[Dict[str, str]]:
    options = []
    for idx, phrase_data in enumerate(phrases):
        options.append(
            {
                "id": idx,
                "text": phrase_data["text"],
                "emoji": phrase_data["emoji"],
            }
        )
    return options


def fetch_options(category: str) -> None:
    context = build_context(category)
    phrases = generate_ai_options(category, context)
    if not phrases:
        return  # Error already displayed by generate_ai_options
    st.session_state.options = build_option_payload(category, phrases)
    st.session_state.stage = "phrases"


def synthesize_audio(text: str) -> Optional[str]:
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        gTTS(text=text, lang="en").write_to_fp(tmp)
        tmp.close()
        return tmp.name
    except Exception as exc:  # pragma: no cover - Streamlit surface
        st.warning(f"Unable to generate audio: {exc}")
        return None


def reset_flow() -> None:
    st.session_state.stage = "intro"
    st.session_state.selected_category = None
    st.session_state.options = []
    st.session_state.last_phrase = None
    st.session_state.audio_file = None
    st.session_state.play_triggered = False


# --- UI Sections ----------------------------------------------------------- #


def inject_custom_css() -> None:
    """Inject custom CSS to match the prototype design - optimized for autistic users"""
    css = """
    <style>
    :root {
        --bg: #f9fbff;
        --card: #ffffff;
        --accent: #60be9b;
        --accent-soft: #d9f5e9;
        --text: #1a202c;
        --muted: #718096;
        --danger: #f56565;
        --radius-lg: 24px;
        --radius-md: 16px;
        --shadow-sm: 0 3px 14px rgba(15, 23, 42, 0.05);
        --shadow-md: 0 4px 18px rgba(15, 23, 42, 0.06);
        --shadow-lg: 0 8px 24px rgba(15, 23, 42, 0.06);
    }
    
    /* Force light theme */
    .stApp {
        background: linear-gradient(180deg, #f9fbff 0%, #f4f7fb 100%) !important;
    }
    
    /* Remove all Streamlit default styling */
    .main .block-container {
        max-width: 420px;
        padding: 1.5rem 1.5rem;
        background: transparent;
    }
    
    /* Hide Streamlit UI elements */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    
    /* Remove default button styling */
    button {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    .echomind-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        padding: 0.5rem 0;
    }
    
    .echomind-title-wrap {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .bubble-icon {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: var(--accent-soft);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }
    
    .echomind-title {
        font-weight: 700;
        font-size: 18px;
        margin: 0;
        color: var(--text);
    }
    
    .echomind-subtitle {
        font-size: 11px;
        color: var(--muted);
        margin: 0;
    }
    
    .pill {
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.8);
        font-size: 10px;
        color: var(--muted);
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    
    .pill-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #48bb78;
    }
    
    .badge {
        padding: 2px 6px;
        border-radius: 999px;
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        background: var(--accent-soft);
        color: var(--muted);
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .primary-btn-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
        margin: 2rem 0;
    }
    
    .primary-btn {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        font-size: 18px;
        font-weight: 700;
        background: var(--accent);
        color: white;
        border: none;
        box-shadow: 0 10px 30px rgba(96, 190, 155, 0.55);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 8px;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        touch-action: manipulation;
        -webkit-tap-highlight-color: transparent;
    }
    
    .primary-btn:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 12px 35px rgba(96, 190, 155, 0.65);
    }
    
    .primary-btn:active {
        transform: translateY(0px) scale(0.98);
        box-shadow: 0 6px 20px rgba(96, 190, 155, 0.5);
    }
    
    .primary-btn:focus-visible {
        outline: 3px solid rgba(96, 190, 155, 0.5);
        outline-offset: 4px;
    }
    
    .primary-btn-icon {
        width: 70px;
        height: 70px;
        border-radius: 50%;
        background: rgba(255,255,255,0.18);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
    }
    
    .hint {
        font-size: 12px;
        color: var(--muted);
        text-align: center;
        margin-top: 0.5rem;
    }
    
    .card {
        background: var(--card);
        border-radius: var(--radius-lg);
        padding: 18px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }
    
    .tile-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin: 1rem 0;
    }
    
    .tile-btn {
        padding: 18px 14px;
        border-radius: var(--radius-md);
        background: var(--card);
        box-shadow: var(--shadow-md);
        border: 2px solid transparent;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 10px;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        touch-action: manipulation;
        min-height: 140px;
        -webkit-tap-highlight-color: transparent;
    }
    
    .tile-btn:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
        border-color: var(--accent-soft);
    }
    
    .tile-btn:active {
        transform: translateY(0px) scale(0.98);
        box-shadow: var(--shadow-sm);
    }
    
    .tile-btn:focus-visible {
        outline: 3px solid var(--accent-soft);
        outline-offset: 2px;
    }
    
    .tile-icon-wrap {
        width: 64px;
        height: 64px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        margin-bottom: 4px;
        text-align: center;
        padding: 4px;
        flex-shrink: 0;
    }
    
    .tile-body .tile-icon-wrap {
        background: #ffeec2;
    }
    
    .tile-feelings .tile-icon-wrap {
        background: #e1e4ff;
    }
    
    .tile-activities .tile-icon-wrap {
        background: #ffd6e8;
    }
    
    .tile-safety .tile-icon-wrap {
        background: #ffe1e1;
    }
    
    .tile-label {
        font-size: 13px;
        font-weight: 600;
        text-align: center;
        color: var(--text);
    }
    
    .tile-sub {
        font-size: 10px;
        color: var(--muted);
        text-align: center;
    }
    
    .suggestion-list {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin: 1rem 0;
    }
    
    .suggestion-btn {
        width: 100%;
        text-align: left;
        padding: 16px 18px;
        border-radius: var(--radius-md);
        background: var(--card);
        border: 2px solid transparent;
        box-shadow: var(--shadow-sm);
        display: flex;
        align-items: center;
        gap: 12px;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        touch-action: manipulation;
        min-height: 60px;
        -webkit-tap-highlight-color: transparent;
    }
    
    .suggestion-btn:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.1);
        border-color: var(--accent-soft);
    }
    
    .suggestion-btn:active {
        transform: translateY(0px) scale(0.99);
        box-shadow: var(--shadow-sm);
    }
    
    .suggestion-btn:focus-visible {
        outline: 3px solid var(--accent-soft);
        outline-offset: 2px;
    }
    
    .suggestion-icon {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: var(--accent-soft);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        flex-shrink: 0;
    }
    
    .suggestion-text-main {
        font-size: 14px;
        font-weight: 600;
        color: var(--text);
    }
    
    .none-btn {
        width: 100%;
        padding: 14px 12px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        color: var(--danger);
        background: rgba(254, 226, 226, 0.7);
        border: 2px solid transparent;
        cursor: pointer;
        margin-top: 0.5rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        touch-action: manipulation;
        min-height: 48px;
    }
    
    .none-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(245, 101, 101, 0.2);
        border-color: rgba(245, 101, 101, 0.3);
    }
    
    .none-btn:active {
        transform: translateY(0px);
    }
    
    .back-btn {
        width: 100%;
        padding: 12px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.9);
        color: var(--muted);
        font-size: 13px;
        font-weight: 600;
        border: 2px solid transparent;
        box-shadow: 0 3px 12px rgba(15,23,42,0.06);
        cursor: pointer;
        margin-top: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        touch-action: manipulation;
        min-height: 48px;
    }
    
    .back-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 16px rgba(15,23,42,0.1);
        border-color: var(--accent-soft);
    }
    
    .back-btn:active {
        transform: translateY(0px);
    }
    
    .play-card {
        text-align: center;
        margin: 2rem 0;
    }
    
    .play-icon {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: var(--accent-soft);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 8px;
        font-size: 32px;
    }
    
    .play-phrase {
        font-size: 18px;
        font-weight: 700;
        margin: 1rem 0;
        color: var(--text);
    }
    
    .play-btn {
        margin-top: 10px;
        padding: 12px 20px;
        border-radius: 999px;
        background: var(--accent);
        color: white;
        font-size: 14px;
        font-weight: 600;
        border: none;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        touch-action: manipulation;
        min-height: 48px;
        box-shadow: 0 4px 12px rgba(96, 190, 155, 0.3);
    }
    
    .play-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(96, 190, 155, 0.4);
    }
    
    .play-btn:active {
        transform: translateY(0px);
        box-shadow: 0 2px 8px rgba(96, 190, 155, 0.3);
    }
    
    .play-btn:focus-visible {
        outline: 3px solid rgba(96, 190, 155, 0.5);
        outline-offset: 2px;
    }
    
    .section-title {
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: var(--text);
    }
    
    /* Accessibility improvements for autistic users */
    * {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Improve text readability */
    body, .stApp {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
        line-height: 1.6 !important;
    }
    
    /* Better contrast for text */
    .card p, .card h2 {
        color: var(--text) !important;
    }
    
    /* Larger touch targets - minimum 44x44px for accessibility */
    button {
        min-height: 44px !important;
        min-width: 44px !important;
    }
    
    /* Smooth animations - not jarring */
    * {
        transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    /* Remove distracting elements */
    .stApp > div:first-child {
        padding-top: 0 !important;
    }
    
    /* Better spacing for clarity */
    .card {
        margin-bottom: 1.5rem;
    }
    
    /* Ensure emojis are properly sized */
    .tile-icon-wrap, .suggestion-icon, .bubble-icon, .play-icon {
        font-size: inherit !important;
        line-height: 1 !important;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    
    /* Remove default Streamlit styling and ensure border radius */
    .stButton > button {
        width: 100%;
        border-radius: 16px !important;
    }
    
    /* Default button styling - light colors with border radius */
    button:not([kind="primary"]):not([data-testid*="cat-"]):not([data-testid*="phrase-"]):not([data-testid*="back_"]):not([data-testid*="none_"]):not([data-testid*="play_"]):not([data-testid*="back_home"]) {
        color: #1a202c !important;
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 16px !important;
    }
    
    /* Primary buttons keep their accent color */
    button[kind="primary"] {
        color: white !important;
        background: var(--accent) !important;
        border: none !important;
        border-radius: 16px !important;
    }
    
    /* Ensure all buttons have proper border radius as fallback */
    button {
        border-radius: 16px !important;
        user-select: none;
        -webkit-user-select: none;
    }
    
    /* Better focus indicators for keyboard navigation */
    *:focus-visible {
        outline-width: 3px !important;
        outline-style: solid !important;
        outline-offset: 2px !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_header() -> None:
    """Render the app header matching the prototype"""
    inject_custom_css()
    
    header_html = """
    <div class="echomind-header">
        <div class="echomind-title-wrap">
            <div class="bubble-icon">💬</div>
            <div>
                <div class="echomind-title">EchoMind</div>
                <div class="echomind-subtitle">Tap · Choose · Speak</div>
            </div>
        </div>
        <div class="pill">
            <span class="pill-dot"></span>
            Demo child
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Add GPS listener script once per page load
    if "gps_listener_added" not in st.session_state:
        listener_js = """
        <script>
        (function() {
            // Listen for postMessage from iframe
            window.addEventListener('message', function(event) {
                if (event.data && event.data.type === 'gps_coordinates') {
                    const lat = event.data.lat;
                    const lng = event.data.lng;
                    const currentUrl = window.location.href.split('?')[0];
                    const newUrl = currentUrl + '?lat=' + encodeURIComponent(lat) + '&lng=' + encodeURIComponent(lng) + '&gps_updated=true';
                    console.log('Updating URL from postMessage:', newUrl);
                    window.location.href = newUrl;
                }
            });
            
            // Check sessionStorage immediately and periodically as fallback
            function checkSessionStorage() {
                const lat = sessionStorage.getItem('gps_lat');
                const lng = sessionStorage.getItem('gps_lng');
                if (lat && lng && !window.location.search.includes('lat=')) {
                    console.log('Found GPS in sessionStorage, updating URL:', lat, lng);
                    const currentUrl = window.location.href.split('?')[0];
                    const newUrl = currentUrl + '?lat=' + encodeURIComponent(lat) + '&lng=' + encodeURIComponent(lng) + '&gps_updated=true';
                    window.location.href = newUrl;
                    return true;
                }
                return false;
            }
            
            // Check immediately
            checkSessionStorage();
            
            // Check more frequently when GPS might be requested
            setInterval(checkSessionStorage, 300);
        })();
        </script>
        """
        components.html(listener_js, height=0)
        st.session_state.gps_listener_added = True


def render_context_log() -> None:
    """Display GPS and time context information"""
    with st.expander("📊 Context Log (GPS & Time)", expanded=False):
        datetime_info = get_current_datetime()
        
        st.markdown("### Time Context")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Date:** {datetime_info['date']}")
            st.write(f"**Time:** {datetime_info['time']}")
        with col2:
            st.write(f"**Day of Week:** {datetime_info['day_of_week']}")
            st.write(f"**Time of Day:** {datetime_info['time_of_day']}")
        
        st.markdown("### GPS Context")
        if st.session_state.latitude and st.session_state.longitude:
            st.success(f"**Coordinates:** {st.session_state.latitude:.6f}, {st.session_state.longitude:.6f}")
            if st.session_state.location_name:
                st.write(f"**Location Name:** {st.session_state.location_name}")
        else:
            st.warning("**GPS:** Not available")
            if st.session_state.gps_requested:
                st.info("⏳ Waiting for browser location permission...")
        
        # Debug: Show query params if present
        query_params = st.query_params
        if query_params:
            st.markdown("### Debug: Query Parameters")
            st.json(dict(query_params))
        
        # Show what will be sent to Gemini
        st.markdown("### Context Sent to Gemini")
        context = build_context(st.session_state.selected_category or "N/A")
        context_lines = []
        for key, value in context.items():
            if value:
                if key == "last_phrase":
                    context_lines.append(f"- Last phrase spoken: {value}")
                else:
                    context_lines.append(f"- {key.replace('_', ' ').title()}: {value}")
        st.code("\n".join(context_lines), language="text")


def render_location_status() -> None:
    """Display GPS location status and button to request location"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.session_state.gps_requested:
            st.warning("🔄 Requesting location... Please allow location access in your browser.")
        elif st.session_state.latitude and st.session_state.longitude:
            st.success(f"📍 Location: {st.session_state.latitude:.6f}, {st.session_state.longitude:.6f}")
        else:
            st.info("📍 Location: Not available")
    
    with col2:
        if st.button("📍 Get Location", help="Request GPS location from your device", disabled=st.session_state.gps_requested):
            st.session_state.gps_requested = True
            st.rerun()
    
    # Render GPS component if requested
    render_gps_location()


def render_stage_intro() -> None:
    """Render the intro screen with large circular button"""
    card_html = """
    <div class="card">
        <span class="badge">Step 1 · Activation</span>
        <h2 style="margin:10px 0 6px;font-size:18px;color:var(--text);">I want to speak</h2>
        <p style="margin:0;font-size:13px;color:var(--muted);">
            One big, calm button in the center. The child taps once to tell the system
            "I want to say something." This minimizes decisions and motor effort.
        </p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Add custom styling for circular button - larger and more accessible
    st.markdown("""
    <style>
    button[kind="primary"][data-testid*="speak"] {
        width: 200px !important;
        height: 200px !important;
        border-radius: 50% !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 10px 30px rgba(96, 190, 155, 0.55) !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
        margin: 2rem auto !important;
        white-space: pre-line !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        touch-action: manipulation !important;
        -webkit-tap-highlight-color: transparent !important;
    }
    button[kind="primary"][data-testid*="speak"]:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 35px rgba(96, 190, 155, 0.65) !important;
    }
    button[kind="primary"][data-testid*="speak"]:active {
        transform: translateY(0px) scale(0.98) !important;
        box-shadow: 0 6px 20px rgba(96, 190, 155, 0.5) !important;
    }
    button[kind="primary"][data-testid*="speak"]:focus-visible {
        outline: 3px solid rgba(96, 190, 155, 0.5) !important;
        outline-offset: 4px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎙️\n\nI want to speak", use_container_width=True, type="primary", key="speak_main"):
            st.session_state.stage = "categories"
            st.rerun()
    
    st.markdown('<p class="hint">Tap once to start. No menus, no scrolling, no text search.</p>', unsafe_allow_html=True)


def render_categories() -> None:
    """Render category selection screen with tiles"""
    card_html = """
    <div class="card">
        <span class="badge">Step 2 · High-level choice</span>
        <h2 style="margin:10px 0 4px;font-size:17px;color:var(--text);">What is it about?</h2>
        <p style="margin:0;font-size:12px;color:var(--muted);">
            We keep only four big categories so the child is never overwhelmed:
            body needs, feelings & sensory, activities & people, and help & safety.
        </p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Category tiles in 2x2 grid
    category_info = {
        "Body & Needs": {
            "emoji": "🍎",
            "class": "tile-body"
        },
        "Feelings & Sensory": {
            "emoji": "💛",
            "class": "tile-feelings"
        },
        "Activities & People": {
            "emoji": "🎨",
            "class": "tile-activities"
        },
        "Help & Safety": {
            "emoji": "🆘",
            "class": "tile-safety"
        }
    }
    
    # Create 2x2 grid using columns
    col1, col2 = st.columns(2)
    categories_list = list(CATEGORY_CONFIG.items())
    
    # Add styling for category buttons - light colors with proper border radius
    st.markdown("""
    <style>
    button[data-testid*="cat-"] {
        padding: 20px 16px !important;
        border-radius: 16px !important;
        background: #ffffff !important;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.08) !important;
        border: 2px solid #e2e8f0 !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 12px !important;
        min-height: 140px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #1a202c !important;
        transition: all 0.15s ease !important;
        touch-action: manipulation !important;
        -webkit-tap-highlight-color: transparent !important;
        white-space: pre-line !important;
    }
    button[data-testid*="cat-"]:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12) !important;
        border-color: var(--accent) !important;
        background: #f8fafc !important;
    }
    button[data-testid*="cat-"]:active {
        transform: translateY(0px) scale(0.98) !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08) !important;
    }
    button[data-testid*="cat-"]:focus-visible {
        outline: 3px solid var(--accent-soft) !important;
        outline-offset: 2px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with col1:
        for idx in range(0, len(categories_list), 2):
            label, emoji = categories_list[idx]
            info = category_info[label]
            button_text = f"{emoji}\n\n{label}"
            if st.button(button_text, key=f"cat-{idx}", use_container_width=True):
                st.session_state.selected_category = label
                st.session_state.stage = "loading"
                st.rerun()

    with col2:
        for idx in range(1, len(categories_list), 2):
            label, emoji = categories_list[idx]
            info = category_info[label]
            button_text = f"{emoji}\n\n{label}"
            if st.button(button_text, key=f"cat-{idx}", use_container_width=True):
                st.session_state.selected_category = label
                st.session_state.stage = "loading"
                st.rerun()
    
    # Style back button - improved accessibility
    st.markdown("""
    <style>
    button[data-testid*="back_intro"] {
        background: rgba(255,255,255,0.9) !important;
        color: var(--muted) !important;
        border: 2px solid transparent !important;
        box-shadow: 0 3px 12px rgba(15,23,42,0.06) !important;
        border-radius: 999px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        padding: 12px 14px !important;
        margin-top: 1rem !important;
        min-height: 48px !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease !important;
        touch-action: manipulation !important;
    }
    button[data-testid*="back_intro"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 16px rgba(15,23,42,0.1) !important;
        border-color: var(--accent-soft) !important;
    }
    button[data-testid*="back_intro"]:active {
        transform: translateY(0px) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("← Back to \"I want to speak\"", key="back_intro", use_container_width=True):
        reset_flow()


def render_phrase_options() -> None:
    """Render phrase suggestions screen"""
    category = st.session_state.selected_category
    
    card_html = f"""
    <div class="card">
        <span class="badge">Step 3 · AI suggestions</span>
        <h2 style="margin:10px 0 4px;font-size:17px;color:var(--text);">What do you want to say?</h2>
        <p style="margin:0;font-size:12px;color:var(--muted);">
            Based on the category and context, EchoMind suggests three short phrases.
            Each phrase is generated by Gemini AI with a matching emoji.
        </p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Suggestion list
    st.markdown('<div class="section-title">Tap a sentence</div>', unsafe_allow_html=True)
    
    # Add styling for suggestion buttons - light colors with proper border radius
    st.markdown("""
    <style>
    button[data-testid*="phrase-"] {
        text-align: left !important;
        padding: 18px 20px !important;
        border-radius: 16px !important;
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        box-shadow: 0 3px 14px rgba(15, 23, 42, 0.06) !important;
        display: flex !important;
        align-items: center !important;
        gap: 14px !important;
        margin-bottom: 12px !important;
        min-height: 64px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #1a202c !important;
        transition: all 0.15s ease !important;
        touch-action: manipulation !important;
        -webkit-tap-highlight-color: transparent !important;
    }
    button[data-testid*="phrase-"]:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.12) !important;
        border-color: var(--accent) !important;
        background: #f8fafc !important;
    }
    button[data-testid*="phrase-"]:active {
        transform: translateY(0px) scale(0.99) !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08) !important;
    }
    button[data-testid*="phrase-"]:focus-visible {
        outline: 3px solid var(--accent-soft) !important;
        outline-offset: 2px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    for idx, option in enumerate(st.session_state.options):
        button_text = f"{option['emoji']}  {option['text']}"
        if st.button(button_text, key=f"phrase-{idx}", use_container_width=True):
            st.session_state.last_phrase = option["text"]
            st.session_state.audio_file = synthesize_audio(option["text"])

            # Store the phrase selection in Qdrant for personalization
            try:
                if st.session_state.get("qdrant_initialized"):
                    context = build_context(st.session_state.selected_category)
                    qdrant_manager.store_phrase(
                        child_id=CHILD_ID,
                        category=st.session_state.selected_category,
                        phrase=option["text"],
                        context=context,
                    )
            except Exception as e:
                print(f"Warning: Could not store phrase in Qdrant: {e}")

            st.session_state.stage = "voice"
            st.rerun()
    
    # Style the "None of these" button - improved accessibility
    st.markdown("""
    <style>
    button[data-testid*="none_btn"] {
        color: var(--danger) !important;
        background: rgba(254, 226, 226, 0.7) !important;
        border: 2px solid transparent !important;
        border-radius: 999px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        padding: 14px 12px !important;
        margin-top: 0.5rem !important;
        min-height: 48px !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease !important;
        touch-action: manipulation !important;
    }
    button[data-testid*="none_btn"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(245, 101, 101, 0.2) !important;
        border-color: rgba(245, 101, 101, 0.3) !important;
    }
    button[data-testid*="none_btn"]:active {
        transform: translateY(0px) !important;
    }
    button[data-testid*="back_categories"] {
        background: rgba(255,255,255,0.9) !important;
        color: var(--muted) !important;
        border: 2px solid transparent !important;
        box-shadow: 0 3px 12px rgba(15,23,42,0.06) !important;
        border-radius: 999px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        padding: 12px 14px !important;
        margin-top: 1rem !important;
        min-height: 48px !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease !important;
        touch-action: manipulation !important;
    }
    button[data-testid*="back_categories"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 16px rgba(15,23,42,0.1) !important;
        border-color: var(--accent-soft) !important;
    }
    button[data-testid*="back_categories"]:active {
        transform: translateY(0px) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("✖ None of these · show more options", key="none_btn", use_container_width=True):
        fetch_options(category)
    
    if st.button("← Back to categories", key="back_categories", use_container_width=True):
        st.session_state.stage = "categories"
        st.rerun()


def render_voice_output() -> None:
    """Render voice output screen"""
    if not st.session_state.last_phrase:
        st.session_state.stage = "phrases"
        st.rerun()
        return
    
    card_html = """
    <div class="card play-card">
        <span class="badge">Step 4 · Voice output</span>
        <div class="play-icon">🔊</div>
        <div class="play-phrase">{phrase}</div>
        <p style="margin:0;font-size:12px;color:var(--muted);">
            The system speaks this sentence out loud for the child.
            The icon and phrase stay on screen so the adult sees what was said.
        </p>
    </div>
    """.format(phrase=st.session_state.last_phrase)
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Style buttons - improved accessibility
    st.markdown("""
    <style>
    button[data-testid*="play_again"] {
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: 999px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        padding: 12px 20px !important;
        margin-top: 10px !important;
        min-height: 48px !important;
        box-shadow: 0 4px 12px rgba(96, 190, 155, 0.3) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        touch-action: manipulation !important;
    }
    button[data-testid*="play_again"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(96, 190, 155, 0.4) !important;
    }
    button[data-testid*="play_again"]:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 8px rgba(96, 190, 155, 0.3) !important;
    }
    button[data-testid*="play_again"]:focus-visible {
        outline: 3px solid rgba(96, 190, 155, 0.5) !important;
        outline-offset: 2px !important;
    }
    button[data-testid*="back_home"] {
        background: rgba(255,255,255,0.9) !important;
        color: var(--muted) !important;
        border: 2px solid transparent !important;
        box-shadow: 0 3px 12px rgba(15,23,42,0.06) !important;
        border-radius: 999px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-decoration: none !important;
        margin-top: 1rem !important;
        padding: 12px 14px !important;
        min-height: 48px !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease !important;
        touch-action: manipulation !important;
    }
    button[data-testid*="back_home"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 16px rgba(15,23,42,0.1) !important;
        border-color: var(--accent-soft) !important;
    }
    button[data-testid*="back_home"]:active {
        transform: translateY(0px) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.session_state.audio_file:
        st.audio(st.session_state.audio_file, autoplay=True)
    
    if st.button("▶ Play again", key="play_again", use_container_width=True):
        st.session_state.play_triggered = True

    # Show audio again if play button was clicked
    if st.session_state.get("play_triggered", False) and st.session_state.audio_file:
        st.audio(st.session_state.audio_file, autoplay=False)
    
    if st.button("← Back to \"I want to speak\"", key="back_home", use_container_width=True):
        reset_flow()
        st.rerun()


# --- Main render ----------------------------------------------------------- #


def main() -> None:
    init_session_state()

    # Initialize Qdrant on first run
    if "qdrant_initialized" not in st.session_state:
        try:
            qdrant_manager.init_qdrant()
            st.session_state.qdrant_initialized = True
        except Exception as e:
            st.warning(f"⚠️ Qdrant initialization failed: {e}. App will work without personalization.")
            st.session_state.qdrant_initialized = False

    # Check for GPS data in query parameters (from JavaScript geolocation)
    query_params = st.query_params
    
    # Try to get from query params first
    if "lat" in query_params and "lng" in query_params:
        try:
            lat_str = query_params.get("lat")
            lng_str = query_params.get("lng")
            # Handle both single values and lists
            lat = float(lat_str[0] if isinstance(lat_str, list) else lat_str)
            lng = float(lng_str[0] if isinstance(lng_str, list) else lng_str)
            
            if lat and lng:
                st.session_state.latitude = lat
                st.session_state.longitude = lng
                st.session_state.gps_requested = False
                
                # Clear query params
                new_params = {k: v for k, v in query_params.items() if k not in ["lat", "lng", "gps_updated"]}
                st.query_params.clear()
                for k, v in new_params.items():
                    st.query_params[k] = v
                
                st.rerun()
        except (ValueError, TypeError, IndexError) as e:
            st.warning(f"Error parsing GPS coordinates from URL: {e}")
    
    
    render_header()
    
    # Hide context log for now (GPS disabled)
    # render_context_log()

    stage = st.session_state.stage
    if stage == "intro":
        render_stage_intro()
    elif stage == "categories":
        render_categories()
    elif stage == "loading":
        # Fetch options and transition to phrases
        st.spinner("Loading phrases...")
        fetch_options(st.session_state.selected_category)
        st.rerun()
    elif stage == "phrases":
        render_phrase_options()
    elif stage == "voice":
        render_voice_output()


if __name__ == "__main__":
    main()
