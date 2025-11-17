# EchoMind – Assistive AI Communication for Autistic Children

EchoMind is a simple, visual, AI-powered communication tool for **non-verbal or minimally verbal autistic children**.

The core idea is:

1. The child taps **“I want to speak”**
2. They choose from **3 big visual categories** (e.g. Food, Feelings, Activities)
3. The system shows **3 AI-generated suggestions**, each with an image and simple text
4. The child taps one, and the device **speaks the phrase out loud**

This keeps the interaction **very low effort and low cognitive load** while still giving the child agency and choice.

> ⚠️ At this stage (pre-hackathon), this repository only contains the **project structure**.  
> All functional code will be created during the hackathon to comply with the “original work” rules.

---

## 🎯 Goal

Many existing AAC apps are powerful but complex, with dense grids and many navigation steps.  
EchoMind focuses on **speed, simplicity, and visual clarity**:

- One clear entry point: **“I want to speak”**
- Very limited choices at each step (3 → 3)
- Image-first design for children who cannot read
- Natural voice output so the child can “speak” in real-world situations

---

## 🧠 MVP Tech Stack

- **Language**: Python
- **Frontend + Logic**: Streamlit (`frontend/app.py`)
- **LLM**: Google Gemini (phrase generation)
- **TTS**: gTTS (converts the chosen phrase to audio)
- **Env config**: `python-dotenv`

---

## 🚀 Getting Started

1. **Install dependencies**

   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Environment variables**

   Create a `.env` file in the project root:

   ```
   GEMINI_API_KEY=your_key
   GEMINI_MODEL=gemini-1.5-flash
   ```

3. **Images (optional but recommended)**

   Add category images under `frontend/images/`:

   ```
   frontend/images/
     food/
       apple.png
       snack.png
     feelings/
       calm.png
       break.png
     activities/
       play.png
       outside.png
   ```

   The app automatically pulls whatever files exist inside each folder; filenames do not need to match exactly.

4. **Run Streamlit**

   ```bash
   streamlit run frontend/app.py
   ```

5. **Demo flow**

   - Tap **“I want to speak.”**
   - Choose **Food**, **Feelings**, or **Activities**.
   - Review the **three Gemini-generated phrases** (each with an image or emoji).
   - Tap one → **audio playback** uses gTTS.

---

## 🧩 Additional Notes

- If Gemini is unavailable (no key or API error), the UI falls back to safe, static phrases per category.
- Context (location + time of day + last spoken phrase) is automatically included in prompts so Gemini can stay literal and relevant.
- `prompts/suggestion_prompt.txt` can be edited to tweak tone/format; the Streamlit app loads it automatically when it’s non-empty.
- This MVP intentionally skips persistent storage (e.g., Qdrant) to keep hackathon setup fast. A future iteration can re-introduce personal memory without changing the UI flow.
