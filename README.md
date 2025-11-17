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

## 🧠 Planned Tech Stack (subject to change during hackathon)

- **Backend**
  - Google **Gemini** – intent understanding & phrase generation
  - **Qdrant** – storing and retrieving the child’s preferred phrases
  - Python (FastAPI or similar)

- **Frontend**
  - **Streamlit** or **Gradio** – simple, clean UI for the demo

- **Other**
  - Text-to-Speech API for voice output

---

## 🏗 Repository Structure (initial skeleton)

```bash
backend/
  main.py          # entry point (to be implemented)
  gemini.py        # Gemini-related logic (to be implemented)
  qdrant.py        # Qdrant-related logic (to be implemented)
  tts.py           # text-to-speech logic (to be implemented)
  requirements.txt # dependencies (to be filled during hackathon)

frontend/
  app.py           # UI logic (to be implemented)
  icons/           # visual symbols (food, feelings, activities)
  assets/          # logos, misc visuals

prompts/
  suggestion_prompt.txt  # prompt drafts for AI suggestions
  category_prompt.txt    # prompt drafts for category behaviour
