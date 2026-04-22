import gradio as gr
import base64
import json
import os
import io
from PIL import Image
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── NVIDIA NIM Client ──────────────────────────────────────────────────────────
def _get_api_key() -> str:
    return os.environ.get("NVIDIA_API_KEY", "")

def _get_client() -> OpenAI:
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=_get_api_key(),
        timeout=60.0,
    )

def _get_groq_client() -> Groq:
    """Dedicated client for Groq (Vision OCR)"""
    groq_key = os.environ.get("GROQ_API_KEY", "")
    return Groq(api_key=groq_key)

# ── Prompts ─────────────────────────────────────────────────────────────────────
# OCR prompt: extract raw text ONLY — no formatting, no LaTeX, no JSON.
_OCR_PROMPT = (
    "Transcribe all text and mathematical notation from this image EXACTLY as it appears. "
    "Preserve line breaks, spacing, variables, exponents, fractions, and symbols. "
    "Do NOT attempt to format anything as LaTeX, Markdown, or JSON. "
    "Return ONLY the raw plain text. No explanations."
)

# Reasoning prompt: expert tutor that identifies specific concepts.
_REASONING_SYSTEM_PROMPT = """You are an expert mathematics tutor.

When given a math question, respond ONLY with this exact JSON format:

{
  "formatted_question": "Complete question with math in $...$ using LaTeX like \\frac{dy}{dx}. Do NOT include multiple choice options here.",
  "options": ["(A) ...", "(B) ..."],
  "concepts": ["most specific formula or rule 1", "most specific formula or rule 2"],
  "question": "Do you know the concept of [concept 1] and [concept 2]?"
}

Rules:
- Output ONLY valid JSON. No text before or after.
- formatted_question: The question text only. Use single backslashes in LaTeX (e.g. \\frac, \\int). Never include (A), (B), etc. options.
- options: Extract all multiple-choice options. If none, return [].
- concepts: List most specific concepts needed (e.g. "Chain rule for composite functions").
- question: End with "Do you know the concept of [concept 1] and [concept 2]?" """

# ── AI Helpers ─────────────────────────────────────────────────────────────────

def extract_math_from_image(groq_client: Groq, img_bytes: bytes) -> str:
    """
    Extract text using Groq's vision model for OCR.
    """
    # Resize/Compress for speed
    img = Image.open(io.BytesIO(img_bytes))
    max_size = 1024
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    image_b64 = base64.b64encode(buffered.getvalue()).decode()

    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _OCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        max_tokens=1024,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip().replace('\n', ' ')

def generate_tutor_response(client: OpenAI, question_text: str) -> dict:
    """
    Process the combined question using a fallback chain of models.
    Tries Mistral-Nemotron -> Qwen.
    """
    models_to_try = [
        "mistralai/mistral-nemotron",
        "qwen/qwen3-coder-480b-a35b-instruct"
    ]

    raw = ""
    for model_name in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": _REASONING_SYSTEM_PROMPT},
                    {"role": "user",   "content": question_text},
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0.2,
                timeout=15.0,
            )
            raw = (response.choices[0].message.content or "").strip()

            if raw:
                break
        except Exception as exc:
            pass

    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(raw)
        return {
            "formatted_question": data.get("formatted_question", ""),
            "options":            data.get("options", []),
            "concepts":           data.get("concepts", []),
            "question":           data.get("question", ""),
        }
    except json.JSONDecodeError:
        return {
            "formatted_question": raw[:400],
            "options":            [],
            "concepts":           [],
            "question":           "",
        }

def build_ai_response(user_text: str, image_bytes_list: list) -> dict:
    """
    Orchestrates the two-model pipeline.
    """
    client = _get_client()
    groq_client = _get_groq_client()

    question_parts = []

    for img_bytes in image_bytes_list:
        extracted = extract_math_from_image(groq_client, img_bytes)
        if extracted:
            question_parts.append(extracted)

    if user_text.strip():
        question_parts.append(user_text.strip())

    combined_question = "\n".join(question_parts).strip()

    if not combined_question:
        return {
            "formatted_question": "Please type a math question or upload a clear image of one.",
            "concepts": [],
            "question": "",
        }

    return generate_tutor_response(client, combined_question)

# ── UI Functions ──────────────────────────────────────────────────────────────

def format_assistant_response(data: dict) -> str:
    """Format the AI response for display in chatbot."""
    msg = ""
    if data.get("formatted_question"):
        msg += data["formatted_question"] + "\n\n"
    if data.get("options"):
        for opt in data["options"]:
            msg += opt + "\n"
        msg += "\n"
    if data.get("concepts"):
        msg += "**Core concepts you will need:**\n"
        for c in data["concepts"]:
            msg += f"- {c}\n"
        msg += "\n"
    if data.get("question"):
        msg += f"*{data['question']}*"
    return msg

def show_crop_editor(uploaded_file):
    """No cropping interface, just upload."""
    return gr.ImageEditor(visible=False)

def submit_message(text: str, uploaded_file, history: list):
    """Handle message submission."""
    if not text and not uploaded_file:
        return history, "", None

    user_msg = text or ""
    image_bytes_list = []

    if uploaded_file:
        img = Image.open(uploaded_file)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        img_bytes = buf.getvalue()
        image_bytes_list.append(img_bytes)
        user_msg += " [Image uploaded and processed]"

    # Generate AI response
    ai_data = build_ai_response(text, image_bytes_list)
    assistant_msg = format_assistant_response(ai_data)

    # Update history
    new_history = history + [(user_msg, assistant_msg)]

    return new_history, "", None

# ── Main App ───────────────────────────────────────────────────────────────────

def create_app():
    css = """
    .gradio-container {
        background: #ffffff !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .chatbot {
        border: 1px solid #e1e5e9;
        border-radius: 8px;
    }
    .textbox {
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }
    .button {
        border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        font-weight: 500;
    }
    .button:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    """
    with gr.Blocks(title="StepSolve", css=css) as app:
        gr.Markdown("# StepSolve - Learn Mathematics Step by Step")

        chatbot = gr.Chatbot(height=500)
        history_state = gr.State([])

        with gr.Row():
            text_input = gr.Textbox(
                placeholder="Type your math question...",
                scale=4,
                container=False
            )
            file_input = gr.File(
                label="📎",
                file_types=["image"],
                scale=1,
                container=False
            )

        submit_btn = gr.Button("Solve", variant="primary", scale=1)

        # Event handlers
        submit_btn.click(
            fn=submit_message,
            inputs=[text_input, file_input, history_state],
            outputs=[chatbot, text_input, file_input]
        ).then(
            fn=lambda h: h,
            inputs=history_state,
            outputs=history_state
        )

    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(theme=gr.themes.Soft())
