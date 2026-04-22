"""
StepSolve - Mathematics Tutor App built with NiceGUI

This app uses a two-model system:
1. OCR Model: Extracts clean LaTeX + text from uploaded images
2. Main Reasoning Model: mistral-nemotron (primary) with Qwen as fallback

Features:
- Beautiful ChatGPT/Grok-style interface with Tailwind styling
- Image upload with automatic cropping modal (Google Lens style)
- Settings panel for NVIDIA API key
- Math rendering with KaTeX
"""

import base64
import io
import json
import os
from PIL import Image
from openai import OpenAI
from nicegui import ui, app
from dotenv import load_dotenv

load_dotenv()

# ── Prompts ─────────────────────────────────────────────────────────────────────

# OCR Prompt (for vision model):
OCR_PROMPT = """
You are an expert mathematics OCR extractor.
Your ONLY job is to read the uploaded image and return the question as clean, accurate text with proper LaTeX.
- Extract the COMPLETE question exactly as written.
- Convert all math into clean inline LaTeX.
- Use exactly TWO backslashes for every LaTeX command (\\\\frac, \\\\sin, \\\\theta, etc.).
- Output ONLY the cleaned question text. No extra text.
"""

# Reasoning Prompt (for main tutor model):
REASONING_PROMPT = """
You are StepSolve, an expert mathematics tutor.

When given a math question, respond ONLY in this exact JSON format — no extra text:

{
  "formatted_question": "Complete question with math in $...$ using LaTeX like \\frac{dy}{dx}. Do NOT include multiple choice options here.",
  "options": ["(A) ...", "(B) ..."],
  "concepts": ["most specific formula or rule 1", "most specific formula or rule 2"],
  "question": "Do you know the concept of [concept 1] and [concept 2]?"
}

Rules:
- concepts must be extremely granular and specific (never broad like "Differentiation")
- Always use the exact question format at the end
- Never give any solution or steps
"""

# ── State Management ────────────────────────────────────────────────────────────

class AppState:
    """Manage application state including chat history and settings."""
    
    def __init__(self):
        self.chat_history = []
        self.api_key = os.environ.get("NVIDIA_API_KEY", "")
        self.pending_image = None
        self.cropped_image = None
        
    def add_message(self, role: str, content: str, image_data: str = None):
        """Add a message to chat history."""
        self.chat_history.append({
            "role": role,
            "content": content,
            "image": image_data
        })
        
    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []
        
    def set_api_key(self, key: str):
        """Set the NVIDIA API key."""
        self.api_key = key

# Global state instance
state = AppState()

# ── NVIDIA NIM Client ──────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    """Get OpenAI-compatible client for NVIDIA NIM."""
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=state.api_key,
        timeout=60.0,
    )

# ── AI Helpers ─────────────────────────────────────────────────────────────────

def extract_math_from_image(img_bytes: bytes) -> str:
    """
    Extract text from image using NVIDIA's vision model for OCR.
    Returns clean LaTeX + text.
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

    client = get_client()
    
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
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

def generate_tutor_response(question_text: str) -> dict:
    """
    Process the question using a fallback chain of models.
    Tries Mistral-Nemotron -> Qwen.
    """
    models_to_try = [
        "mistralai/mistral-nemotron",
        "qwen/qwen3-coder-480b-a35b-instruct"
    ]

    raw = ""
    client = get_client()
    
    for model_name in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": REASONING_PROMPT},
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
            continue

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
    1. OCR model extracts text from images
    2. Reasoning model processes the combined question
    """
    question_parts = []

    for img_bytes in image_bytes_list:
        extracted = extract_math_from_image(img_bytes)
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

    return generate_tutor_response(combined_question)

# ── UI Components ───────────────────────────────────────────────────────────────

def render_response(data: dict) -> str:
    """Render the AI response as formatted HTML with KaTeX support."""
    html = ""
    
    if data.get("formatted_question"):
        html += f'<div class="text-lg mb-3">{data["formatted_question"]}</div>'
    
    if data.get("options"):
        html += '<div class="mb-3">'
        for opt in data["options"]:
            html += f'<div class="ml-4 py-1">{opt}</div>'
        html += '</div>'
    
    if data.get("concepts"):
        html += '<div class="font-semibold mt-4 mb-2">Core concepts you will need:</div>'
        html += '<ul class="list-disc ml-6">'
        for c in data["concepts"]:
            html += f'<li class="py-1">{c}</li>'
        html += '</ul>'
    
    if data.get("question"):
        html += f'<div class="mt-4 italic text-gray-600">{data["question"]}</div>'
    
    return html

def create_chat_message(role: str, content: str, image_data: str = None) -> None:
    """Create a chat message element."""
    is_user = role == "user"
    
    with ui.element('div').classes(f'flex w-full {"justify-end" if is_user else "justify-start"} mb-4'):
        with ui.element('div').classes(f'max-w-[80%] rounded-2xl px-4 py-3 {"bg-blue-500 text-white" if is_user else "bg-gray-100 text-gray-800"}'):
            if image_data:
                with ui.element('div').classes('mb-2'):
                    ui.image(image_data).classes('max-h-48 rounded-lg')
            
            if is_user:
                ui.label(content).classes('whitespace-pre-wrap')
            else:
                # Assistant response with rendered HTML
                ui.html(render_response(json.loads(content) if content.startswith('{') else {"formatted_question": content, "options": [], "concepts": [], "question": ""}))

def refresh_chat():
    """Refresh the chat display."""
    chat_container.clear()
    with chat_container:
        for msg in state.chat_history:
            create_chat_message(msg["role"], msg["content"], msg.get("image"))

# ── Cropping Modal ──────────────────────────────────────────────────────────────

cropper_dialog = None
cropper_image = None
cropped_result = None

def open_cropper(image_file):
    """Open the cropping modal when an image is uploaded."""
    global cropped_result
    cropped_result = None
    
    # Read and store the image
    img = Image.open(image_file)
    
    # Convert to base64 for display
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    img_data_uri = f"data:image/png;base64,{img_base64}"
    
    state.pending_image = img
    
    # Show the cropper dialog
    with ui.dialog() as dialog, ui.card().classes('w-[600px] p-4'):
        ui.label('Crop the image to focus on the math problem').classes('text-lg font-semibold mb-2')
        
        # Display the image
        with ui.element('div').classes('relative overflow-hidden border rounded-lg'):
            ui.image(img_data_uri).classes('w-full')
        
        ui.label('Note: For advanced cropping, the full image will be processed. You can describe which part to focus on in your message.').classes('text-sm text-gray-500 mt-2')
        
        with ui.row().classes('w-full justify-end mt-4 gap-2'):
            ui.button('Cancel', on_click=dialog.close).props('flat color=grey')
            ui.button('Use Full Image', icon='check', on_click=lambda: confirm_crop(dialog)).props('color=primary')
    
    dialog.open()

def confirm_crop(dialog):
    """Confirm using the full image (no cropping for now)."""
    global cropped_result
    cropped_result = state.pending_image
    dialog.close()
    
    # Auto-submit after crop confirmation
    if cropped_result and send_input.value:
        handle_send()
    elif cropped_result:
        # If only image was uploaded
        handle_send()

# ── Message Handler ─────────────────────────────────────────────────────────────

send_input = None
chat_container = None
loading_spinner = None

def handle_send():
    """Handle sending a message."""
    global cropped_result
    
    text = send_input.value if send_input else ""
    
    if not text and not state.pending_image and not cropped_result:
        return
    
    # Process image if present
    image_bytes_list = []
    user_image_data = None
    
    img_to_process = cropped_result or state.pending_image
    
    if img_to_process:
        buf = io.BytesIO()
        if isinstance(img_to_process, Image.Image):
            img_to_process.save(buf, format="JPEG", quality=95)
        else:
            img_to_process.save(buf, format="JPEG", quality=95)
        img_bytes = buf.getvalue()
        image_bytes_list.append(img_bytes)
        
        # Create display image
        display_img = Image.open(io.BytesIO(img_bytes))
        display_buf = io.BytesIO()
        display_img.save(display_buf, format="PNG")
        user_image_data = "data:image/png;base64," + base64.b64encode(display_buf.getvalue()).decode()
    
    user_msg = text.strip() if text else "[Image uploaded]"
    
    # Add user message to history
    state.add_message("user", user_msg, user_image_data)
    refresh_chat()
    
    # Clear inputs
    if send_input:
        send_input.value = ""
    state.pending_image = None
    cropped_result = None
    
    # Show loading
    with chat_container:
        with ui.element('div').classes('flex w-full justify-start mb-4'):
            with ui.element('div').classes('bg-gray-100 rounded-2xl px-4 py-3'):
                ui.spinner('dots', size='md').classes('text-gray-500')
    
    # Generate AI response in background
    ui.navigate.to('#chat-end')
    
    try:
        ai_data = build_ai_response(text, image_bytes_list)
        
        # Remove loading spinner
        refresh_chat()
        
        # Add assistant response
        state.add_message("assistant", json.dumps(ai_data))
        refresh_chat()
        
    except Exception as e:
        # Remove loading spinner
        refresh_chat()
        state.add_message("assistant", json.dumps({
            "formatted_question": f"Error: {str(e)}",
            "options": [],
            "concepts": [],
            "question": ""
        }))
        refresh_chat()
        ui.notify(f"Error: {str(e)}", color='negative')
    
    ui.navigate.to('#chat-end')

def handle_file_upload(e):
    """Handle file upload event."""
    if e.content:
        # Get the uploaded file
        files = e.content
        if files:
            # Open the first file
            with open(files[0], 'rb') as f:
                img = Image.open(f)
                state.pending_image = img
            # Open cropper
            open_cropper(files[0])

def clear_chat():
    """Clear the chat history."""
    state.clear_history()
    refresh_chat()
    ui.notify("Chat cleared", color='info')

# ── Settings Dialog ─────────────────────────────────────────────────────────────

def open_settings():
    """Open the settings dialog."""
    with ui.dialog() as dialog, ui.card().classes('w-[400px] p-6'):
        ui.label('Settings').classes('text-xl font-semibold mb-4')
        
        with ui.input(label='NVIDIA API Key', password=True, password_toggle_button=True) as api_key_input:
            api_key_input.value = state.api_key
            api_key_input.classes('w-full')
        
        ui.label('Get your API key from https://build.nvidia.com').classes('text-sm text-gray-500 mt-2')
        
        with ui.row().classes('w-full justify-end mt-4 gap-2'):
            ui.button('Cancel', on_click=dialog.close).props('flat color=grey')
            def save_settings():
                state.set_api_key(api_key_input.value)
                ui.notify("API key saved", color='positive')
                dialog.close()
            ui.button('Save', on_click=save_settings).props('color=primary')
    
    dialog.open()

# ── Main App ────────────────────────────────────────────────────────────────────

@ui.page('/')
def main_page():
    """Main page with chat interface."""
    global send_input, chat_container, loading_spinner
    
    # Custom CSS for modern styling
    ui.add_head_html('''
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .chat-container {
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        .message-user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .message-assistant {
            background: #f3f4f6;
            color: #1f2937;
        }
        
        /* KaTeX styling */
        .katex {
            font-size: 1.1em;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #a1a1a1;
        }
    </style>
    ''')
    
    # Include KaTeX for math rendering
    ui.add_head_html('''
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            renderMathInElement(document.body, {
                delimiters: [
                    {left: "$$", right: "$$", display: true},
                    {left: "$", right: "$", display: false}
                ],
                throwOnError: false
            });
        });
    </script>
    ''')
    
    with ui.column().classes('w-full min-h-screen p-4 md:p-8'):
        # Header
        with ui.row().classes('w-full justify-between items-center mb-4'):
            ui.label('StepSolve — Learn Mathematics Step by Step').classes('text-2xl md:text-3xl font-bold text-white')
            with ui.row().classes('gap-2'):
                ui.button(icon='settings', on_click=open_settings).props('flat round color=white').tooltip('Settings')
                ui.button(icon='delete', on_click=clear_chat).props('flat round color=white').tooltip('Clear Chat')
        
        # Chat container
        with ui.card().classes('chat-container flex-1 flex flex-col w-full max-w-4xl mx-auto overflow-hidden'):
            # Chat messages area
            with ui.scroll_area().classes('flex-1 p-4').style('height: 60vh;') as chat_area:
                chat_container = ui.column().classes('w-full').style('min-height: 100%;')
                chat_container.style('display: flex; flex-direction: column;')
                
                # Empty state
                with chat_container:
                    with ui.element('div').classes('flex-1 flex items-center justify-center text-gray-400'):
                        ui.label('Ask a math question or upload an image to get started').classes('text-lg')
            
            # Input area
            with ui.element('div').classes('p-4 border-t border-gray-200'):
                with ui.row().classes('w-full items-end gap-2'):
                    # File upload button
                    with ui.upload(on_upload=handle_file_upload, multiple=False).props('accept=image/*') as uploader:
                        ui.button(icon='image', color='grey').props('round flat').tooltip('Upload image')
                    
                    # Text input
                    send_input = ui.input(placeholder='Type your math question...').classes('flex-1').props('outlined dense')
                    send_input.on('keydown.enter', handle_send)
                    
                    # Send button
                    ui.button(icon='send', on_click=handle_send).props('round unelevated color=primary').tooltip('Send')
        
        # Invisible anchor for scrolling to bottom
        ui.element('div').props('id=chat-end')

# Run the app
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title='StepSolve',
        host='0.0.0.0',
        port=8080,
        reload=False,
        storage_secret='stepsolve-secret-key'
    )
