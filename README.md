# StepSolve - Learn Mathematics Step by Step

StepSolve is an AI-powered mathematics tutor built with Gradio, designed to help students understand math concepts through interactive conversations and image analysis. The app uses advanced AI models to extract text from handwritten or printed math problems, analyze them, and provide structured responses highlighting key concepts.

## Features

- **Clean Chat Interface**: Modern, ChatGPT/Grok-style UI with chat history and responsive design
- **Multimodal Input**: Type text questions or upload images of math problems
- **Smart Image Processing**: Automatic OCR (Optical Character Recognition) to extract math notation from images
- **Interactive Cropping**: Built-in image editor for precise cropping before analysis
- **AI-Powered Analysis**: Two-model system for accurate math problem understanding:
  - **OCR Model**: Groq's vision model for text extraction
  - **Reasoning Model**: NVIDIA NIM models (Mistral-Nemotron primary, Qwen fallback) for concept analysis
- **LaTeX Math Rendering**: Proper display of mathematical formulas and equations
- **Concept Identification**: Automatically identifies and explains key mathematical concepts needed
- **Fallback System**: Robust error handling with model fallbacks for reliability

## How It Works

1. **Input**: Users can type math questions or upload images/photos of problems
2. **Processing**:
   - If image: Automatic cropping interface appears, then OCR extracts text
   - Text is combined with any typed input
3. **Analysis**: Reasoning model analyzes the problem and responds with:
   - Formatted question in LaTeX
   - Multiple choice options (if applicable)
   - Key concepts to understand
   - Targeted question about the concepts
4. **Display**: Clean, readable response with proper math rendering

## Installation

### Prerequisites

- Python 3.8+
- API keys from NVIDIA and Groq

### Setup

1. **Clone or download** the repository

2. **Install dependencies**:

   ```bash
   pip install gradio openai groq pillow python-dotenv
   ```

3. **Set up API keys**:
   - Get NVIDIA API key: [build.nvidia.com](https://build.nvidia.com)
   - Get Groq API key: [groq.com](https://groq.com)
   - Create a `.env` file in the project root:
     ```
     NVIDIA_API_KEY=your_nvidia_api_key_here
     GROQ_API_KEY=your_groq_api_key_here
     ```

## Usage

1. **Run the app**:

   ```bash
   python app.py
   ```

2. **Open in browser**: The app will launch at `http://localhost:7860`

3. **Ask questions**:
   - Type your math question in the text box
   - Or upload an image of a math problem
   - If uploading an image, use the cropping tool to select the relevant area
   - Click "Send" to get AI analysis

## API Architecture

### OCR Pipeline

- **Model**: Groq Llama-4-Scout-17B-16E-Instruct
- **Purpose**: Extract clean text from images
- **Processing**: Image compression, base64 encoding, vision API call

### Reasoning Pipeline

- **Primary Model**: Mistral-Nemotron (via NVIDIA NIM)
- **Fallback Model**: Qwen3-Coder-480B-A35B-Instruct
- **Purpose**: Analyze math problems and identify concepts
- **Output Format**: Strict JSON with formatted question, options, concepts, and response question

## Supported Math Topics

The app can handle various mathematics topics including:

- Algebra
- Calculus (derivatives, integrals)
- Geometry
- Trigonometry
- Statistics
- And more advanced topics

## Error Handling

- Automatic model fallback if primary model fails
- Graceful degradation for malformed responses
- Clear error messages for missing API keys or network issues

## Customization

The app's prompts and model configurations can be modified in `app.py`:

- `_OCR_PROMPT`: Customize OCR extraction behavior
- `_REASONING_SYSTEM_PROMPT`: Adjust the tutor's response style
- Model priorities in `generate_tutor_response()`

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve StepSolve.

## License

This project is open-source. Please check the license file for details.

## Acknowledgments

- Built with [Gradio](https://gradio.app) for the web interface
- Powered by [NVIDIA NIM](https://build.nvidia.com) and [Groq](https://groq.com) APIs
- Math rendering using LaTeX and Markdown
