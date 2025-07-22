# AI Voice Assistant

## Overview
AI Voice Assistant is a web-based, AI-powered application built with Streamlit. The application provides the following features:
- **💬 Text Chat:** Communicate with an AI language model.
- **🎤 Voice Chat:** Use voice commands to interact with the AI.
- **📸 OCR Text Extraction:** Extract text from images with enhanced preprocessing.
- **Real-time Speech Recognition:** Convert spoken queries into text.

## Features
- **Interactive Chat:** Both text and voice interfaces let you ask questions and receive responses.
- **OCR Capabilities:** Process images to extract text using Tesseract OCR with additional noise reduction and enhancement.
- **Configurable Settings:** Options to toggle text-to-speech and view usage tips.
- **User-Friendly Format:** A clean, intuitive UI provided by Streamlit.

## Installation and Setup

1. **Clone the Repository:**

   ```sh
   git clone <repository-url>
   cd AI_assistant
   ```

2. **Create a Virtual Environment and Install Dependencies:**

   For Windows:
   ```sh
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

   For macOS/Linux:
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**

   Create a `.env` file in the project root (do NOT commit your actual API key):

   ```env
   GROQ_API_KEY=<your_api_key_here>
   ```

4. **Install Tesseract OCR:**

   - Download and install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).
   - Verify or update the `pytesseract.pytesseract.tesseract_cmd` path in the code (see [main.py](c:\Users\MR\OneDrive\Desktop\AI_assistant\main.py)) to match your Tesseract installation.

## Usage

Start the application by running:

```sh
streamlit run main.py
```

Then, choose an action from the interface:
- **💬 Text Chat:** Type your query into the text input and press "Send 📤" to receive a response.
- **🎤 Voice Chat:** Click "🎤 Start Listening" to record your question. The application processes your speech and presents the response.
- **📸 Extract Text from Image:** Upload an image. Use the "🔍 Extract Text" button to perform OCR. You can also ask follow-up questions about the extracted text.

## Files Included for Deployment
- **main.py:** The main application source code containing the implementation of chat, voice recognition, and OCR functionalities.
- **requirements.txt:** Lists all the dependencies required.
- **Readme.md:** Documentation for the project.
- **.gitignore:** Configured to exclude environment files and temporary files.
- **(Optional) .env.example:** Consider adding this file as a guideline for required environment variables (do not include sensitive data).

## Files Excluded from Deployment
- **.env:** Actual environment file with sensitive data is excluded. Ensure it is listed in the [`.gitignore`](c:\Users\MR\OneDrive\Desktop\AI_assistant\.gitignore).
- **Temporary/Debug Files:** Any caching, log, or Postman instruction files (configured within .gitignore) are not deployed.

## Dependencies
Key libraries and tools for this project:

- [Streamlit](https://streamlit.io/)
- [Pytesseract](https://pypi.org/project/pytesseract/)
- [OpenCV (opencv-python)](https://pypi.org/project/opencv-python/)
- [NumPy](https://numpy.org/)
- [Pillow](https://pillow.readthedocs.io/)
- [Groq](#) (Language model API client)
- [pyttsx3](https://pypi.org/project/pyttsx3/)
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

## License
This project is free to use.

## Additional Information
- **TTS and Voice Recognition:** The application leverages text-to-speech and voice recognition modules. In case of any loading or processing issues, make sure that microphone and audio configurations are correctly set up.
- **Support:** For more details or troubleshooting, please refer to the code comments within [main.py](c:\Users\MR\OneDrive\Desktop\AI_assistant\main.py).

Enjoy using the AI Voice Assistant!