# Video Transcription and Query Assistant
## Project Description
This project aims to extract transcripts from YouTube videos and enable interactive querying using Large Language Models (LLMs). The system supports models from Hugging Face Hub as well as the phi3 model from Ollama.

# Features
Extracts metadata and transcripts from YouTube videos.
Supports multiple languages: Portuguese (pt, pt-BR) and English (en).
Allows querying the transcript to get summaries or specific information.
Utilizes LLMs from Hugging Face Hub or Ollama's phi3 model.
Project Structure
The project is divided into two main methods:

## Method 1 - Learning: A detailed, educational approach that demonstrates the extraction and interaction process with transcripts.
## Method 2 - Unified Pipeline: A modular approach with reusable functions, simplifying the extraction and query workflow.
Dependencies
This project requires the following libraries:

langchain-community
langchain-huggingface
langchain_ollama
youtube-transcript-api
pytube
To install the dependencies, use:
