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

pip install langchain-community langchain-huggingface langchain_ollama
pip install youtube-transcript-api
pip install pytube --upgrade

#Configuration
##Hugging Face Hub
To use Hugging Face Hub models, configure the access token:

import os
import getpass

os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass()

##Ollama phi3
To use the phi3 model from Ollama:

##Download Ollama from: https://ollama.com/download
In the terminal, run:
ollama pull phi3

#How to Use
##Method 1 - Learning
This method is educational and explores the following steps:

Extracts video information, including metadata and transcript.
Saves the transcript to a .txt file.
Sets up models using Hugging Face Hub or Ollama.
Creates custom prompts and tests queries.
##Method 2 - Unified Pipeline
A modular method with reusable functions for:

Loading videos and extracting information more efficiently.
Setting up the LLM and prompt chain.
Querying the transcript.

#Usage Example:
url_video = "https://www.youtube.com/watch?v=PeMlggyqz0Y&t=6s"
query_user = "summarize in an easy-to-understand way"
model_class = "ollama" # Options: ["hf_hub", "ollama"]
language = ["pt", "pt-BR", "en"]

interpret_video(url_video, query_user, model_class, language)


