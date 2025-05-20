# LLM Factory Examples

This repository contains simple examples showing how to use the OpenAI API to communicate with LLM Factory Endpoints for various AI tasks.

## Setup Instructions

1. Retrieve your API key from LLM Factory: https://llm-factory.ai.uky.edu/api-keys
2. Copy `config.ini.example` to `config.ini` and add your API key to the `openai_api_key` field
3. Choose the model ID you would like to use

## Example Scripts

* `chat_completions_test.py` - Chat with a language model
* `embeddings_test.py` - Create vector embeddings from text
* `transcriptions_test.py` - Generate transcriptions from audio files
* `vision_test.py` - Process images with multimodal models
* `tools_test.py` - Use tool-augmented language models
* `simple_rag.py` - Implement a basic RAG (Retrieval-Augmented Generation) system

Run each script after configuring to test different capabilities of the LLM Factory API.