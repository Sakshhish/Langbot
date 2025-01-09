# First, let's outline what we need:

* * A dataset to serve as our knowledge base
  * A way to index and retrieve information
  * A language model to generate responses
  * A chat interface to interact with users

# RAG Chatbot

A Retrieval-Augmented Generation chatbot built with LangChain and HuggingFace.

## Prerequisites

- Python 3.8+
- HuggingFace account and API token
- CSV dataset

## Installation

1. Clone the repository:

```bash
git clone <https://github.com/Sakshhish/Langbot.git>
cd <Langbot>
```

2. Create and activate virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

3. Install required packages:

```bash
pip install langchain langchain_community langchain-huggingface faiss-cpu sentence-transformers huggingface_hub rich
```

## Setting Up HuggingFace Token

1. Visit https://huggingface.co/
2. Create an account or login
3. Go to Settings → Access Tokens
4. Create new token with following permissions:
   - Read repository contents
   - Read models and datasets
   - Read organization info
   - Inference API
5. Copy the generated token

## Preparing Your Data

Place your CSV file in the project directory. The CSV should have these columns:

- Language
- Year
- Paradigm
- Typing

Example CSV format:

```csv
Language,Year,Paradigm,Typing
Python,1991,Multi-paradigm,Dynamically typed
Java,1995,Object-oriented,Statically typed
```

## Running the Application

1. Run the application:

```bash
python index.py
```

2. When prompted, enter your HuggingFace token
3. Start chatting! Available commands:

   - Type questions normally for answers
   - `history`: View chat history
   - `clear`: Clear chat history
   - `quit`: Exit application

## Example Questions

- "When was Python created?"
- "Which languages are multi-paradigm?"
- "What is the typing system of Java?"
- "Which languages were created in 1995?"

## Troubleshooting

1. Token Error:

   - Ensure token is copied correctly
   - Check token permissions on HuggingFace
2. Data Loading Error:

   - Verify CSV file format
   - Check file path
3. Model Error:

   - Check internet connection
   - Verify token permissions
