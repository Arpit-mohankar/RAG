# PDF Chat with RAG (Retrieval-Augmented Generation)

This project is a **Streamlit web app** that lets you upload a PDF and chat with it using advanced AI. It leverages **LangChain**, **OpenAI**, and **Qdrant** to provide context-aware answers from your PDF documents.

## Features

- Upload any PDF and instantly chat with its contents.
- Uses OpenAI's GPT models for concise, friendly answers.
- Fast, semantic search powered by Qdrant vector database.
- Built with modern Python libraries and Streamlit for a smooth UI.

---

## Folder Structure

```
.
├── main.py              # Streamlit app source code
├── requirements.txt     # All Python dependencies
├── pyproject.toml       # Project metadata and dependencies
├── .python-version      # Python version (3.11)
├── .venv/               # (Optional) Local virtual environment
└── README.md            # Project documentation
```

> **Note:** The `.venv/` folder is for your local Python environment and can be ignored.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repo-url>
cd rag1
```

### 2. Create a Virtual Environment (Recommended)

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
or, if you prefer using `pyproject.toml`:
```bash
pip install .
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root with your API keys:
```
openai_api_key=YOUR_OPENAI_API_KEY
qdrant_api_key=YOUR_QDRANT_API_KEY
```

### 5. Run the App

```bash
streamlit run main.py
```

Open the provided local URL in your browser.

---

## How it Works

1. **Upload a PDF**: The app splits your PDF into chunks and creates vector embeddings.
2. **Ask Questions**: Your queries are matched to the most relevant PDF sections using Qdrant.
3. **Get Answers**: OpenAI's GPT model answers your question using only the PDF context.

---

## Credits

Made by Arpit Mohankar ❤️  
Built with [Streamlit](https://streamlit.io/), [LangChain](https://www.langchain.com/), [OpenAI](https://openai.com/), and [Qdrant](https://qdrant.tech/).
