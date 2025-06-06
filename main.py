import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
import tempfile
from PyPDF2 import PdfReader

# --- Config ---
st.set_page_config(page_title="📚 PDF Chat", page_icon="📄", layout="wide")
load_dotenv()

# --- Environment ---
openai_api_key = os.getenv("openai_api_key")
qdrant_api_key = os.getenv("qdrant_api_key")
qdrant_url = "https://40103eb0-922e-457b-bbcb-213e10aea2b2.eu-central-1-0.aws.cloud.qdrant.io:6333"
collection_name = "chat_pdf_collection"

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "qdrant_store" not in st.session_state:
    st.session_state.qdrant_store = None
if "chat_ready" not in st.session_state:
    st.session_state.chat_ready = False

# --- Sidebar ---
with st.sidebar:
    st.title("📄 Chat with your PDF")
    st.markdown("1. Upload a PDF\n2. Ask anything like a conversation")
    st.info("Built with LangChain, OpenAI, Qdrant")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file:
        with st.spinner("Processing PDF..."):
            # Save file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                path = tmp.name

            # PDF Preview
            pdf_reader = PdfReader(path)
            st.success(f"{len(pdf_reader.pages)} pages loaded!")

            # Load and split
            loader = PyPDFLoader(path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
            chunks = splitter.split_documents(docs)

            # Embed and index
            embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
            qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            qdrant_client.delete_collection(collection_name=collection_name)

            qdrant_store = Qdrant.from_documents(
                documents=chunks,
                embedding=embedding,
                url=qdrant_url,
                api_key=qdrant_api_key,
                collection_name=collection_name
            )

            st.session_state.qdrant_store = qdrant_store
            st.session_state.chat_ready = True
            st.success("✅ Ready to chat!")

# --- Chat Interface ---
st.title("💬 PDF Chat Interface")

if not st.session_state.chat_ready:
    st.info("Please upload a PDF from the sidebar to start chatting.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask something about the PDF...")
    if user_query:
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.spinner("Searching..."):
            results = st.session_state.qdrant_store.similarity_search(user_query, k=3)

        if results:
            context = "\n\n".join([doc.page_content for doc in results])
            pages = ", ".join([str(doc.metadata.get("page_number", "N/A")) for doc in results])

            # LLM Answering
            openai_client = OpenAI(api_key=openai_api_key)
            prompt = f"""
You are a helpful assistant. Use the following context from a PDF to answer the user's question.

Context:
\"\"\"
{context}
\"\"\"

Question:
{user_query}

Answer in a clear, friendly, and concise manner (under 150 words). If the answer is not in the context, say "I couldn’t find the answer in the PDF."
"""

            try:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a concise, friendly PDF assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                )
                answer = response.choices[0].message.content.strip()
                reply = f"{answer}\n\n📄 *Based on pages: {pages}*"
            except Exception as e:
                reply = f"⚠️ Error with LLM summarization: {e}\n\nFallback:\n{results[0].page_content}\n📄 Page: {pages}"
        else:
            reply = "❌ I couldn’t find a relevant answer."

        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

# --- Footer ---
st.markdown("<hr><center>Made by Arpit Mohankar ❤️</center>", unsafe_allow_html=True)
