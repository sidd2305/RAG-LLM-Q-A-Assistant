import os
import pickle
import time
from langchain.llms import HuggingFaceEndpoint
import pandas as pd
from langchain_community.embeddings import FakeEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from pypdf import PdfReader
import streamlit as st
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Dynamic token loading
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "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")

# Streamlit App Title
st.title("LLM ESA RAG Tool 📈")

# Function to initialize Hugging Face LLM
def initialize_llm():
    return HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )

# Sidebar
st.sidebar.title("Options")
app_mode = st.sidebar.radio("Choose an option", ["Upload File", "Chat with LLM"])

# Initialize variables for chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# File Upload and Processing
if app_mode == "Upload File":
    st.write("Upload a file to use as a reference for the LLM Q&A System")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt", "pdf", "json"])
    file_path = "faiss_store_openai.pkl"
    llm = initialize_llm()
    main_placeholder = st.empty()
    data = []

    if uploaded_file:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            data = [Document(page_content=row.to_string(), metadata={"source": "uploaded_csv"}) for _, row in df.iterrows()]
        elif uploaded_file.type == "text/plain":
            data = [Document(page_content=uploaded_file.read().decode("utf-8"), metadata={"source": "uploaded_txt"})]
        elif uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    data.append(Document(page_content=text, metadata={"source": "uploaded_pdf"}))
        elif uploaded_file.type == "application/json":
            content = pd.read_json(uploaded_file)
            data = [Document(page_content=row.to_json(), metadata={"source": "uploaded_json"}) for _, row in content.iterrows()]

        if data:
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Text Splitter...Started...✅✅✅")
            docs = text_splitter.split_documents(data)

            embeddings = FakeEmbeddings(size=500)
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            main_placeholder.text("Embedding Vector Started Building...✅✅✅")
            time.sleep(2)

            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)

# Chat Functionality
elif app_mode == "Chat with LLM":
    st.write("Chat with the LLM! Your session's context will be maintained.")
    llm = initialize_llm()
    file_path = "faiss_store_openai.pkl"
    
    # Display chat history
    for message in st.session_state["chat_history"]:
        if message["type"] == "user":
            st.write(f"**You:** {message['content']}")
        else:
            st.write(f"**LLM:** {message['content']}")
    
    # Input for user message
    query = st.text_input("Ask me a question or request a recommendation:")
    
    if query:
        # Save user query to chat history
        st.session_state["chat_history"].append({"type": "user", "content": query})
        
        # Attempt to generate an answer
        answer = ""
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)
                answer = result.get("answer", "I couldn't find an answer.")
        else:
            # Default response if no files have been uploaded
            answer = llm(query)["text"]
        
        # Display answer and save to chat history
        st.write(f"**LLM:** {answer}")
        st.session_state["chat_history"].append({"type": "llm", "content": answer})
