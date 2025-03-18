import os
import streamlit as st
import speech_recognition as sr
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from google.generativeai import configure, GenerativeModel

# ✅ SETUP: Replace with your Gemini API Key
GEMINI_API_KEY = "AIzaSyDyOQa8cnZcO9227h8W26tgMxRHv6Ma7xM"
configure(api_key=GEMINI_API_KEY)

# ✅ Load PDFs Privately from Folder
PDF_FOLDER = "DataSets/"
if not os.path.exists(PDF_FOLDER):
    os.makedirs(PDF_FOLDER)

# ✅ Load PDFs & Create Vector Store
def load_and_index_pdfs():
    pdf_files = [os.path.join(PDF_FOLDER, f) for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    if not pdf_files:
        return None
    
    documents = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# ✅ Conversational AI with Gemini (Only Answers if PDF Has Relevant Data)
def chat_with_gemini(prompt, context, history):
    if not context.strip():
        return "I'm still learning. I don't have enough information on that topic yet."

    model = GenerativeModel("gemini-2.0-pro-exp-02-05")  # ✅ UPDATED MODEL
    conversation = "\n".join(history) + "\nUser: " + prompt
    response = model.generate_content(conversation)
    return response.text

# ✅ Streamlit UI Setup
st.set_page_config(page_title="DGCT Guide for AI&DS", page_icon="📘")
st.image("Assests/logo.png", width=150)  # Ensure 'logo.png' exists
st.title("📘 DGCT Guide for AI&DS")
st.subheader("Conversational AI Chatbot")

# ✅ Load Vector Database
vector_db = load_and_index_pdfs()
retriever = vector_db.as_retriever() if vector_db else None

# ✅ Initialize Chat History State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Chat History Toggle (Open/Close)
with st.sidebar:
    show_history = st.checkbox("📜 Show Chat History", value=False)

if show_history:
    st.sidebar.subheader("Previous Conversations")
    for i in range(0, len(st.session_state.chat_history), 2):
        st.sidebar.markdown(f"🧑‍💬 **You:** {st.session_state.chat_history[i]}")
        if i + 1 < len(st.session_state.chat_history):
            st.sidebar.markdown(f"🤖 **AI:** {st.session_state.chat_history[i + 1]}")
    st.sidebar.markdown("---")
    if st.sidebar.button("❌ Clear Chat History"):
        st.session_state.chat_history = []
        st.sidebar.success("Chat history cleared!")

# ✅ Display Chat Messages
for i in range(0, len(st.session_state.chat_history), 2):
    with st.chat_message("user"):
        st.markdown(st.session_state.chat_history[i])  # User Query
    if i + 1 < len(st.session_state.chat_history):
        with st.chat_message("assistant"):
            st.markdown(st.session_state.chat_history[i + 1])  # AI Response

# ✅ Chat Input
query = st.chat_input("Ask a question...")
if query:
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.spinner("Thinking... 💡"):
        context = ""
        if retriever:
            docs = retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in docs])

        final_prompt = f"{context}\n\nUser: {query}"
        response = chat_with_gemini(final_prompt, context, st.session_state.chat_history)

        # ✅ Store Chat History
        st.session_state.chat_history.append(f"{query}")  # User message
        st.session_state.chat_history.append(f"{response}")  # AI response

        with st.chat_message("assistant"):
            st.markdown(response)
