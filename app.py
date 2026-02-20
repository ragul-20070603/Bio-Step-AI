import streamlit as st
import google.generativeai as genai
import PyPDF2
import faiss
import numpy as np
import tempfile
from sentence_transformers import SentenceTransformer

# ==============================
# 🔐 APP CONFIG & API
# ==============================
st.set_page_config(page_title="Bio-Step AI", page_icon="🧬", layout="wide")

# Use Streamlit secrets for the API Key
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("Please set the GOOGLE_API_KEY in Streamlit Secrets.")

# Initialize Session State (Our "Long-term Memory")
if 'student_stats' not in st.session_state:
    st.session_state.student_stats = {
        "quizzes_taken": 0, "average_score": 0.0, "weak_topics": [], "completed_modules": 0
    }
if 'index' not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []
    st.session_state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ==============================
# 📄 BACKEND LOGIC (Ported from Colab)
# ==============================
def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text: text += page_text + "\n"
    return text

def build_db(text):
    words = text.split()
    chunks = [" ".join(words[i:i+300]) for i in range(0, len(words), 250)]
    embeddings = st.session_state.embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return index, chunks

def retrieve(query, top_k=3):
    query_embedding = st.session_state.embed_model.encode([query])
    D, I = st.session_state.index.search(np.array(query_embedding).astype('float32'), top_k)
    return [st.session_state.chunks[i] for i in I[0]]

# ==============================
# 🎨 FRONTEND UI
# ==============================
st.title("🧬 Bio-Step AI: Personalized Biotech Learning")

with st.sidebar:
    st.header("📂 Knowledge Ingestion")
    uploaded_file = st.file_uploader("Upload Biotech PDF", type="pdf")
    if uploaded_file and st.button("Analyze & Index"):
        with st.spinner("Processing Knowledge Base..."):
            raw_text = extract_text_from_pdf(uploaded_file)
            idx, chks = build_db(raw_text)
            st.session_state.index = idx
            st.session_state.chunks = chks
            st.session_state.full_text = raw_text
            st.success("Knowledge Base Ready!")

if st.session_state.index:
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💬 Multi-Agent Chat", "🧠 Quiz"])

    with tab1:
        st.subheader("Learning Analytics")
        col1, col2 = st.columns(2)
        col1.metric("Progress", f"{st.session_state.student_stats['completed_modules']*10}%")
        col2.metric("Mastery", f"{st.session_state.student_stats['average_score']:.1f}%")
        
        if st.session_state.student_stats['weak_topics']:
            st.warning(f"Review Recommended: {', '.join(set(st.session_state.student_stats['weak_topics']))}")

    with tab2:
        st.subheader("Multi-Agent Verified Tutor")
        user_query = st.chat_input("Ask a question about the document...")
        if user_query:
            context = "\n\n".join(retrieve(user_query))
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            # Agent Loop
            with st.spinner("Tutor drafting... Critic verifying..."):
                tutor_reply = model.generate_content(f"Explain this: {user_query}\nContext: {context}").text
                verified = model.generate_content(f"Verify and fix errors in this biotech explanation based on the context:\n{tutor_reply}\nContext: {context}").text
                st.chat_message("assistant").write(verified)

    with tab3:
        st.subheader("Adaptive Quiz")
        if st.button("Generate Quiz"):
            model = genai.GenerativeModel("gemini-2.5-flash")
            context = "\n\n".join(retrieve("key definitions and facts", top_k=5))
            quiz = model.generate_content(f"Create a 5-question MCQ quiz based on:\n{context}").text
            st.write(quiz)
            
            # Mock update for demo
            score = st.slider("Select your score (for analytics demo)", 0, 5, 3)
            if st.button("Submit Score"):
                st.session_state.student_stats['quizzes_taken'] += 1
                st.session_state.student_stats['average_score'] = (score/5)*100
                st.success("Dashboard Updated!")
else:

    st.info("Please upload a document to unlock the learning modules.")
