import streamlit as st
import google.generativeai as genai
import PyPDF2
import faiss
import numpy as np
import time
from PIL import Image
from sentence_transformers import SentenceTransformer

# ==============================
# 🔐 CONFIG & MEMORY
# ==============================
st.set_page_config(page_title="Bio-Step AI", page_icon="🧬", layout="wide")

# Connect to Secrets
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("Missing GOOGLE_API_KEY in Secrets!")

# Session State for Adaptive Memory
if 'student_stats' not in st.session_state:
    st.session_state.student_stats = {
        "progress": 0, "mastery": 0.0, "quizzes": 0, "weak_topics": []
    }
if 'index' not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []
    st.session_state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ==============================
# 📄 BACKEND UTILITIES
# ==============================
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])

def build_db(text):
    words = text.split()
    chunks = [" ".join(words[i:i+300]) for i in range(0, len(words), 250)]
    embeddings = st.session_state.embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return index, chunks

def retrieve(query, top_k=3):
    query_emb = st.session_state.embed_model.encode([query])
    _, I = st.session_state.index.search(np.array(query_emb).astype('float32'), top_k)
    return [st.session_state.chunks[i] for i in I[0]]

# ==============================
# 🎨 UI & FEATURES
# ==============================
st.title("🧬 Bio-Step AI: Advanced Biotech Learning")

with st.sidebar:
    st.header("📂 Knowledge Base")
    file = st.file_uploader("Upload PDF", type="pdf")
    if file and st.button("Initialize System"):
        with st.spinner("Indexing Biotech Data..."):
            text = extract_text(file)
            st.session_state.index, st.session_state.chunks = build_db(text)
            st.session_state.full_text = text
            st.success("System Ready!")

if st.session_state.index:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "💬 Multi-Agent Chat", "🧠 Quiz", "📸 Vision Lab"])

    # FEATURE 1: ADAPTIVE DASHBOARD
    with tab1:
        st.subheader("Your Progress")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mastery Score", f"{st.session_state.student_stats['mastery']}%")
        col2.metric("Quizzes Done", st.session_state.student_stats['quizzes'])
        
        # Progress Bar
        st.write(f"Learning Path Progress: {st.session_state.student_stats['progress']}%")
        st.progress(st.session_state.student_stats['progress'] / 100)
        
        if st.session_state.student_stats['weak_topics']:
            st.warning(f"⚠️ Focus Needed: {', '.join(set(st.session_state.student_stats['weak_topics']))}")

    # FEATURE 2: MULTI-AGENT VERIFIED CHAT
    with tab2:
        st.subheader("Verified Biotech Tutor")
        if q := st.chat_input("Ask about the notes..."):
            st.chat_message("user").write(q)
            context = "\n".join(retrieve(q))
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            with st.spinner("Tutor drafting... Critic verifying..."):
                # Multi-Agent Loop
                draft = model.generate_content(f"Explain this biotech concept: {q}\nContext: {context}").text
                verified = model.generate_content(f"As a Scientific Critic, fix any errors in this draft using ONLY the context: {draft}\nContext: {context}").text
                st.chat_message("assistant").write(verified)

    # FEATURE 3: QUIZ MODULE
    with tab3:
        if st.button("Generate New Quiz"):
            model = genai.GenerativeModel("gemini-2.5-flash")
            quiz = model.generate_content(f"Create a 5-question MCQ from: {st.session_state.chunks[:5]}").text
            st.write(quiz)
            # Simulated Score Update
            score = st.slider("Select your score (for demo)", 0, 5, 4)
            if st.button("Update Stats"):
                st.session_state.student_stats['quizzes'] += 1
                st.session_state.student_stats['mastery'] = (score/5)*100
                st.session_state.student_stats['progress'] += 10
                st.rerun()

    # FEATURE 4: VISION LAB ANALYSIS
    with tab4:
        st.subheader("Lab Vision Agent")
        img_file = st.file_uploader("Upload Lab Image (Gel, Chart, etc.)", type=['jpg','png','jpeg'])
        if img_file:
            img = Image.open(img_file)
            st.image(img, caption="Uploaded Lab Data", use_container_width=True)
            if st.button("Analyze Image"):
                model = genai.GenerativeModel("gemini-2.5-flash")
                with st.spinner("Analyzing visual data..."):
                    res = model.generate_content(["Identify this biotech lab result and explain the observations.", img])
                    st.info(res.text)
else:
    st.info("Upload a PDF to unlock features.")
