import streamlit as st
import google.generativeai as genai
import PyPDF2
import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import streamlit_authenticator as stauth
import time

# ==========================================
# 🔐 1. INITIALIZATION & STATE
# ==========================================
st.set_page_config(page_title="Bio-Step AI", page_icon="🧬", layout="wide")

# App State - Keeps everything in memory
if 'messages' not in st.session_state: st.session_state.messages = []
if 'index' not in st.session_state: st.session_state.index = None
if 'chunks' not in st.session_state: st.session_state.chunks = []
if 'last_sim' not in st.session_state: st.session_state.last_sim = ""
if 'last_quiz' not in st.session_state: st.session_state.last_quiz = ""
if 'stats' not in st.session_state: 
    st.session_state.stats = {"progress": 15, "mastery": 0, "quizzes": 0}

@st.cache_resource
def load_models():
    # Local NLP model for the search engine (Does not need API keys)
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_models()

# ==========================================
# 🔑 2. AUTHENTICATION
# ==========================================
names = ['Ragul P', 'Reena J', 'Pranathi P K', 'Rohith G']
usernames = ['ragul', 'reena', 'pranathi', 'rohith']
passwords = ['rag2007', 'reen2006', 'pran2007', 'rgm2006']

authenticator = stauth.Authenticate(
    {'usernames': {u: {'name': n, 'password': p} for u, n, p in zip(usernames, names, passwords)}},
    'biostep_cookie', 'auth_key', cookie_expiry_days=30
)

if not st.session_state.get("authentication_status"):
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>🧬 Bio-Step AI</h2>", unsafe_allow_html=True)
        authenticator.login(location='main')
    st.stop()

# ==========================================
# 🛠️ 3. FAIL-SAFE API ENGINE
# ==========================================
def call_gemini(prompt, is_vision=False, img=None):
    # YOUR KEYS - Add as many as you want here
    keys = [
        "AIzaSyCHfiE1hNG1s1c72KA72KZe0oPF_eQokwE",
        "AIzaSyBKzsDQVtaVcuOlHYpmccrpMu-AMsCtKDA",
        "AIzaSyD__6irN9Wt3uKcXv6vC8HyId0q4JeQTBw",
        "AIzaSyD__6irN9Wt3uKcXv6vC8HyId0q4JeQTBw",
        "AIzaSyC76Do94Ar6h2_KWf3exuiyH87wTyo6YlM",
        "AIzaSyDebtMwQfFNyBH7tT0s93OqZBV1D64VrXY",
        "AIzaSyDaHu71YNQfQaiRz0LT3GT29OJvztccRDk",
        "AIzaSyAVO73Dp_X-ymGNCdCHS6P1PsCAp6sKmgQ"
    ]
    
    # Using 1.5-Flash-8B: Best free quota for 2026
    model_name = "gemini-1.5-flash-8b"
    
    for k in keys:
        try:
            genai.configure(api_key=k)
            model = genai.GenerativeModel(model_name)
            if is_vision and img:
                return model.generate_content([prompt, img]).text
            return model.generate_content(prompt).text
        except Exception as e:
            # If a key fails (Rate limit), it moves to the next one
            continue
            
    return "⚠️ All API keys are busy. Please wait 60 seconds for the free tier to reset."

# ==========================================
# 🎨 4. UNIFIED SIDEBAR (ONLY ONE)
# ==========================================
with st.sidebar:
    st.markdown(f"### 🧬 Welcome, {st.session_state['name']}")
    theme = st.toggle("☀️ Light Mode", value=False)
    authenticator.logout('Logout', 'sidebar')
    st.markdown("---")
    
    st.subheader("📂 Knowledge Base")
    file = st.file_uploader("Upload Study PDF", type="pdf", label_visibility="collapsed")
    
    if file and st.button("🚀 Initialize Neural Engine", use_container_width=True):
        with st.spinner("Building Knowledge Base..."):
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
            words = text.split()
            chunks = [" ".join(words[i:i+300]) for i in range(0, len(words), 250)]
            
            # FAISS Vector Indexing
            embeddings = embed_model.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
            
            st.session_state.index, st.session_state.chunks = index, chunks
            st.success("Knowledge Base Ready!")

# ==========================================
# 🚀 5. MAIN APP FEATURES
# ==========================================
if st.session_state.index:
    t1, t2, t3, t4, t5, t6 = st.tabs(["📊 Stats", "💬 Tutor", "🧠 Quiz", "📸 Vision", "🧪 Lab Sim", "📚 Research"])
    
    with t1: # Dashboard
        st.subheader("Learning Progress")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mastery", f"{st.session_state.stats['mastery']}%")
        c2.metric("Quizzes", st.session_state.stats['quizzes'])
        c3.metric("Goal", "100%")
        st.progress(st.session_state.stats['progress'] / 100)

    with t2: # Tutor (RAG Mode)
        st.subheader("AI Biotech Tutor")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.write(msg["content"])
            
        if q := st.chat_input("Ask about your PDF..."):
            st.session_state.messages.append({"role": "user", "content": q})
            with st.chat_message("user"): st.write(q)
            
            # Find relevant info in PDF
            query_emb = embed_model.encode([q])
            _, I = st.session_state.index.search(np.array(query_emb).astype('float32'), 2)
            context = "\n".join([st.session_state.chunks[i] for i in I[0]])
            
            ans = call_gemini(f"Based on this info: {context}\n\nUser Question: {q}")
            st.session_state.messages.append({"role": "assistant", "content": ans})
            st.rerun()

    with t3: # Quiz
        st.subheader("Practice Quiz")
        if st.button("Generate MCQ from PDF"):
            st.session_state.last_quiz = call_gemini(f"Create a 3-question MCQ quiz based on: {st.session_state.chunks[:2]}")
        if st.session_state.last_quiz:
            st.markdown(st.session_state.last_quiz)

    with t4: # Vision
        st.subheader("Visual Analysis")
        vfile = st.file_uploader("Upload Lab Photo", type=['jpg','png'])
        if vfile and st.button("Analyze Image"):
            img = Image.open(vfile)
            st.image(img, use_container_width=True)
            st.info(call_gemini("Analyze this biotech image technical findings.", True, img))

    with t5: # Simulation
        st.subheader("Lab Logic Simulator")
        proto = st.text_input("Experiment to Simulate", "PCR Cycle")
        if st.button("Generate Logic"):
            st.session_state.last_sim = call_gemini(f"Generate Python code to simulate: {proto}")
        if st.session_state.last_sim: st.code(st.session_state.last_sim)

    with t6: # Research
        st.subheader("Research Scout")
        topic = st.text_input("Search Breakthroughs", "CRISPR")
        if st.button("Scout"):
            st.write(call_gemini(f"Summarize 3 recent breakthroughs in {topic}"))

else:
    st.info("👈 Please upload your PDF in the sidebar and click 'Initialize' to unlock the platform.")
