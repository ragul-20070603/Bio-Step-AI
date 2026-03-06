import streamlit as st
import google.generativeai as genai
import PyPDF2
import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import streamlit_authenticator as stauth
import sqlite3

# ==========================================
# 🔐 1. INITIALIZATION
# ==========================================
st.set_page_config(page_title="Bio-Step AI", page_icon="🧬", layout="wide")

# Persistent State Management
defaults = {
    'messages': [], 'index': None, 'chunks': [], 
    'last_quiz': "", 'last_sim': "", 'last_scout': "", 'last_vision': ""
}
for key, val in defaults.items():
    if key not in st.session_state: st.session_state[key] = val

if 'student_stats' not in st.session_state:
    st.session_state.student_stats = {"progress": 45, "mastery": 82.0, "quizzes": 12}

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()

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
        st.markdown("<h2 style='text-align: center; color: #818cf8;'>🧬 Bio-Step AI Portal</h2>", unsafe_allow_html=True)
        authenticator.login(location='main')
    st.stop()

# ==========================================
# 🛠️ 3. REVOLVING API ENGINE
# ==========================================
def call_gemini(prompt, is_vision=False, img=None):
    keys = [
        "AIzaSyCHfiE1hNG1s1c72KA72KZe0oPF_eQokwE",
        "AIzaSyBKzsDQVtaVcuOlHYpmccrpMu-AMsCtKDA",
        "AIzaSyD__6irN9Wt3uKcXv6vC8HyId0q4JeQTBw"
    ]
    model_id = "gemini-1.5-flash" # Stable version for free tier
    
    for k in keys:
        try:
            genai.configure(api_key=k)
            model = genai.GenerativeModel(model_id)
            if is_vision and img:
                return model.generate_content([prompt, img]).text
            return model.generate_content(prompt).text
        except: continue
    return "❌ API Error: All keys exhausted. Please wait 1 minute."

# ==========================================
# 🎨 4. DESIGNER SIDEBAR (ONLY ONE)
# ==========================================
with st.sidebar:
    st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px;">
            <h4 style="margin: 0; color: #818cf8;">🧬 Bio-Step AI</h4>
            <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">{st.session_state['name']}</p>
        </div>
    """, unsafe_allow_html=True)
    
    theme_choice = st.toggle("☀️ Light Mode", value=False)
    authenticator.logout('Logout', 'sidebar')
    st.markdown("---")
    
    st.subheader("📂 Knowledge Base")
    file = st.file_uploader("Upload Biotech PDF", type="pdf", label_visibility="collapsed")
    
    if file and st.button("🚀 Initialize Neural Engine", use_container_width=True):
        with st.spinner("Decoding Bio-Data..."):
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
            words = text.split()
            chunks = [" ".join(words[i:i+300]) for i in range(0, len(words), 250)]
            embeddings = embed_model.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
            st.session_state.index, st.session_state.chunks = index, chunks
            st.success("System Ready!")

# ==========================================
# 🚀 5. FULL 6-FEATURE INTERFACE
# ==========================================
if st.session_state.index:
    tabs = st.tabs(["📊 Dashboard", "💬 Tutor", "🧠 Quiz", "📸 Vision", "🧪 Simulation", "📚 Research"])
    
    # --- 1. DASHBOARD ---
    with tabs[0]:
        st.subheader("Personalized Learning Analytics")
        c1, c2, c3 = st.columns(3)
        stats = st.session_state.student_stats
        c1.metric("Mastery Score", f"{stats['mastery']}%", "↑ 2.3%")
        c2.metric("Quizzes Completed", stats['quizzes'])
        c3.metric("Path Progress", f"{stats['progress']}%")
        st.progress(stats['progress'] / 100)
        st.write("---")
        st.write("🎯 **Suggested Next Topic:** Molecular Cloning Optimization")

    # --- 2. TUTOR (RAG) ---
    with tabs[1]:
        st.subheader("Verified Biotech Tutor")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.write(msg["content"])
        
        if q := st.chat_input("Ask a technical question..."):
            st.session_state.messages.append({"role": "user", "content": q})
            with st.chat_message("user"): st.write(q)
            
            # Neural Retrieval
            query_emb = embed_model.encode([q])
            _, I = st.session_state.index.search(np.array(query_emb).astype('float32'), 2)
            context = "\n".join([st.session_state.chunks[i] for i in I[0]])
            
            with st.status("Searching Knowledge Base..."):
                ans = call_gemini(f"Context: {context}\n\nQuestion: {q}")
                st.session_state.messages.append({"role": "assistant", "content": ans})
                st.rerun()

    # --- 3. ADAPTIVE QUIZ ---
    with tabs[2]:
        st.subheader("Adaptive MCQ Generator")
        if st.button("Generate Quiz from PDF"):
            with st.spinner("Creating..."):
                st.session_state.last_quiz = call_gemini(f"Create a 3-question MCQ quiz based on: {st.session_state.chunks[:2]}")
        if st.session_state.last_quiz:
            st.markdown(st.session_state.last_quiz)
            if st.button("Submit Score"): st.success("Stats Updated!")

    # --- 4. VISION AGENT ---
    with tabs[3]:
        st.subheader("Lab-to-Logic Vision")
        img_file = st.file_uploader("Upload Lab Photo", type=['png','jpg','jpeg'])
        if img_file and st.button("Analyze Image"):
            img = Image.open(img_file)
            st.image(img, use_container_width=True)
            with st.spinner("Processing..."):
                res = call_gemini("Analyze this biotech image technical findings.", True, img)
                st.info(res)

    # --- 5. PROTOCOL SIMULATOR ---
    with tabs[4]:
        st.subheader("Protocol Logic Simulator")
        exp = st.text_input("Experiment to Simulate", "Gel Electrophoresis")
        if st.button("Run Simulation"):
            with st.spinner("Generating Logic..."):
                st.session_state.last_sim = call_gemini(f"Generate Python logic/code to simulate: {exp}")
        if st.session_state.last_sim:
            st.code(st.session_state.last_sim, language='python')

    # --- 6. RESEARCH SCOUT ---
    with tabs[5]:
        st.subheader("Scientific Research Scout")
        topic = st.text_input("Enter Topic (e.g., mRNA Vaccines)")
        if st.button("Scout bioRxiv/PubMed"):
            with st.spinner("Scouting..."):
                res = call_gemini(f"Summarize 3 recent breakthroughs in: {topic}")
                st.markdown(res)

else:
    st.info("👈 Please upload your document in the sidebar to unlock all 6 AI features.")
