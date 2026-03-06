import streamlit as st
import google.generativeai as genai
import PyPDF2
import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import sqlite3
import streamlit_authenticator as stauth

# ==========================================
# 🔐 1. INITIALIZATION
# ==========================================
st.set_page_config(page_title="Bio-Step AI", page_icon="🧬", layout="wide")

# Persistent State Management
defaults = {
    'last_quiz': "", 'last_sim': "", 'last_scout': "", 
    'last_vision': "", 'messages': [], 'index': None, 'chunks': []
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

if 'student_stats' not in st.session_state:
    st.session_state.student_stats = {"progress": 0, "mastery": 0.0, "quizzes": 0, "weak_topics": []}

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

st.session_state.embed_model = load_embed_model()

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
    col1, col2, col3 = st.columns([1, 1.2, 1]) 
    with col2:
        st.write("#")
        st.markdown("<h2 style='text-align: center; color: #818cf8;'>🧬 Bio-Step AI Portal</h2>", unsafe_allow_html=True)
        authenticator.login(location='main')
    if not st.session_state.get("authentication_status"):
        st.stop() 

name = st.session_state["name"]
username = st.session_state["username"]

# ==========================================
# 🎨 3. THE ONE AND ONLY SIDEBAR (DESIGNER VERSION)
# ==========================================
with st.sidebar:
    # Profile Card
    st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px;">
            <h4 style="margin: 0; color: #818cf8;">🧬 Bio-Step AI</h4>
            <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Student Portal</p>
            <hr style="margin: 15px 0; opacity: 0.2;">
            <p style="margin: 0; font-weight: 600;">Welcome, {name}!</p>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("⚙️ Preferences")
    theme_choice = st.toggle("☀️ Light Mode", value=False)
    authenticator.logout('Logout', 'sidebar')
    
    st.markdown("---")

    # Knowledge Base Uploader (The only uploader in the whole code)
    st.subheader("📂 Knowledge Base")
    file = st.file_uploader("Upload Notes (PDF)", type="pdf", label_visibility="collapsed")
    
    if file:
        st.markdown(f"**Ready to Index:** `{file.name}`")
        if st.button("🚀 Initialize Neural Engine", use_container_width=True):
            with st.spinner("Decoding Bio-Data..."):
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = "\n".join([p.extract_text() for p in pdf_reader.pages if p.extract_text()])
                
                # Semantic Chunking & Indexing
                words = full_text.split()
                chunks = [" ".join(words[i:i+300]) for i in range(0, len(words), 250)]
                embeddings = st.session_state.embed_model.encode(chunks)
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(np.array(embeddings).astype('float32'))
                
                st.session_state.index = index
                st.session_state.chunks = chunks
                st.success("System Ready!")
                st.balloons()

    st.markdown("---")
    st.markdown(f"""<div style="font-size: 0.8rem; opacity: 0.6; text-align: center;">V 3.0.4-Flash | Neural: all-MiniLM-L6-v2</div>""", unsafe_allow_html=True)

# Theme Styles
if theme_choice:
    st.markdown("<style>.stApp { background-color: #fdfdfd; color: #1e293b; }</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>.stApp { background: radial-gradient(circle at top right, #111827, #010409); color: #f9fafb; }</style>", unsafe_allow_html=True)

# ==========================================
# 🛠️ 4. BACKEND UTILITIES
# ==========================================
def call_gemini_safe(prompt, is_vision=False, img=None):
    # CRITICAL: This checks your Streamlit Secrets
    api_keys = st.secrets.get("GEMINI_KEYS", [])
    if not api_keys:
        return "⚠️ Error: API Key Missing. Please add 'GEMINI_KEYS' to your Streamlit Secrets."
    
    for key in api_keys:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            if is_vision and img:
                response = model.generate_content([prompt, img])
            else:
                response = model.generate_content(prompt)
            return response.text
        except Exception:
            continue
    return "❌ Error: API Connection Failed. Check internet or key validity."

def retrieve(query, top_k=3):
    query_emb = st.session_state.embed_model.encode([query])
    _, I = st.session_state.index.search(np.array(query_emb).astype('float32'), top_k)
    return [st.session_state.chunks[i] for i in I[0]]

# ==========================================
# 🚀 5. MAIN INTERFACE
# ==========================================
if st.session_state.index:
    tabs = st.tabs(["📊 Dashboard", "💬 Tutor", "🧠 Quiz", "📸 Vision", "🧪 Simulation", "📚 Research"])
    
    with tabs[0]: # Dashboard
        st.subheader("Personalized Learning Analytics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mastery Score", f"{st.session_state.student_stats['mastery']}%")
        c2.metric("Quizzes Completed", st.session_state.student_stats['quizzes'])
        c3.metric("Path Progress", f"{st.session_state.student_stats['progress']}%")
        st.progress(st.session_state.student_stats['progress'] / 100)

    with tabs[1]: # Tutor
        st.subheader("Verified Biotech Tutor")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        if q := st.chat_input("Ask a technical question..."):
            st.session_state.messages.append({"role": "user", "content": q})
            with st.chat_message("user"): st.markdown(q)
            context = "\n".join(retrieve(q))
            with st.status("Thinking..."):
                ans = call_gemini_safe(f"Context: {context}\n\nQuestion: {q}")
                st.session_state.messages.append({"role": "assistant", "content": ans})
                st.rerun()

    with tabs[2]: # Quiz
        st.subheader("Adaptive Assessment")
        if st.button("Generate Contextual Quiz"):
            st.session_state.last_quiz = call_gemini_safe(f"Create a 5 MCQ quiz from: {st.session_state.chunks[:3]}")
        if st.session_state.last_quiz:
            st.write(st.session_state.last_quiz)

    with tabs[3]: # Vision
        st.subheader("Vision Agent")
        img_file = st.file_uploader("Upload Lab Image", type=['jpg','png','jpeg'])
        if img_file and st.button("Analyze Image"):
            img = Image.open(img_file)
            st.image(img, use_container_width=True)
            res = call_gemini_safe("Analyze this biotech image.", is_vision=True, img=img)
            st.info(res)

    with tabs[4]: # Simulation
        st.subheader("Protocol Simulator")
        proto = st.text_input("Experiment Name", value="Bacterial Growth Kinetics")
        if st.button("Generate Simulation"):
            with st.spinner("Processing..."):
                res = call_gemini_safe(f"Generate Python code to simulate: {proto}")
                st.session_state.last_sim = res
        if st.session_state.last_sim:
            st.code(st.session_state.last_sim, language='python')

    with tabs[5]: # Research
        st.subheader("Research Scout")
        topic = st.text_input("Topic")
        if st.button("Scout Research"):
            res = call_gemini_safe(f"Summarize 3 breakthroughs in: {topic}")
            st.markdown(res)
else:
    st.info("👈 Please upload a PDF in the sidebar and click 'Initialize Neural Engine' to unlock the platform.")
