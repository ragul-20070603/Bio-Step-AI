import streamlit as st
import google.generativeai as genai
import PyPDF2
import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from google.api_core import exceptions
import sqlite3
import streamlit_authenticator as stauth

# ==========================================
# 🔐 1. INITIALIZATION & SESSION PERSISTENCE
# ==========================================
st.set_page_config(page_title="Bio-Step AI", page_icon="🧬", layout="wide")

# Initialize Session States
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

# --- DATABASE SETUP ---
def init_db():
    with sqlite3.connect('biostep_users.db') as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users 
                     (username TEXT PRIMARY KEY, name TEXT, mastery REAL, progress INTEGER)''')
        conn.commit()

init_db()

# ==========================================
# 🔑 2. AUTHENTICATION CONFIG
# ==========================================
names = ['Ragul P', 'Reena J', 'Pranathi P K', 'Rohith G']
usernames = ['ragul', 'reena', 'pranathi', 'rohith']
passwords = ['rag2007', 'reen2006', 'pran2007', 'rgm2006']

authenticator = stauth.Authenticate(
    {'usernames': {u: {'name': n, 'password': p} for u, n, p in zip(usernames, names, passwords)}},
    'biostep_cookie', 'auth_key', cookie_expiry_days=30
)

# --- RENDER LOGIN ---
if not st.session_state.get("authentication_status"):
    col1, col2, col3 = st.columns([1, 1.2, 1]) 
    with col2:
        st.write("#")
        st.markdown("<h2 style='text-align: center; color: #818cf8;'>🧬 Bio-Step AI Portal</h2>", unsafe_allow_html=True)
        authenticator.login(location='main')
        if st.session_state.get("authentication_status") == False:
            st.error('Username/password is incorrect')
        elif st.session_state.get("authentication_status") is None:
            st.info('Please enter your biotech credentials')
    if not st.session_state.get("authentication_status"):
        st.stop() 

# --- USER CONTEXT ---
name = st.session_state["name"]
username = st.session_state["username"]

# ==========================================
# 🎨 3. UI CUSTOMIZATION (THEMING)
# ==========================================
with st.sidebar:
    st.title(f"Welcome, {name}!")
    theme_choice = st.toggle("☀️ Light Mode", value=False)
    st.write("---")
    authenticator.logout('Logout', 'sidebar')

if theme_choice:
    st.markdown("""
        <style>
        .stApp { background-color: #fdfdfd; color: #1e293b; }
        .stTabs [data-baseweb="tab-panel"] { 
            background: white; border: 1px solid #e2e8f0; border-radius: 20px; padding: 30px;
        }
        .stTabs [aria-selected="true"] { background-color: #4f46e5 !important; color: white !important; }
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp { background: radial-gradient(circle at top right, #111827, #010409); color: #f9fafb; }
        .stTabs [data-baseweb="tab-panel"] { 
            background: rgba(22, 27, 34, 0.7); backdrop-filter: blur(12px);
            border: 1px solid #30363d; border-radius: 24px; padding: 32px;
        }
        .stTabs [data-baseweb="tab"] { background-color: #21262d; border-radius: 12px; color: #8b949e; padding: 10px 20px; }
        .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #4f46e5, #7c3aed) !important; color: white !important; }
        </style>
        """, unsafe_allow_html=True)

# ==========================================
# 🛠️ 4. BACKEND UTILITIES
# ==========================================
def call_gemini_safe(prompt, is_vision=False, img=None):
    api_keys = st.secrets.get("GEMINI_KEYS", [])
    if not api_keys: return "Error: No API keys in Secrets."
    
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
    return "Error: API Keys exhausted or connection failed."

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

# ==========================================
# 🚀 5. CORE APP FEATURES
# ==========================================
with st.sidebar:
    st.header("📂 Knowledge Ingestion")
    file = st.file_uploader("Upload Syllabus/Notes (PDF)", type="pdf")
    if file and st.button("Build AI Knowledge Base"):
        with st.spinner("Processing..."):
            text = extract_text(file)
            st.session_state.index, st.session_state.chunks = build_db(text)
            st.success("System Ready!")

if st.session_state.index:
    tabs = st.tabs(["📊 Dashboard", "💬 Tutor", "🧠 Quiz", "📸 Vision", "🧪 Simulation", "📚 Research"])
    
    # 1. DASHBOARD
    with tabs[0]:
        st.subheader("Personalized Learning Analytics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mastery Score", f"{st.session_state.student_stats['mastery']}%")
        c2.metric("Quizzes Completed", st.session_state.student_stats['quizzes'])
        c3.metric("Path Progress", f"{st.session_state.student_stats['progress']}%")
        st.progress(st.session_state.student_stats['progress'] / 100)

    # 2. TUTOR (RAG)
    with tabs[1]:
        
        st.subheader("Verified Biotech Tutor")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
        if q := st.chat_input("Ask a technical question..."):
            st.session_state.messages.append({"role": "user", "content": q})
            with st.chat_message("user"): st.markdown(q)
            context = "\n".join(retrieve(q))
            with st.status("Agentic Verification..."):
                draft = call_gemini_safe(f"Explain: {q}\nContext: {context}")
                verified = call_gemini_safe(f"Critic: Fix errors based ONLY on context: {draft}\nContext: {context}")
                with st.chat_message("assistant"): st.markdown(verified)
                st.session_state.messages.append({"role": "assistant", "content": verified})

    # 3. QUIZ
    with tabs[2]:
        st.subheader("Adaptive Assessment")
        if st.button("Generate Contextual Quiz"):
            with st.spinner("Creating..."):
                st.session_state.last_quiz = call_gemini_safe(f"Create a 5 MCQ quiz from: {st.session_state.chunks[:5]}")
        if st.session_state.last_quiz:
            st.write(st.session_state.last_quiz)
            score = st.slider("Score (0-5)", 0, 5, 4)
            topic = st.text_input("Topic Tested")
            if st.button("Submit to Dashboard"):
                st.session_state.student_stats['quizzes'] += 1
                st.session_state.student_stats['mastery'] = (score/5)*100
                st.session_state.student_stats['progress'] = min(100, st.session_state.student_stats['progress'] + 10)
                if score < 4: st.session_state.student_stats['weak_topics'].append(topic)
                with sqlite3.connect('biostep_users.db') as conn:
                    c = conn.cursor()
                    c.execute("INSERT OR REPLACE INTO users VALUES (?, ?, ?, ?)", 
                              (username, name, st.session_state.student_stats['mastery'], st.session_state.student_stats['progress']))
                    conn.commit()
                st.rerun()

    # 4. VISION
    with tabs[3]:
        st.subheader("Lab-to-Logic Vision Agent")
        img_file = st.file_uploader("Upload Lab Visuals", type=['jpg','png','jpeg'])
        if img_file and st.button("Analyze Visual Data"):
            img = Image.open(img_file)
            st.image(img, use_container_width=True)
            with st.spinner("Analyzing..."):
                st.session_state.last_vision = call_gemini_safe("Analyze this biotech image technical findings.", is_vision=True, img=img)
        if st.session_state.last_vision: st.info(st.session_state.last_vision)

    # 5. SIMULATION
    with tabs[4]:
        st.subheader("Protocol Logic Simulator")
        proto = st.text_input("Experiment Name")
        if st.button("Simulate Outcome"):
            with st.spinner("Running Logic..."):
                st.session_state.last_sim = call_gemini_safe(f"Generate Python logic for protocol: {proto}")
        if st.session_state.last_sim: st.code(st.session_state.last_sim, language='python')

    # 6. RESEARCH
    with tabs[5]:
        st.subheader("Research Scout")
        topic_scout = st.text_input("Search Latest Literature")
        if st.button("Scout bioRxiv/PubMed"):
            with st.spinner("Scouting..."):
                st.session_state.last_scout = call_gemini_safe(f"Find 3 paper summaries about: {topic_scout}")
        if st.session_state.last_scout: st.markdown(st.session_state.last_scout)
else:
    st.info("👈 Please upload a Biotech document in the sidebar to unlock the platform.")
