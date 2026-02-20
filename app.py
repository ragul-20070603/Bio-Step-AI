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
# 🔐 CONFIG & AUTHENTICATION
# ==========================================
st.set_page_config(page_title="Bio-Step AI", page_icon="🧬", layout="wide")

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('biostep_users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, name TEXT, mastery REAL, progress INTEGER)''')
    conn.commit()
    conn.close()

init_db()

# --- AUTHENTICATION CONFIG ---
names = ['Ragul P', 'Reena J', 'Pranathi P K', 'Rohith G']
usernames = ['ragul', 'reena', 'pranathi', 'rohith']
passwords = ['rag2007', 'reen2006', 'pran2007', 'rgm2006']

# Note: In production, passwords must be hashed using stauth.Hasher(passwords).generate()
authenticator = stauth.Authenticate(
    {'usernames': {u: {'name': n, 'password': p} for u, n, p in zip(usernames, names, passwords)}},
    'biostep_cookie', 'auth_key', cookie_expiry_days=30
)

# --- RENDER LOGIN (FIXED SYNTAX) ---
# Argument 1 is now 'location'. Form name goes into the 'fields' dict.
login_result = authenticator.login(location='main')

# New versions return a tuple, but status is also in session_state
if st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')
    st.stop()
elif st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')
    st.stop()

# --- IF AUTHENTICATED ---
name = st.session_state["name"]
username = st.session_state["username"]

st.sidebar.title(f"Welcome, {name}!")
authenticator.logout('Logout', 'sidebar')

# ==========================================
# 🎨 UI CUSTOMIZATION (DARK MODE)
# ==========================================
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e0e6ed; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .stTabs [data-baseweb="tab-panel"] { background-color: #161b22; padding: 25px; border-radius: 15px; border: 1px solid #30363d; box-shadow: 0 10px 15px rgba(0, 0, 0, 0.3); margin-top: 15px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #1f2937; border-radius: 8px; color: #9ca3af; font-weight: 600; padding: 0 20px; }
    .stTabs [aria-selected="true"] { background-color: #4f46e5 !important; color: #ffffff !important; border-bottom: 2px solid #818cf8 !important; }
    [data-testid="stMetricValue"] { color: #818cf8; font-weight: 800; }
    .stButton>button { border-radius: 10px; border: none; background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%); color: white; font-weight: bold; padding: 0.6rem 2rem; }
    .stButton>button:hover { opacity: 0.9; transform: scale(1.02); color: white; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 🛠️ BACKEND UTILITIES & KEY ROTATION
# ==========================================
def call_gemini_safe(prompt, is_vision=False, img=None):
    api_keys = st.secrets.get("GEMINI_KEYS", [])
    for key in api_keys:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            if is_vision and img:
                response = model.generate_content([prompt, img])
            else:
                response = model.generate_content(prompt)
            return response.text
        except (exceptions.ResourceExhausted, exceptions.Unauthenticated):
            continue
    return "Error: All API keys exhausted."

if 'student_stats' not in st.session_state:
    st.session_state.student_stats = {"progress": 0, "mastery": 0.0, "quizzes": 0, "weak_topics": []}
if 'index' not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []
    st.session_state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

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
# 🚀 CORE APP FEATURES
# ==========================================
with st.sidebar:
    st.header("📂 Knowledge Ingestion")
    file = st.file_uploader("Upload Syllabus/Notes (PDF)", type="pdf")
    if file and st.button("Build AI Knowledge Base"):
        with st.spinner("Processing..."):
            text = extract_text(file)
            st.session_state.index, st.session_state.chunks = build_db(text)
            st.session_state.full_text = text
            st.success("System Ready!")

if st.session_state.index:
    tabs = st.tabs(["📊 Dashboard", "💬 Tutor", "🧠 Quiz", "📸 Vision", "🧪 Simulation", "📚 Research"])
    tab1, tab2, tab3, tab4, tab5, tab6 = tabs

    with tab1:
        st.subheader("Personalized Learning Analytics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mastery Score", f"{st.session_state.student_stats['mastery']}%")
        c2.metric("Quizzes Completed", st.session_state.student_stats['quizzes'])
        c3.metric("Path Progress", f"{st.session_state.student_stats['progress']}%")
        st.progress(st.session_state.student_stats['progress'] / 100)
        if st.session_state.student_stats['weak_topics']:
            st.warning(f"⚠️ Knowledge Gaps: {', '.join(set(st.session_state.student_stats['weak_topics']))}")

    with tab2:
        st.subheader("Verified Biotech Tutor")
        if q := st.chat_input("Ask a technical question..."):
            st.chat_message("user").write(q)
            context = "\n".join(retrieve(q))
            with st.spinner("Drafting & Verifying..."):
                draft = call_gemini_safe(f"Explain this biotech concept: {q}\nContext: {context}")
                verified = call_gemini_safe(f"Scientific Critic: Correct this draft using ONLY the context: {draft}\nContext: {context}")
                st.chat_message("assistant").write(verified)

    with tab3:
        st.subheader("Adaptive Assessment")
        if st.button("Generate Contextual Quiz"):
            quiz = call_gemini_safe(f"Create a 5-question MCQ from: {st.session_state.chunks[:5]}")
            st.write(quiz)
            score = st.slider("Score (0-5)", 0, 5, 4)
            topic = st.text_input("Topic Tested")
            if st.button("Submit to Dashboard"):
                st.session_state.student_stats['quizzes'] += 1
                st.session_state.student_stats['mastery'] = (score/5)*100
                st.session_state.student_stats['progress'] = min(100, st.session_state.student_stats['progress'] + 10)
                if score < 4: st.session_state.student_stats['weak_topics'].append(topic)
                # Database Update
                conn = sqlite3.connect('biostep_users.db')
                c = conn.cursor()
                c.execute("REPLACE INTO users VALUES (?, ?, ?, ?)", (username, name, st.session_state.student_stats['mastery'], st.session_state.student_stats['progress']))
                conn.commit(); conn.close()
                st.rerun()

    with tab4:
        st.subheader("Lab-to-Logic Vision Agent")
        img_file = st.file_uploader("Upload Gel/Chart", type=['jpg','png','jpeg'])
        if img_file and st.button("Analyze Visual Data"):
            img = Image.open(img_file)
            st.image(img, use_container_width=True)
            res = call_gemini_safe("Analyze this biotech image and explain results.", is_vision=True, img=img)
            st.info(res)

    with tab5:
        st.subheader("Protocol logic Simulator")
        proto = st.text_input("Experiment Name (e.g. CRISPR In-Vitro)")
        if st.button("Simulate Outcome"):
            sim = call_gemini_safe(f"Generate Python logic for this experiment based on notes: {proto}")
            st.code(sim, language='python')

    with tab6:
        st.subheader("Research Scout")
        topic = st.text_input("Search Latest Literature")
        if st.button("Scout bioRxiv/PubMed"):
            res = call_gemini_safe(f"Find 3 hypothetical recent paper summaries about: {topic}")
            st.markdown(res)
else:
    st.info("👈 Please upload a Biotech document in the sidebar to unlock the platform.")
