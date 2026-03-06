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

st.set_page_config(page_title="Bio-Step AI", page_icon="🧬", layout="wide")

if 'last_quiz' not in st.session_state: st.session_state.last_quiz = ""
if 'last_sim' not in st.session_state: st.session_state.last_sim = ""
if 'last_scout' not in st.session_state: st.session_state.last_scout = ""
if 'last_vision' not in st.session_state: st.session_state.last_vision = ""
if 'messages' not in st.session_state: st.session_state.messages = []

if 'embed_model' not in st.session_state:
    with st.spinner("Initializing Scientific Neural Engine..."):
        st.session_state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

if 'student_stats' not in st.session_state:
    st.session_state.student_stats = {"progress": 0, "mastery": 0.0, "quizzes": 0, "weak_topics": []}

if 'index' not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []

def init_db():
    conn = sqlite3.connect('biostep_users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, name TEXT, mastery REAL, progress INTEGER)''')
    conn.commit()
    conn.close()

init_db()

names = ['Ragul P', 'Reena J', 'Pranathi P K', 'Rohith G']
usernames = ['ragul', 'reena', 'pranathi', 'rohith']
passwords = ['rag2007', 'reen2006', 'pran2007', 'rgm2006']

authenticator = stauth.Authenticate(
    {'usernames': {u: {'name': n, 'password': p} for u, n, p in zip(usernames, names, passwords)}},
    'biostep_cookie', 'auth_key', cookie_expiry_days=30
)

if not st.session_state.get("authentication_status"):
    st.write("#")
    st.write("#")
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center; color: #2dd4bf;'>🧬 Bio-Step AI Portal</h2>", unsafe_allow_html=True)
        authenticator.login(location='main')
        if st.session_state.get("authentication_status") == False:
            st.error('Username/password is incorrect')
        elif st.session_state.get("authentication_status") is None:
            st.info('Please enter your biotech credentials')
    if not st.session_state.get("authentication_status"):
        st.stop()

name = st.session_state["name"]
username = st.session_state["username"]

with st.sidebar:
    st.title(f"Welcome, {name}!")
    theme = st.toggle("☀️ Light Mode", value=False)
    st.write("---")
    authenticator.logout('Logout', 'sidebar')

if theme:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
        .stApp {
            background-color: #f4f6f7;
            background-image:
                radial-gradient(circle at 20% 10%, rgba(45,212,191,0.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 90%, rgba(20,184,166,0.06) 0%, transparent 50%);
            color: #1e2a2a;
            font-family: 'DM Sans', sans-serif;
        }
        [data-testid="stSidebar"] {
            background-color: #e8edef;
            border-right: 1px solid #c8d4d4;
        }
        [data-testid="stSidebar"] * { color: #1e2a2a !important; }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2 { color: #0f766e !important; }
        .stTabs [data-baseweb="tab-list"] { background: transparent; gap: 6px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #dce6e6;
            border-radius: 10px;
            color: #4a6060;
            font-weight: 500;
            padding: 10px 22px;
            font-family: 'DM Sans', sans-serif;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0f766e !important;
            color: #f0fdfb !important;
            box-shadow: 0 4px 14px rgba(15,118,110,0.3);
        }
        .stTabs [data-baseweb="tab-panel"] {
            background: rgba(255,255,255,0.75);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(200,212,212,0.6);
            border-radius: 18px;
            box-shadow: 0 6px 24px rgba(15,118,110,0.07);
            padding: 28px;
            margin-top: 10px;
        }
        .stChatMessage {
            background-color: #ffffff !important;
            border: 1px solid #c8d4d4 !important;
            border-radius: 14px;
        }
        .stChatMessage [data-testid="stMarkdownContainer"] p { color: #1e2a2a !important; font-size: 1rem; }
        .stButton>button {
            width: 100%; border-radius: 10px; height: 48px;
            background: #0f766e; border: none; font-weight: 600;
            color: #f0fdfb; font-family: 'DM Sans', sans-serif;
            letter-spacing: 0.3px; transition: background 0.2s, box-shadow 0.2s;
        }
        .stButton>button:hover { background: #0d6b64; box-shadow: 0 4px 16px rgba(15,118,110,0.25); }
        .stTextInput>div>div>input, .stChatInputContainer textarea {
            background: #ffffff !important; border: 1px solid #a8c0c0 !important;
            border-radius: 10px !important; color: #1e2a2a !important;
            font-family: 'DM Mono', monospace !important;
        }
        [data-testid="metric-container"] {
            background: #ffffff; border: 1px solid #c8d4d4; border-radius: 14px; padding: 16px;
        }
        [data-testid="metric-container"] label { color: #4a6060 !important; }
        [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #0f766e !important; font-weight: 700; }
        [data-testid="stProgressBar"] > div { background-color: #2dd4bf !important; }
        .stCode { font-family: 'DM Mono', monospace !important; }
        h1, h2, h3 { color: #0f766e; font-family: 'DM Sans', sans-serif; font-weight: 700; }
        </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
        .stApp {
            background-color: #0d1a1a;
            background-image:
                radial-gradient(ellipse at 15% 0%, rgba(45,212,191,0.10) 0%, transparent 45%),
                radial-gradient(ellipse at 85% 100%, rgba(20,184,166,0.08) 0%, transparent 45%);
            color: #e2f0ef;
            font-family: 'DM Sans', sans-serif;
        }
        [data-testid="stSidebar"] { background-color: #0a1212; border-right: 1px solid #1a3333; }
        [data-testid="stSidebar"] * { color: #b2cece !important; }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #2dd4bf !important; }
        .stTabs [data-baseweb="tab-list"] { background: transparent; gap: 6px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #132424; border-radius: 10px; color: #7aaaa8;
            font-weight: 500; padding: 10px 22px; font-family: 'DM Sans', sans-serif;
            border: 1px solid #1a3333;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #0f766e, #0d9488) !important;
            color: #f0fdfb !important; border-color: transparent !important;
            box-shadow: 0 4px 18px rgba(13,148,136,0.35);
        }
        .stTabs [data-baseweb="tab-panel"] {
            background: rgba(13, 26, 26, 0.80); backdrop-filter: blur(14px);
            border: 1px solid #1a3333; border-radius: 20px;
            box-shadow: 0 20px 50px rgba(0,0,0,0.45); padding: 32px; margin-top: 10px;
        }
        .stChatMessage {
            background-color: #132424 !important; border: 1px solid #1e3c3c !important; border-radius: 14px;
        }
        .stChatMessage [data-testid="stMarkdownContainer"] p {
            color: #d4ecea !important; font-size: 1.05rem; line-height: 1.7;
        }
        .stButton>button {
            width: 100%; border-radius: 10px; height: 48px;
            background: linear-gradient(90deg, #0f766e, #0d9488); border: none;
            font-weight: 700; color: #f0fdfb; font-family: 'DM Sans', sans-serif;
            letter-spacing: 0.3px; transition: opacity 0.2s, box-shadow 0.2s;
        }
        .stButton>button:hover { opacity: 0.88; box-shadow: 0 6px 20px rgba(13,148,136,0.40); }
        .stTextInput>div>div>input, .stChatInputContainer textarea {
            background: #132424 !important; border: 1px solid #1e3c3c !important;
            border-radius: 10px !important; color: #e2f0ef !important;
            font-family: 'DM Mono', monospace !important;
        }
        [data-testid="metric-container"] {
            background: #132424; border: 1px solid #1e3c3c; border-radius: 14px; padding: 16px;
        }
        [data-testid="metric-container"] label { color: #7aaaa8 !important; }
        [data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #2dd4bf !important; font-weight: 700; font-size: 1.6rem;
        }
        [data-testid="stProgressBar"] > div { background-color: #2dd4bf !important; }
        .stCode, code {
            background: #0a1212 !important; border: 1px solid #1e3c3c !important;
            font-family: 'DM Mono', monospace !important; color: #5eead4 !important;
        }
        .stAlert { border-radius: 12px !important; border-left-color: #2dd4bf !important; }
        h1, h2, h3 { color: #2dd4bf; font-family: 'DM Sans', sans-serif; font-weight: 700; }
        [data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stTickBar"] { color: #7aaaa8; }
        </style>
    """, unsafe_allow_html=True)

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
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        if q := st.chat_input("Ask a technical question..."):
            st.session_state.messages.append({"role": "user", "content": q})
            with st.chat_message("user"): st.markdown(q)
            context = "\n".join(retrieve(q))
            with st.status("Agentic Verification...") as status:
                draft = call_gemini_safe(f"Explain: {q}\nContext: {context}")
                verified = call_gemini_safe(f"Critic: Fix errors in draft based ONLY on context: {draft}\nContext: {context}")
                status.update(label="Response Verified", state="complete")
                with st.chat_message("assistant"): st.markdown(verified)
                st.session_state.messages.append({"role": "assistant", "content": verified})

    with tab3:
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
                conn = sqlite3.connect('biostep_users.db')
                c = conn.cursor()
                c.execute("REPLACE INTO users VALUES (?, ?, ?, ?)", (username, name, st.session_state.student_stats['mastery'], st.session_state.student_stats['progress']))
                conn.commit(); conn.close(); st.rerun()

    with tab4:
        st.subheader("Lab-to-Logic Vision Agent")
        img_file = st.file_uploader("Upload Lab Visuals", type=['jpg','png','jpeg'])
        if img_file and st.button("Analyze Visual Data"):
            img = Image.open(img_file); st.image(img, width='stretch')
            with st.spinner("Analyzing..."):
                st.session_state.last_vision = call_gemini_safe("Analyze this biotech image technical findings.", is_vision=True, img=img)
        if st.session_state.last_vision: st.info(st.session_state.last_vision)

    with tab5:
        st.subheader("Protocol Logic Simulator")
        proto = st.text_input("Experiment Name")
        if st.button("Simulate Outcome"):
            with st.spinner("Running Logic..."):
                st.session_state.last_sim = call_gemini_safe(f"Generate Python logic for protocol: {proto}")
        if st.session_state.last_sim: st.code(st.session_state.last_sim, language='python')

    with tab6:
        st.subheader("Research Scout")
        topic_scout = st.text_input("Search Latest Literature")
        if st.button("Scout bioRxiv/PubMed"):
            with st.spinner("Scouting..."):
                st.session_state.last_scout = call_gemini_safe(f"Find 3 paper summaries about: {topic_scout}")
        if st.session_state.last_scout: st.markdown(st.session_state.last_scout)
else:
    st.info("👈 Please upload a Biotech document in the sidebar to unlock the platform.")
