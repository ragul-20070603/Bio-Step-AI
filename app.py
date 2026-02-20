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

# Persistent Storage for AI Outputs (Prevents Vanishing)
if 'last_quiz' not in st.session_state: st.session_state.last_quiz = ""
if 'last_sim' not in st.session_state: st.session_state.last_sim = ""
if 'last_scout' not in st.session_state: st.session_state.last_scout = ""
if 'last_vision' not in st.session_state: st.session_state.last_vision = ""

# Neural Engine & Stats Initialization
if 'embed_model' not in st.session_state:
    with st.spinner("Initializing Scientific Neural Engine..."):
        st.session_state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

if 'student_stats' not in st.session_state:
    st.session_state.student_stats = {"progress": 0, "mastery": 0.0, "quizzes": 0, "weak_topics": []}

if 'index' not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('biostep_users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, name TEXT, mastery REAL, progress INTEGER)''')
    conn.commit()
    conn.close()

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

# --- RENDER CENTERED LOGIN ---
if not st.session_state.get("authentication_status"):
    st.write("#") 
    st.write("#")
    col1, col2, col3 = st.columns([1, 1.2, 1]) 
    with col2:
        st.markdown("<h2 style='text-align: center; color: #818cf8;'>🧬 Bio-Step AI Portal</h2>", unsafe_allow_html=True)
        authenticator.login(location='main')
        
        if st.session_state.get("authentication_status") == False:
            st.error('Username/password is incorrect')
        elif st.session_state.get("authentication_status") is None:
            st.info('Please enter your biotech credentials')
    
    if not st.session_state.get("authentication_status"):
        st.stop() 

# --- ACCESS USER DETAILS ---
name = st.session_state["name"]
username = st.session_state["username"]

st.sidebar.title(f"Welcome, {name}!")
authenticator.logout('Logout', 'sidebar')

# ==========================================
# 🎨 3. UI CUSTOMIZATION (DARK MODE)
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
# 🛠️ 4. BACKEND UTILITIES
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
            st.session_state.full_text = text
            st.success("System Ready!")

if st.session_state.index:
    tabs = st.tabs(["📊 Dashboard", "💬 Tutor", "🧠 Quiz", "📸 Vision", "🧪 Simulation", "📚 Research"])
    tab1, tab2, tab3, tab4, tab5, tab6 = tabs

    # 1. DASHBOARD
    with tab1:
        st.subheader("Personalized Learning Analytics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mastery Score", f"{st.session_state.student_stats['mastery']}%")
        c2.metric("Quizzes Completed", st.session_state.student_stats['quizzes'])
        c3.metric("Path Progress", f"{st.session_state.student_stats['progress']}%")
        st.progress(st.session_state.student_stats['progress'] / 100)
        if st.session_state.student_stats['weak_topics']:
            st.warning(f"⚠️ Knowledge Gaps: {', '.join(set(st.session_state.student_stats['weak_topics']))}")

    # 2. MULTI-AGENT CHAT
    # 2. MULTI-AGENT CHAT (Fixed for Persistence)
with tab2:
    st.subheader("Verified Biotech Tutor")
    
    # Initialize chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if q := st.chat_input("Ask a technical question..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
            
        context = "\n".join(retrieve(q))
        
        with st.status("Agentic Reasoning In Progress...") as status:
            st.write("Tutor drafting response...")
            draft = call_gemini_safe(f"Explain this biotech concept: {q}\nContext: {context}")
            st.write("Scientific Critic verifying accuracy...")
            verified = call_gemini_safe(f"Scientific Critic: Correct errors in this draft using ONLY the context: {draft}\nContext: {context}")
            status.update(label="Response Verified", state="complete")
            
            # Display and SAVE the verified response to history
            with st.chat_message("assistant"):
                st.markdown(verified)
            st.session_state.messages.append({"role": "assistant", "content": verified})

    # 3. QUIZ (Fixed for Persistence)
    with tab3:
        st.subheader("Adaptive Assessment")
        if st.button("Generate Contextual Quiz"):
            with st.spinner("Creating Quiz..."):
                st.session_state.last_quiz = call_gemini_safe(f"Create a 5-question MCQ from: {st.session_state.chunks[:5]}")
        
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
                conn.commit(); conn.close()
                st.rerun()

    # 4. VISION (Fixed for Persistence)
    with tab4:
        st.subheader("Lab-to-Logic Vision Agent")
        img_file = st.file_uploader("Upload Gel/Chart", type=['jpg','png','jpeg'])
        if img_file and st.button("Analyze Visual Data"):
            img = Image.open(img_file)
            st.image(img, width='stretch')
            with st.spinner("Analyzing Image..."):
                st.session_state.last_vision = call_gemini_safe("Analyze this biotech image and explain results.", is_vision=True, img=img)
        
        if st.session_state.last_vision:
            st.info(st.session_state.last_vision)

    # 5. SIMULATION (Fixed for Persistence)
    with tab5:
        st.subheader("Protocol logic Simulator")
        proto = st.text_input("Experiment Name (e.g. CRISPR In-Vitro)")
        if st.button("Simulate Outcome"):
            with st.spinner("Generating Simulation..."):
                st.session_state.last_sim = call_gemini_safe(f"Generate Python logic for this experiment based on notes: {proto}")
        
        if st.session_state.last_sim:
            st.code(st.session_state.last_sim, language='python')

    # 6. RESEARCH (Fixed for Persistence)
    with tab6:
        st.subheader("Research Scout")
        topic_scout = st.text_input("Search Latest Literature")
        if st.button("Scout bioRxiv/PubMed"):
            with st.spinner("Searching Papers..."):
                st.session_state.last_scout = call_gemini_safe(f"Find 3 hypothetical recent paper summaries about: {topic_scout}")
        
        if st.session_state.last_scout:
            st.markdown(st.session_state.last_scout)
        else:
            st.info("👈 Please upload a Biotech document in the sidebar to unlock the platform.")


