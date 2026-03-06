import streamlit as st
import google.generativeai as genai
import PyPDF2
import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import streamlit_authenticator as stauth

# ==========================================
# 🔐 1. INITIALIZATION
# ==========================================
st.set_page_config(page_title="Bio-Step AI", page_icon="🧬", layout="wide")

if 'messages' not in st.session_state: st.session_state.messages = []
if 'index' not in st.session_state: st.session_state.index = None
if 'chunks' not in st.session_state: st.session_state.chunks = []

@st.cache_resource
def load_embed_model():
    # Efficient local embeddings for the RAG engine
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()

# ==========================================
# 🔑 2. AUTHENTICATION (The "Gatekeeper")
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
# 🛠️ 3. REVOLVING API ENGINE (GEMINI 3.1)
# ==========================================
def call_gemini_3_1(prompt, is_vision=False, img=None):
    # YOUR KEYS - The system will rotate through these if one hits a limit
    keys = [
        "AIzaSyCHfiE1hNG1s1c72KA72KZe0oPF_eQokwE",
        "AIzaSyBKzsDQVtaVcuOlHYpmccrpMu-AMsCtKDA",
        "AIzaSyD__6irN9Wt3uKcXv6vC8HyId0q4JeQTBw"
    ]
    
    # MARCH 2026 BEST FREE MODEL: gemini-3.1-flash-lite-preview
    model_id = "gemini-3.1-flash-lite-preview" 
    
    for k in keys:
        try:
            genai.configure(api_key=k)
            model = genai.GenerativeModel(model_id)
            if is_vision and img:
                return model.generate_content([prompt, img]).text
            return model.generate_content(prompt).text
        except Exception as e:
            # Silently try the next key if this one fails
            continue
            
    return "❌ Connection Failed: All API keys are currently exhausted. Please wait 60 seconds."

# ==========================================
# 🎨 4. UNIFIED SIDEBAR
# ==========================================
with st.sidebar:
    st.title(f"Hi, {st.session_state['name']}!")
    authenticator.logout('Logout', 'sidebar')
    st.markdown("---")
    st.subheader("📂 Knowledge Base")
    file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
    
    if file and st.button("🚀 Index Bio-Data", use_container_width=True):
        with st.spinner("Analyzing..."):
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
            words = text.split()
            # Neural Chunking
            chunks = [" ".join(words[i:i+300]) for i in range(0, len(words), 250)]
            embeddings = embed_model.encode(chunks)
            # FAISS Vector Indexing (The "Novelty" part)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
            st.session_state.index, st.session_state.chunks = index, chunks
            st.success("Neural Engine Online!")

# ==========================================
# 🚀 5. SOLUTION INTERFACE
# ==========================================
if st.session_state.index:
    t1, t2, t3 = st.tabs(["💬 RAG Tutor", "🧪 Lab Sim", "📸 Vision Agent"])
    
    with t1: # RAG Tutor
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.write(msg["content"])
        if q := st.chat_input("Ask a technical question..."):
            st.session_state.messages.append({"role": "user", "content": q})
            # Retrieve relevant facts from Vector DB
            query_emb = embed_model.encode([q])
            _, I = st.session_state.index.search(np.array(query_emb).astype('float32'), 2)
            context = "\n".join([st.session_state.chunks[i] for i in I[0]])
            # Generate grounded response
            ans = call_gemini_3_1(f"Use this context: {context}\n\nQuestion: {q}")
            st.session_state.messages.append({"role": "assistant", "content": ans})
            st.rerun()

    with t2: # Lab Simulation
        st.subheader("Bacterial Kinetics Simulator")
        if st.button("Generate Python Sim"):
            code = call_gemini_3_1("Generate Python code to simulate CRISPR gene editing efficacy.")
            st.code(code, language='python')

    with t3: # Vision
        img_file = st.file_uploader("Upload Gel Electrophoresis Image", type=['png','jpg'])
        if img_file and st.button("Analyze Image"):
            img = Image.open(img_file)
            st.image(img, caption="Lab Sample")
            res = call_gemini_3_1("Identify the bands and anomalies in this biotech image.", True, img)
            st.info(res)
else:
    st.info("👈 Upload your study materials in the sidebar to power the AI.")
