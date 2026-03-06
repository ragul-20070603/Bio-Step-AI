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

# Persistent State
if 'messages' not in st.session_state: st.session_state.messages = []
if 'index' not in st.session_state: st.session_state.index = None
if 'chunks' not in st.session_state: st.session_state.chunks = []
if 'last_sim' not in st.session_state: st.session_state.last_sim = ""

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
        st.markdown("<h2 style='text-align: center;'>🧬 Bio-Step AI Portal</h2>", unsafe_allow_html=True)
        authenticator.login(location='main')
    st.stop()

# ==========================================
# 🎨 3. THE CLEAN SIDEBAR (ONLY ONE BLOCK)
# ==========================================
with st.sidebar:
    st.title(f"Welcome, {st.session_state['name']}!")
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
            st.success("Knowledge Base Built!")

# ==========================================
# 🛠️ 4. ROBUST API UTILITY
# ==========================================
def call_gemini(prompt, is_vision=False, img=None):
    # These are the keys from your screenshot
    keys = [
        "AIzaSyCHfiE1hNG1s1c72KA72KZe0oPF_eQokwE",
        "AIzaSyBKzsDQVtaVcuOlHYpmccrpMu-AMsCtKDA",
        "AIzaSyD__6irN9Wt3uKcXv6vC8HyId0q4JeQTBw"
    ]
    
    for k in keys:
        try:
            genai.configure(api_key=k)
            # Try 1.5-flash first, fallback to pro if restricted
            model_name = "gemini-1.5-flash" 
            model = genai.GenerativeModel(model_name)
            
            if is_vision and img:
                return model.generate_content([prompt, img]).text
            return model.generate_content(prompt).text
        except Exception as e:
            continue # Try next key
    return "❌ Error: All keys failed or Rate Limited."

# ==========================================
# 🚀 5. MAIN TABS
# ==========================================
if st.session_state.index:
    tabs = st.tabs(["💬 Tutor", "🧪 Simulation", "📸 Vision", "📚 Research"])
    
    with tabs[0]: # Tutor
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.write(msg["content"])
        if q := st.chat_input("Ask about your PDF..."):
            st.session_state.messages.append({"role": "user", "content": q})
            # Simple RAG: find top 2 chunks
            query_emb = embed_model.encode([q])
            _, I = st.session_state.index.search(np.array(query_emb).astype('float32'), 2)
            context = "\n".join([st.session_state.chunks[i] for i in I[0]])
            ans = call_gemini(f"Context: {context}\n\nQuestion: {q}")
            st.session_state.messages.append({"role": "assistant", "content": ans})
            st.rerun()

    with tabs[1]: # Simulation
        st.subheader("Protocol Simulator")
        exp = st.text_input("Experiment Name", "Bacterial Growth Kinetics")
        if st.button("Generate Simulation"):
            st.session_state.last_sim = call_gemini(f"Generate Python code for: {exp}")
        if st.session_state.last_sim: st.code(st.session_state.last_sim)

    with tabs[2]: # Vision
        img_file = st.file_uploader("Upload Lab Photo", type=['png','jpg'])
        if img_file and st.button("Analyze"):
            img = Image.open(img_file)
            st.image(img)
            st.write(call_gemini("Analyze this biotech image.", True, img))

    with tabs[3]: # Research
        topic = st.text_input("Research Topic")
        if st.button("Search"):
            st.write(call_gemini(f"Summarize breakthroughs in {topic}"))
else:
    st.info("👈 Please upload your Biotech PDF and click 'Initialize' to start.")
