import streamlit as st
import google.generativeai as genai
import PyPDF2
import faiss
import numpy as np
import time
from PIL import Image
from sentence_transformers import SentenceTransformer
from google.api_core import exceptions

# ==============================
# 🔐 CONFIG & KEY ROTATION
# ==============================
st.set_page_config(page_title="Bio-Step AI", page_icon="🧬", layout="wide")

# Function to rotate through multiple keys
def call_gemini_safe(prompt, is_vision=False, img=None):
    # Fetch keys from Streamlit Secrets
    api_keys = st.secrets.get("GEMINI_KEYS", [])
    
    if not api_keys:
        st.error("No API keys found in Secrets!")
        return None

    for key in api_keys:
        try:
            genai.configure(api_key=key)
            model_name = "gemini-2.5-flash"
            model = genai.GenerativeModel(model_name)
            
            if is_vision and img:
                response = model.generate_content([prompt, img])
            else:
                response = model.generate_content(prompt)
            
            return response.text
            
        except (exceptions.ResourceExhausted, exceptions.Unauthenticated) as e:
            # If rate limit (429) or auth error occurs, try next key
            st.warning(f"Key rotation triggered due to: {type(e).__name__}")
            continue
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return None
            
    st.error("All API keys exhausted or invalid.")
    return None

# ==============================
# 📊 SESSION STATE
# ==============================
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

    with tab1:
        st.subheader("Your Progress")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mastery Score", f"{st.session_state.student_stats['mastery']}%")
        col2.metric("Quizzes Done", st.session_state.student_stats['quizzes'])
        st.write(f"Learning Path Progress: {st.session_state.student_stats['progress']}%")
        st.progress(st.session_state.student_stats['progress'] / 100)
        if st.session_state.student_stats['weak_topics']:
            st.warning(f"⚠️ Focus Needed: {', '.join(set(st.session_state.student_stats['weak_topics']))}")

    with tab2:
        st.subheader("Verified Biotech Tutor")
        if q := st.chat_input("Ask about the notes..."):
            st.chat_message("user").write(q)
            context = "\n".join(retrieve(q))
            
            with st.spinner("Tutor drafting... Critic verifying..."):
                # Multi-Agent Loop using Safe Call
                draft_prompt = f"Explain this biotech concept: {q}\nContext: {context}"
                draft = call_gemini_safe(draft_prompt)
                
                if draft:
                    critic_prompt = f"As a Scientific Critic, fix any errors in this draft using ONLY the context: {draft}\nContext: {context}"
                    verified = call_gemini_safe(critic_prompt)
                    st.chat_message("assistant").write(verified)

    with tab3:
        if st.button("Generate New Quiz"):
            quiz_prompt = f"Create a 5-question MCQ from: {st.session_state.chunks[:5]}"
            quiz = call_gemini_safe(quiz_prompt)
            if quiz:
                st.write(quiz)
                score = st.slider("Select your score (for demo)", 0, 5, 4)
                if st.button("Update Stats"):
                    st.session_state.student_stats['quizzes'] += 1
                    st.session_state.student_stats['mastery'] = (score/5)*100
                    st.session_state.student_stats['progress'] += 10
                    st.rerun()

    with tab4:
        st.subheader("Lab Vision Agent")
        img_file = st.file_uploader("Upload Lab Image", type=['jpg','png','jpeg'])
        if img_file:
            img = Image.open(img_file)
            st.image(img, caption="Uploaded Lab Data", use_container_width=True)
            if st.button("Analyze Image"):
                with st.spinner("Analyzing visual data..."):
                    vision_prompt = "Identify this biotech lab result and explain the observations."
                    res = call_gemini_safe(vision_prompt, is_vision=True, img=img)
                    if res:
                        st.info(res)
else:
    st.info("Upload a PDF to unlock features.")
