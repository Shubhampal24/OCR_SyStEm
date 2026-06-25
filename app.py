import streamlit as st
from PIL import Image
import json
import os
import fitz  # PyMuPDF for PDF support

# Import our custom backend modules
from src.core.logger import get_logger
from src.core.config import settings
from src.image_processing import ImageProcessor
from src.ocr_engine import OCREngine
from src.llm_engine import LLMEngine
from src.validation import Validator

logger = get_logger(__name__)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="DocuAI v2 | Intelligent Extraction",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Custom CSS ---
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    try:
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        logger.warning(f"Could not load custom CSS: {e}")

load_css()

# --- Initialize Session State ---
# We use session state to keep data persistent so the chatbot remembers history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "extraction_done" not in st.session_state:
    st.session_state.extraction_done = False
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "final_data" not in st.session_state:
    st.session_state.final_data = {}
if "validation_warnings" not in st.session_state:
    st.session_state.validation_warnings = None

# --- Caching Heavy AI Models ---
@st.cache_resource
def load_models():
    logger.info("Initializing backend engines via Streamlit cache.")
    img_proc = ImageProcessor()
    ocr = OCREngine()
    llm = LLMEngine()
    return img_proc, ocr, llm

# --- Sidebar: Enterprise Settings & Engine Status ---
with st.sidebar:
    st.title("✨ DocuAI v2")
    st.markdown("---")
    st.markdown("**Engine Status:**")
    
    # Sleek boot sequence UI
    with st.status("Booting Neural Engines...", expanded=True) as status:
        st.write("Initializing RapidOCR (ONNX)...")
        st.write(f"Loading LLM: {settings.LLM_MODEL_NAME}...")
        try:
            image_processor, ocr_engine, llm_engine = load_models()
            status.update(label="All Engines Online 🟢", state="complete", expanded=False)
            models_loaded = True
        except Exception as e:
            status.update(label="Engine Failure 🔴", state="error", expanded=True)
            st.error(f"Error: {e}")
            models_loaded = False
            
    st.markdown("---")
    confidence_thresh = st.slider("OCR Confidence Threshold", 0.0, 1.0, 0.50, 0.05)
    st.caption("Lowering this increases noise but captures faint text. RapidOCR defaults to 0.5.")
    
    if st.button("🗑️ Reset Session"):
        st.session_state.clear()
        st.rerun()

if not models_loaded:
    st.stop()

# --- Main Enterprise UI ---
st.title("Intelligent Document Processing")
st.markdown("Automate data extraction using PaddleOCR Vision and HuggingFace LLMs with Zero Hallucination.")
st.markdown("---")

uploaded_file = st.file_uploader("Drop your document here (PDF, PNG, JPG)", type=["pdf", "png", "jpg", "jpeg", "webp", "tiff", "jfif"])

if uploaded_file is not None:
    # Check if this is a NEW file
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.extraction_done = False
        st.session_state.chat_history = []  # Clear chat history for new doc
        
    main_col1, main_col2 = st.columns([1, 1.5], gap="large")
    
    with main_col1:
        st.subheader("Document Preview")
        if uploaded_file.name.lower().endswith(".pdf"):
            st.info("Extracting first page of PDF...")
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            page = doc.load_page(0)
            pix = page.get_pixmap()
            pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            uploaded_file.seek(0)
        else:
            pil_image = Image.open(uploaded_file)
            
        st.image(pil_image, clamp=True)
        
    with main_col2:
        st.subheader("AI Pipeline")
            
        if not st.session_state.extraction_done:
            if st.button("▶ Run Extraction Sequence", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Phase 2
                status_text.info("⚙️ Optimizing Image (Size & Contrast)...")
                try:
                    processed_image_array = image_processor.process_image(pil_image)
                    progress_bar.progress(30)
                except Exception as e:
                    st.error(f"Image Processing Failed: {e}")
                    st.stop()
                    
                # Phase 3
                status_text.info("👁️ Extracting Text via RapidOCR (PaddleOCR ONNX)...")
                try:
                    raw_text = ocr_engine.extract_text(processed_image_array, confidence_threshold=confidence_thresh)
                    progress_bar.progress(60)
                    if not raw_text.strip():
                        st.warning("OCR could not find any readable text. Please try another image.")
                        st.stop()
                except Exception as e:
                    st.error(f"OCR Failed: {e}")
                    st.stop()
                    
                # Phase 4
                status_text.info("🧠 Analyzing context with LLM (Few-Shot Prompting)...")
                try:
                    raw_json = llm_engine.extract_information(raw_text)
                    progress_bar.progress(85)
                except Exception as e:
                    st.error(f"LLM Failed: {e}")
                    st.stop()
                    
                # Phase 5
                status_text.info("🛡️ Enforcing Business Rules (Soft Validation)...")
                final_data = Validator.validate_data(raw_json)
                
                # Save to session
                st.session_state.raw_text = raw_text
                st.session_state.final_data = final_data
                st.session_state.validation_warnings = final_data.pop("validation_warnings", None)
                st.session_state.extraction_done = True
                
                progress_bar.progress(100)
                status_text.success("✅ Extraction Complete.")
                st.rerun() # Refresh to show tabs
        else:
            # Show Results in modern Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Structured Data", "💬 Chat with Doc", "📝 Raw OCR", "🛡️ Validation Log"])
            
            with tab1:
                doc_type = st.session_state.final_data.get("document_type", "Unknown")
                st.metric(label="Detected Document Type", value=doc_type.upper())
                
                st.markdown("### Extracted Entities")
                for key, value in st.session_state.final_data.items():
                    if key not in ["document_type", "corrected_text"] and value is not None:
                        display_name = key.replace("_", " ").title()
                        st.text_input(label=display_name, value=str(value), disabled=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                json_string = json.dumps(st.session_state.final_data, indent=2)
                st.download_button(label="⬇️ Download JSON", file_name="extracted_data.json", mime="application/json", data=json_string)
                        
            with tab2:
                st.markdown("### Interactive Chatbot")
                st.caption("Ask questions about this document. Memory is preserved.")
                
                # Display chat history
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])
                
                # Chat input
                if prompt := st.chat_input("Ask a question..."):
                    # Display user msg
                    with st.chat_message("user"):
                        st.write(prompt)
                    # Add to history
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    
                    # Get AI response
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = llm_engine.answer_question_with_history(
                                st.session_state.chat_history, 
                                st.session_state.raw_text
                            )
                            st.write(response)
                            st.session_state.chat_history.append({"role": "assistant", "content": response})

            with tab3:
                st.markdown("### Raw RapidOCR Output")
                st.text_area("What the Vision Engine 'saw'", st.session_state.raw_text, height=200, disabled=True)
                if "corrected_text" in st.session_state.final_data:
                    st.markdown("### LLM Corrected Text")
                    st.text_area("Grammatically fixed text", st.session_state.final_data["corrected_text"], height=150, disabled=True)
                
            with tab4:
                if st.session_state.validation_warnings:
                    st.warning("⚠️ Data Validation Warnings Found")
                    st.info(st.session_state.validation_warnings)
                else:
                    st.success("✅ All data perfectly matches Indian business formatting rules.")
