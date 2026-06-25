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
    page_title="DocuAI | Enterprise Document Processing",
    page_icon="⚡",
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

# --- Caching Heavy AI Models ---
@st.cache_resource
def load_models():
    # We do NOT use st.spinner here because we handle the UI loading state in the sidebar
    logger.info("Initializing backend engines via Streamlit cache.")
    img_proc = ImageProcessor()
    ocr = OCREngine()
    llm = LLMEngine()
    return img_proc, ocr, llm

# --- Sidebar: Enterprise Settings & Engine Status ---
with st.sidebar:
    st.title("⚡ DocuAI Settings")
    st.markdown("---")
    st.markdown("**Engine Status:**")
    
    # Sleek boot sequence UI
    with st.status("Booting Neural Engines...", expanded=True) as status:
        st.write("Initializing OpenCV Filters...")
        st.write(f"Warming up EasyOCR ({settings.OCR_LANGUAGE})...")
        st.write(f"Loading LLM: {settings.LLM_MODEL_NAME}...")
        try:
            image_processor, ocr_engine, llm_engine = load_models()
            status.update(label="All Engines Online", state="complete", expanded=False)
            models_loaded = True
        except Exception as e:
            status.update(label="Engine Failure", state="error", expanded=True)
            st.error(f"Error: {e}")
            models_loaded = False
            
    st.markdown("---")
    confidence_thresh = st.slider("OCR Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    st.caption("Lowering this increases noise but captures faint text.")

if not models_loaded:
    st.stop()

# --- Main Enterprise UI ---
st.title("Intelligent Document Processing")
st.markdown("Automate data extraction from Invoices, Receipts, and Identity Documents using local, privacy-first AI.")
st.markdown("---")

# Create a massive dropzone
uploaded_file = st.file_uploader("Drop your document here (PDF, PNG, JPG, JPEG, WEBP, TIFF, BMP, JFIF)", type=["pdf", "png", "jpg", "jpeg", "webp", "tiff", "bmp", "jfif"])

if uploaded_file is not None:
    # Use columns for a sleek dashboard layout (Image on left, Extraction on right)
    main_col1, main_col2 = st.columns([1, 1.5], gap="large")
    
    with main_col1:
        st.subheader("Document Preview")
        
        # Handle PDF conversion if necessary
        if uploaded_file.name.lower().endswith(".pdf"):
            st.info("Extracting first page of PDF...")
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            page = doc.load_page(0)  # Extract first page
            pix = page.get_pixmap()
            pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            uploaded_file.seek(0)
        else:
            pil_image = Image.open(uploaded_file)
            
        # Display image with clean styling
        st.image(pil_image, use_column_width=True, clamp=True)
        
    with main_col2:
        st.subheader("Extraction Pipeline")
        
        # Clear session state if a new file is uploaded
        if "current_file" not in st.session_state or st.session_state["current_file"] != uploaded_file.name:
            st.session_state["current_file"] = uploaded_file.name
            st.session_state["extraction_done"] = False
            
        if st.button("▶ Run Extraction Sequence", use_container_width=True, type="primary"):
            
            # Custom Progress UI
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Phase 2
            status_text.info("⚙️ Optimizing Image for OCR...")
            try:
                processed_image_array = image_processor.process_image(pil_image)
                progress_bar.progress(25)
            except Exception as e:
                st.error(f"Image Processing Failed: {e}")
                st.stop()
                
            # Phase 3
            status_text.info("👁️ Extracting Raw Text via EasyOCR...")
            try:
                raw_text = ocr_engine.extract_text(processed_image_array, confidence_threshold=confidence_thresh)
                progress_bar.progress(50)
                if not raw_text:
                    st.warning("OCR could not find any readable text.")
                    st.stop()
            except Exception as e:
                st.error(f"OCR Failed: {e}")
                st.stop()
                
            # Phase 4
            status_text.info("🧠 Analyzing context with LLM... (Warning: You don't have a GPU. This can take 2-5 minutes on a normal CPU! Please wait...)")
            try:
                raw_json = llm_engine.extract_information(raw_text)
                progress_bar.progress(75)
            except Exception as e:
                st.error(f"LLM Failed: {e}")
                st.stop()
                
            # Phase 5
            status_text.info("🛡️ Enforcing Business Rules...")
            validation_errors = None
            final_data = raw_json
            try:
                final_data = Validator.validate_data(raw_json)
            except Exception as e:
                validation_errors = str(e)
            
            # Save everything to session state so it persists during Chatbot queries
            st.session_state['extraction_done'] = True
            st.session_state['raw_text'] = raw_text
            st.session_state['final_data'] = final_data
            st.session_state['validation_errors'] = validation_errors
            
            progress_bar.progress(100)
            status_text.success("✅ Extraction Complete.")
            
            st.markdown("---")
            
        # If extraction is done, show the results
        if st.session_state.get('extraction_done', False):
            final_data = st.session_state['final_data']
            raw_text = st.session_state['raw_text']
            validation_errors = st.session_state['validation_errors']
            
            # Show Results in modern Tabs instead of a long scrollable page
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Structured Data", "📝 OCR & Text", "🛡️ Validation Log", "💬 Ask AI"])
            
            with tab1:
                # Format data beautifully using metrics and text inputs instead of raw JSON string
                doc_type = final_data.get("document_type", "Unknown")
                st.metric(label="Detected Document Type", value=doc_type.upper())
                
                st.markdown("### Extracted Entities")
                for key, value in final_data.items():
                    if key not in ["document_type", "corrected_text"] and value is not None:
                        # Clean up the key names for display
                        display_name = key.replace("_", " ").title()
                        st.text_input(label=display_name, value=str(value), disabled=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                # Download button for the JSON
                json_string = json.dumps(final_data, indent=2)
                st.download_button(
                    label="⬇️ Download JSON",
                    file_name="extracted_data.json",
                    mime="application/json",
                    data=json_string,
                    use_container_width=True
                )
                        
            with tab2:
                st.markdown("### Raw OCR Output")
                st.text_area("What the engine 'saw'", raw_text, height=150, disabled=True)
                st.markdown("### LLM Corrected Text")
                st.text_area("Grammatically fixed text", final_data.get("corrected_text", "Not available."), height=150, disabled=True)
                
            with tab3:
                if validation_errors:
                    st.error("⚠️ Data Validation Warnings Found")
                    st.warning(validation_errors)
                else:
                    st.success("✅ All data perfectly matches Indian business formatting rules.")
                    
            with tab4:
                st.markdown("### Query the Document")
                st.info("Ask the AI a question about this document.")
                user_q = st.text_input("Your Question:")
                if st.button("Ask") and user_q:
                    with st.spinner("AI is thinking..."):
                        answer = llm_engine.answer_question(user_q, raw_text)
                        st.success(answer)
