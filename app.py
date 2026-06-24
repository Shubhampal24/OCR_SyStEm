import streamlit as st
from PIL import Image
import json
import os

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
uploaded_file = st.file_uploader("Drop your document here (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Use columns for a sleek dashboard layout (Image on left, Extraction on right)
    main_col1, main_col2 = st.columns([1, 1.5], gap="large")
    
    with main_col1:
        st.subheader("Document Preview")
        pil_image = Image.open(uploaded_file)
        # Display image with clean styling
        st.image(pil_image, use_column_width=True, clamp=True)
        
    with main_col2:
        st.subheader("Extraction Pipeline")
        
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
            status_text.info("🧠 Analyzing context with LLM...")
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
            
            progress_bar.progress(100)
            status_text.success("✅ Extraction Complete.")
            
            st.markdown("---")
            
            # Show Results in modern Tabs instead of a long scrollable page
            tab1, tab2, tab3 = st.tabs(["📊 Structured Data", "📝 Raw OCR Output", "🛡️ Validation Log"])
            
            with tab1:
                # Format data beautifully using metrics and text inputs instead of raw JSON string
                doc_type = final_data.get("document_type", "Unknown")
                st.metric(label="Detected Document Type", value=doc_type.upper())
                
                st.markdown("### Extracted Entities")
                for key, value in final_data.items():
                    if key != "document_type" and value is not None:
                        # Clean up the key names for display (e.g. "total_amount" -> "Total Amount")
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
                st.text_area("What the engine 'saw'", raw_text, height=300, disabled=True)
                
            with tab3:
                if validation_errors:
                    st.error("⚠️ Data Validation Warnings Found")
                    st.warning(validation_errors)
                else:
                    st.success("✅ All data perfectly matches Indian business formatting rules.")
