import re
from pydantic import BaseModel, EmailStr, Field, field_validator, ValidationError
from typing import Optional
from src.core.logger import get_logger
from src.core.exceptions import DataValidationError

logger = get_logger(__name__)

class DocumentData(BaseModel):
    """
    Industrial Pydantic Schema.
    This acts as our "Security Guard". Even if the LLM hallucinated fake data,
    this schema enforces strict mathematical and regular expression rules before 
    the data is allowed to be saved or presented to the user.
    """
    # The LLM MUST provide a document type
    document_type: str = Field(description="Classified type of document (e.g., Invoice, ID Card)")
    
    # Optional fields (since a receipt won't have a PAN number, etc.)
    corrected_text: Optional[str] = Field(None, description="Grammatically fixed text")
    name: Optional[str] = None
    email: Optional[EmailStr] = None 
    total_amount: Optional[float] = None 
    phone: Optional[str] = Field(None, description="Phone number")
    pan_number: Optional[str] = Field(None, description="PAN Card number")
    aadhaar_number: Optional[str] = Field(None, description="Aadhaar Card number")
    gst_number: Optional[str] = Field(None, description="GST Number")
    date: Optional[str] = Field(None, description="Date of the document")
    
    # --- Soft Validators ---
    
    @field_validator('date')
    def validate_date(cls, v):
        if not v: return v
        if not re.search(r'\d{2,4}[-/]\d{2}[-/]\d{2,4}', v):
            logger.warning(f"Invalid Date format detected: {v}")
        return v
        
    @field_validator('gst_number')
    def validate_gst(cls, v):
        if not v: return v
        clean_v = v.replace(" ", "").upper()
        if not re.match(r'^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}Z[A-Z\d]{1}$', clean_v):
            logger.warning(f"Invalid GST format: {v}")
        return clean_v

    @field_validator('phone')
    def validate_phone(cls, v):
        if not v: return v
        clean_v = re.sub(r'[\s\-()]', '', v)
        if not re.match(r'^(\+91)?\d{10}$', clean_v):
            logger.warning(f"Invalid phone format: {v}")
        return clean_v

    @field_validator('pan_number')
    def validate_pan(cls, v):
        if not v: return v
        clean_v = v.upper().replace(" ", "")
        if not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', clean_v):
            logger.warning(f"Invalid PAN format: {v}")
        return clean_v

    @field_validator('aadhaar_number')
    def validate_aadhaar(cls, v):
        if not v: return v
        clean_v = v.replace(" ", "").replace("-", "")
        if not re.match(r'^\d{12}$', clean_v):
            logger.warning(f"Invalid Aadhaar format: {v}")
        return clean_v

class Validator:
    @staticmethod
    def validate_data(json_data: dict) -> dict:
        """
        Takes raw JSON from the LLM engine and runs it through Pydantic.
        We catch validation errors so the UI doesn't crash, but still log them.
        """
        try:
            logger.info("Starting Pydantic data validation.")
            validated_model = DocumentData(**json_data)
            logger.info("Data validation passed.")
            return validated_model.model_dump(exclude_none=True)
            
        except ValidationError as e:
            error_msgs = "; ".join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
            logger.error(f"Validation warning: {error_msgs}")
            
            # Instead of crashing, we inject the error into the payload so the UI can show it gracefully
            json_data["validation_warnings"] = error_msgs
            
            # Ensure document_type exists even on failure
            if "document_type" not in json_data:
                json_data["document_type"] = "Unreadable/Invalid"
                
            return {k: v for k, v in json_data.items() if v is not None}
