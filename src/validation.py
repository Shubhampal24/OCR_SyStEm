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
    name: Optional[str] = None
    email: Optional[EmailStr] = None # Built-in robust email regex validation
    total_amount: Optional[float] = None # Will automatically cast string "100.50" to float
    phone: Optional[str] = Field(None, description="Phone number")
    pan_number: Optional[str] = Field(None, description="PAN Card number")
    aadhaar_number: Optional[str] = Field(None, description="Aadhaar Card number")
    gst_number: Optional[str] = Field(None, description="GST Number")
    date: Optional[str] = Field(None, description="Date of the document")
    
    # --- Pydantic Validators (The Security Guards) ---
    
    @field_validator('date')
    def validate_date(cls, v):
        if not v: return v
        # Simple regex to ensure it looks like a date (DD/MM/YYYY or YYYY-MM-DD or DD-MM-YYYY)
        if not re.search(r'\d{2,4}[-/]\d{2}[-/]\d{2,4}', v):
            raise ValueError(f"Invalid Date format: {v}. Must be DD/MM/YYYY or similar.")
        return v
        
    @field_validator('gst_number')
    def validate_gst(cls, v):
        if not v: return v
        clean_v = v.replace(" ", "").upper()
        # Indian GST format: 2 numbers + 10 PAN chars + 1 number/letter + Z + 1 number/letter
        if not re.match(r'^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}Z[A-Z\d]{1}$', clean_v):
            raise ValueError(f"Invalid GST Number format: {v}")
        return clean_v

    @field_validator('phone')
    def validate_phone(cls, v):
        if not v: return v
        # Clean up common OCR mistakes in phone numbers (dashes, spaces)
        clean_v = re.sub(r'[\s\-()]', '', v)
        # Indian phone number format validation
        if not re.match(r'^(\+91)?\d{10}$', clean_v):
            raise ValueError(f"Invalid phone number format: {v}")
        return clean_v

    @field_validator('pan_number')
    def validate_pan(cls, v):
        if not v: return v
        clean_v = v.upper().replace(" ", "")
        # Strict PAN rule: 5 Letters, 4 Numbers, 1 Letter
        if not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', clean_v):
            raise ValueError(f"Invalid PAN Card format: {v}")
        return clean_v

    @field_validator('aadhaar_number')
    def validate_aadhaar(cls, v):
        if not v: return v
        clean_v = v.replace(" ", "").replace("-", "")
        # Strict Aadhaar rule: Exactly 12 digits
        if not re.match(r'^\d{12}$', clean_v):
            raise ValueError(f"Invalid Aadhaar format: {v}")
        return clean_v

class Validator:
    @staticmethod
    def validate_data(json_data: dict) -> dict:
        """
        Takes raw JSON from the LLM engine and runs it through the strict Pydantic rules.
        """
        try:
            logger.info("Starting strict Pydantic data validation.")
            # This single line executes all regex checks, type coercions, and email validations.
            validated_model = DocumentData(**json_data)
            
            logger.info("Data validation passed securely.")
            # Return clean dictionary, removing any fields that were empty (None)
            return validated_model.model_dump(exclude_none=True)
            
        except ValidationError as e:
            # If Pydantic catches an error, we intercept it, log the exact field that failed,
            # and raise our custom industrial exception.
            error_msgs = "; ".join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
            logger.error(f"Validation failed: {error_msgs}")
            raise DataValidationError(f"Data failed strict validation rules -> {error_msgs}")
