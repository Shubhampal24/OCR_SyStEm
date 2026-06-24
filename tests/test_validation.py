import pytest
from src.validation import Validator
from src.core.exceptions import DataValidationError

def test_valid_document_data():
    """Test that a perfectly valid document passes all rules and cleans data."""
    data = {
        "document_type": "Invoice",
        "name": "John Doe",
        "email": "test@example.com",
        "phone": "+919876543210",
        "total_amount": "100.50",
        "pan_number": "ABCDE1234F",
        "aadhaar_number": "1234 5678 9012"
    }
    result = Validator.validate_data(data)
    
    assert result["document_type"] == "Invoice"
    assert result["email"] == "test@example.com"
    # Ensure amount was coerced from string to float
    assert result["total_amount"] == 100.5
    # Ensure phone was accepted
    assert result["phone"] == "+919876543210"
    assert result["pan_number"] == "ABCDE1234F"
    # Ensure Aadhaar spaces were stripped
    assert result["aadhaar_number"] == "123456789012"

def test_invalid_email_raises_error():
    """Test that fake emails get blocked."""
    data = {
        "document_type": "Invoice",
        "email": "not-an-email"
    }
    with pytest.raises(DataValidationError) as exc_info:
        Validator.validate_data(data)
    assert "email" in str(exc_info.value).lower()

def test_invalid_pan_format_raises_error():
    """Test that PAN card logic catches bad formats (e.g., number where letter should be)."""
    data = {
        "document_type": "ID Card",
        "pan_number": "1BCDE1234F" # Starts with 1 instead of letter
    }
    with pytest.raises(DataValidationError) as exc_info:
        Validator.validate_data(data)
    assert "PAN Card" in str(exc_info.value)

def test_aadhaar_cleans_dashes_and_catches_invalid_length():
    """Test that Aadhaar cleans up dashes but blocks incorrect lengths."""
    # Valid
    valid_data = {"document_type": "ID Card", "aadhaar_number": "1234-5678-9012"}
    result = Validator.validate_data(valid_data)
    assert result["aadhaar_number"] == "123456789012"
    
    # Invalid length
    invalid_data = {"document_type": "ID Card", "aadhaar_number": "123456"}
    with pytest.raises(DataValidationError):
        Validator.validate_data(invalid_data)
        
def test_missing_document_type_raises_error():
    """Test that missing required fields blocks the pipeline."""
    data = {"name": "John"} # Missing document_type
    with pytest.raises(DataValidationError):
        Validator.validate_data(data)
