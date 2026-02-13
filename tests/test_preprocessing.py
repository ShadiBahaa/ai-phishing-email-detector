"""
Unit tests for the phishing email detector.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.build_features import clean_text


class TestCleanText:
    """Tests for the clean_text function."""
    
    def test_lowercase_conversion(self):
        """Test that text is converted to lowercase."""
        result = clean_text("HELLO WORLD")
        assert result == "hello world"
    
    def test_url_replacement(self):
        """Test that URLs are replaced with 'URL' token."""
        result = clean_text("Visit http://example.com for more info")
        assert "URL" in result
        assert "http" not in result
    
    def test_email_replacement(self):
        """Test that email addresses are replaced with 'EMAIL' token."""
        result = clean_text("Contact us at test@example.com")
        assert "EMAIL" in result
        assert "@" not in result
    
    def test_html_tag_removal(self):
        """Test that HTML tags are removed."""
        result = clean_text("<p>Hello <b>World</b></p>")
        assert "<" not in result
        assert ">" not in result
    
    def test_special_characters_removal(self):
        """Test that special characters are removed."""
        result = clean_text("Hello! How are you? #test")
        assert "!" not in result
        assert "?" not in result
        assert "#" not in result
    
    def test_extra_whitespace_removal(self):
        """Test that extra whitespace is normalized."""
        result = clean_text("Hello    World")
        assert result == "hello world"
    
    def test_empty_string(self):
        """Test handling of empty strings."""
        result = clean_text("")
        assert result == ""
    
    def test_none_input(self):
        """Test handling of None input."""
        result = clean_text(None)
        assert result == ""
    
    def test_combined_preprocessing(self):
        """Test combined preprocessing steps."""
        text = "URGENT! Visit http://phish.com or email admin@evil.com NOW!!!"
        result = clean_text(text)
        
        assert result == "urgent visit URL or email EMAIL now"


class TestModelPrediction:
    """Tests for model prediction functionality."""
    
    @pytest.fixture
    def sample_phishing_text(self):
        """Sample phishing email text."""
        return """
        URGENT: Your account has been compromised!
        Click here immediately to verify: http://suspicious-link.com
        Enter your password to confirm your identity.
        """
    
    @pytest.fixture
    def sample_legitimate_text(self):
        """Sample legitimate email text."""
        return """
        Hi Team,
        Here are the meeting notes from yesterday.
        Please review and let me know if you have questions.
        Best regards
        """
    
    def test_phishing_text_preprocessing(self, sample_phishing_text):
        """Test preprocessing of phishing email."""
        result = clean_text(sample_phishing_text)
        
        # Should contain suspicious keywords
        assert "urgent" in result
        assert "compromised" in result
        assert "verify" in result
        
        # URL should be replaced
        assert "URL" in result
    
    def test_legitimate_text_preprocessing(self, sample_legitimate_text):
        """Test preprocessing of legitimate email."""
        result = clean_text(sample_legitimate_text)
        
        # Should contain normal business keywords
        assert "meeting" in result
        assert "notes" in result
        assert "review" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
