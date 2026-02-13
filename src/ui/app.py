"""
Streamlit Web Application
=========================
Interactive web interface for phishing email detection.

Usage:
    streamlit run src/ui/app.py
"""

import streamlit as st
import joblib
import re
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Define paths
MODEL_PATH = Path("src/models/phishing_model.joblib")
VECTORIZER_PATH = Path("src/models/tfidf_vectorizer.joblib")


@st.cache_resource
def load_model():
    """Load the trained model and vectorizer."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def clean_text(text: str) -> str:
    """Clean and preprocess email text."""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs with token
    text = re.sub(r"http\S+|www\.\S+", "URL", text)
    
    # Replace email addresses with token
    text = re.sub(r"\S+@\S+", "EMAIL", text)
    
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    
    # Remove non-alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def get_risk_level(prob: float) -> tuple:
    """Determine risk level based on probability."""
    if prob >= 0.8:
        return "🔴 HIGH RISK", "red"
    elif prob >= 0.5:
        return "🟠 MEDIUM RISK", "orange"
    elif prob >= 0.3:
        return "🟡 LOW RISK", "yellow"
    else:
        return "🟢 SAFE", "green"


def main():
    # Header
    st.title("🛡️ AI-Powered Phishing Email Detector")
    st.markdown("""
    Analyze emails for potential phishing threats using machine learning.
    Simply paste the email content below and click **Analyze**.
    """)
    
    # Load model
    model, vectorizer = load_model()
    
    if model is None:
        st.error("""
        ⚠️ **Model not found!**
        
        Please train the model first by running:
        ```bash
        python src/data/make_dataset.py
        python src/features/build_features.py
        python src/models/train.py
        ```
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This tool uses a **Logistic Regression** model trained on 
        email text features (TF-IDF) to detect phishing attempts.
        
        **Features analyzed:**
        - Text patterns and keywords
        - URL presence
        - Email address mentions
        - Suspicious phrases
        
        **Risk Levels:**
        - 🟢 Safe: < 30%
        - 🟡 Low Risk: 30-50%
        - 🟠 Medium Risk: 50-80%
        - 🔴 High Risk: > 80%
        """)
        
        st.divider()
        st.header("⚠️ Disclaimer")
        st.markdown("""
        This tool is for **educational purposes only**.
        Always verify suspicious emails through official channels.
        """)
    
    # Main input area
    st.subheader("📧 Email Content")
    email_text = st.text_area(
        "Paste the email content here:",
        height=200,
        placeholder="Enter the email text you want to analyze..."
    )
    
    # Example emails
    with st.expander("📝 Try example emails"):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Load Phishing Example"):
                st.session_state.example_email = """
                URGENT: Your account has been compromised!
                
                Dear Customer,
                
                We have detected suspicious activity on your account. 
                Your account will be suspended within 24 hours unless you verify your information.
                
                Click here immediately to verify: http://suspicious-link.com/verify
                
                Enter your password and credit card details to confirm your identity.
                
                Act now or lose access to your account forever!
                
                Security Team
                """
        
        with col2:
            if st.button("Load Legitimate Example"):
                st.session_state.example_email = """
                Hi Team,
                
                I wanted to follow up on our meeting yesterday regarding the Q4 project timeline.
                
                As discussed, here are the key action items:
                1. Review the budget proposal by Friday
                2. Schedule the client presentation for next week
                3. Update the project documentation
                
                Please let me know if you have any questions.
                
                Best regards,
                John
                """
    
    # Use example if loaded
    if "example_email" in st.session_state:
        email_text = st.session_state.example_email
        st.text_area("Email content:", value=email_text, height=200, disabled=True)
        del st.session_state.example_email
    
    # Analyze button
    if st.button("🔍 Analyze Email", type="primary", use_container_width=True):
        if not email_text.strip():
            st.warning("Please enter some email content to analyze.")
            return
        
        with st.spinner("Analyzing email..."):
            # Preprocess
            cleaned = clean_text(email_text)
            
            # Vectorize
            features = vectorizer.transform([cleaned])
            
            # Predict
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
            
            # Display results
            st.divider()
            st.subheader("📊 Analysis Results")
            
            # Risk score
            risk_level, color = get_risk_level(probability)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risk Score", f"{probability:.1%}")
            
            with col2:
                st.metric("Classification", "PHISHING" if prediction == 1 else "LEGITIMATE")
            
            with col3:
                st.metric("Risk Level", risk_level.split(" ")[0])
            
            # Alert box
            if prediction == 1:
                st.error(f"""
                🚨 **PHISHING DETECTED**
                
                {risk_level}
                
                This email shows characteristics commonly associated with phishing attempts.
                **Do not click any links or provide personal information.**
                """)
            else:
                st.success(f"""
                ✅ **LEGITIMATE EMAIL**
                
                {risk_level}
                
                This email appears to be legitimate based on our analysis.
                However, always exercise caution with unexpected messages.
                """)
            
            # Details
            with st.expander("🔬 Analysis Details"):
                st.markdown(f"""
                **Original length:** {len(email_text)} characters
                
                **Cleaned length:** {len(cleaned)} characters
                
                **Features extracted:** {features.shape[1]} TF-IDF features
                
                **Prediction probability:**
                - Legitimate: {1 - probability:.2%}
                - Phishing: {probability:.2%}
                """)


if __name__ == "__main__":
    main()
