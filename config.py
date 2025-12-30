# Configuration Management
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration"""
    
    # Flask
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    
    # MongoDB
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/recruiter_ai")
    
    # LinkedIn Credentials
    LINKEDIN_EMAIL = os.getenv("LINKEDIN_EMAIL")
    LINKEDIN_PASSWORD = os.getenv("LINKEDIN_PASSWORD")
    
    # LLM Providers
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Default LLM Provider: 'openai' or 'gemini'
    DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    
    @classmethod
    def get_available_llm_providers(cls):
        """Return list of available LLM providers based on configured API keys"""
        providers = []
        if cls.OPENAI_API_KEY:
            providers.append("openai")
        if cls.GOOGLE_API_KEY:
            providers.append("gemini")
        return providers
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []
        
        if not cls.LINKEDIN_EMAIL or not cls.LINKEDIN_PASSWORD:
            errors.append("LinkedIn credentials (LINKEDIN_EMAIL, LINKEDIN_PASSWORD) are required")
        
        if not cls.OPENAI_API_KEY and not cls.GOOGLE_API_KEY:
            errors.append("At least one LLM API key (OPENAI_API_KEY or GOOGLE_API_KEY) is required")
        
        return errors
