"""
RecruiterAI - Flask Application
===============================
Main entry point for the Flask backend API.
"""

from flask import Flask
from flask_cors import CORS

from config import Config


def create_app(config_class=Config):
    """Application factory pattern"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize CORS - allow all origins for development
    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
    
    # Register blueprints
    from routes.api import api_bp
    from routes.health import health_bp
    
    app.register_blueprint(health_bp)
    app.register_blueprint(api_bp, url_prefix="/api")
    
    # Test MongoDB connection on startup
    try:
        from extensions import get_db
        db = get_db()
        db.command("ping")
        print("✅ Connected to MongoDB")
    except Exception as e:
        print(f"⚠️  MongoDB connection warning: {e}")
    
    return app


# Create the app instance
app = create_app()

if __name__ == "__main__":
    # Validate configuration
    errors = Config.validate()
    if errors:
        print("⚠️  Configuration warnings:")
        for error in errors:
            print(f"   - {error}")
        print("\nSome features may not work without proper configuration.")
        print("See .env.example for required environment variables.\n")
    
    print("🚀 Starting RecruiterAI Backend...")
    print(f"   Available LLM providers: {Config.get_available_llm_providers()}")
    app.run(debug=Config.DEBUG, port=5000)
