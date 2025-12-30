"""MongoDB connection and database instance"""

from pymongo import MongoClient
from config import Config

# MongoDB client and database
_client = None
_db = None


def get_db():
    """Get the MongoDB database instance"""
    global _client, _db
    
    if _db is None:
        uri = Config.MONGO_URI
        
        # Ensure tlsAllowInvalidCertificates is in the URI for Atlas
        if "mongodb+srv://" in uri:
            # Add SSL bypass to URI if not already present
            if "?" in uri:
                if "tlsAllowInvalidCertificates" not in uri:
                    uri = uri + "&tlsAllowInvalidCertificates=true"
            else:
                uri = uri + "?tlsAllowInvalidCertificates=true"
        
        _client = MongoClient(uri)
        
        # Extract database name from URI, default to 'recruiter_ai'
        if "/" in uri:
            # Get the part after the last / and before any ?
            path_part = uri.split("@")[-1] if "@" in uri else uri
            if "/" in path_part:
                db_part = path_part.split("/")[-1].split("?")[0]
                db_name = db_part if db_part else "recruiter_ai"
            else:
                db_name = "recruiter_ai"
        else:
            db_name = "recruiter_ai"
        
        _db = _client[db_name]
    
    return _db


def close_db():
    """Close the MongoDB connection"""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None


def reset_connection():
    """Force reset the connection (useful for testing)"""
    close_db()
    return get_db()
