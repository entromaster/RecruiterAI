"""Health check routes"""

from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "RecruiterAI Backend",
        "version": "1.0.0"
    })


@health_bp.route("/", methods=["GET"])
def root():
    """Root endpoint"""
    return jsonify({
        "message": "Welcome to RecruiterAI API",
        "docs": "/api",
        "health": "/health"
    })
