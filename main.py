#!/usr/bin/env python3
"""
Enhanced Poker Strategy & Analytics Platform
Main application entry point
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = ['flask', 'numpy']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ Required package '{package}' found")
        except ImportError:
            missing.append(package)
            logger.error(f"✗ Required package '{package}' missing")
    
    if missing:
        logger.error(f"Missing required packages: {', '.join(missing)}")
        logger.error("Please install with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main entry point"""
    print(" Enhanced Poker Strategy & Analytics Platform")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n Setup incomplete. Please install missing dependencies.")
        sys.exit(1)
    
    # Import and run the Flask app
    try:
        from app import app
        host = os.environ.get('HOST', '127.0.0.1')
        port = int(os.environ.get('PORT', 5000))
        debug = os.environ.get('DEBUG', 'True').lower() == 'true'
        
        print(f"\n Starting server on http://{host}:{port}")
        print("Press Ctrl+C to stop the server")
        print("=" * 50)
        
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()