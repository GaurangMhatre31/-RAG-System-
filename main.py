#!/usr/bin/env python3
"""
Streamlit RAG App - Replit Deployment Entry Point
"""

import os
import sys
import subprocess

def main():
    # Set environment variables for Replit deployment
    os.environ.update({
        'STREAMLIT_SERVER_PORT': '8080',
        'STREAMLIT_SERVER_ADDRESS': '0.0.0.0', 
        'STREAMLIT_SERVER_HEADLESS': 'true',
        'STREAMLIT_SERVER_ENABLE_CORS': 'false',
        'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false',
        'PORT': '8080'
    })
    
    print("🚀 Starting Streamlit RAG App on Replit...")
    print("📡 Server will be available on port 8080")
    
    # Start Streamlit with proper configuration
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port=8080",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--browser.gatherUsageStats=false"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Shutting down Streamlit app...")
        sys.exit(0)

if __name__ == "__main__":
    main()
