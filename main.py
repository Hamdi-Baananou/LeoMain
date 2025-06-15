import streamlit as st
import sys
import os
from pathlib import Path
import subprocess
from playwright.sync_api import sync_playwright

def ensure_playwright_browser():
    chromium_path = os.path.expanduser("~/.cache/ms-playwright/chromium-1169/chrome-linux/chrome")
    if not os.path.exists(chromium_path):
        try:
            print("Installing Playwright browsers...")
            subprocess.run(["playwright", "install", "chromium"], check=True)
        except Exception as e:
            print(f"Playwright browser installation failed: {e}")

# Ensure Playwright browser is installed
ensure_playwright_browser()

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="LEOPARTS",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add both project directories to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "Chatbot"))
sys.path.append(str(current_dir / "Extraction"))

# Import the main functions from both apps
from Chatbot.chatbot import main as chatbot_main
from Extraction.app import main as extraction_main

def initialize_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        # Store all existing state variables
        existing_state = {
            'uploaded_files': st.session_state.get('uploaded_files', []),
            'processed_files': st.session_state.get('processed_files', []),
            'retriever': st.session_state.get('retriever'),
            'pdf_chain': st.session_state.get('pdf_chain'),
            'web_chain': st.session_state.get('web_chain'),
            'evaluation_results': st.session_state.get('evaluation_results', []),
            'evaluation_metrics': st.session_state.get('evaluation_metrics'),
            'extraction_performed': st.session_state.get('extraction_performed', False),
            'scraped_table_html_cache': st.session_state.get('scraped_table_html_cache'),
            'current_part_number_scraped': st.session_state.get('current_part_number_scraped'),
            'pdf_processing_task': st.session_state.get('pdf_processing_task'),
            'pdf_processing_complete': st.session_state.get('pdf_processing_complete', False),
            'pdf_processing_results': st.session_state.get('pdf_processing_results'),
            'chatbot_messages': st.session_state.get('chatbot_messages', [])
        }
        
        # Clear session state
        st.session_state.clear()
        
        # Restore all state variables
        for key, value in existing_state.items():
            st.session_state[key] = value
        
        # Initialize any missing state variables
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
        if 'current_view' not in st.session_state:
            st.session_state.current_view = 'home'

def render_sidebar():
    """Render the navigation sidebar"""
    with st.sidebar:
        st.markdown("<h1 style='text-align: center;'>LEOPARTS</h1>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Navigation buttons with improved styling
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.current_view = 'home'
            st.rerun()
        if st.button("💬 Ask Questions", use_container_width=True):
            st.session_state.current_view = 'chatbot'
            st.rerun()
        if st.button("📄 Upload Documents", use_container_width=True):
            st.session_state.current_view = 'extraction'
            st.rerun()

def render_home():
    """Render the home page"""
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>Welcome to LEOPARTS</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 1.2em;'>Your intelligent assistant for parts information</p>", unsafe_allow_html=True)
        
        # Add some spacing
        st.markdown("<br>" * 2, unsafe_allow_html=True)
        
        # Create two columns for the buttons with improved styling
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.markdown("### Ask Questions")
            st.markdown("Get instant answers about parts and specifications")
            if st.button("💬 Start Chat", use_container_width=True):
                st.session_state.current_view = 'chatbot'
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_right:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.markdown("### Upload Documents")
            st.markdown("Extract information from your documents")
            if st.button("📄 Upload Files", use_container_width=True):
                st.session_state.current_view = 'extraction'
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    if st.session_state.current_view == 'home':
        render_home()
    elif st.session_state.current_view == 'chatbot':
        chatbot_main()
    elif st.session_state.current_view == 'extraction':
        try:
            extraction_main()
        except Exception as e:
            st.error(f"Error in extraction module: {str(e)}")
            st.info("Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    main() 