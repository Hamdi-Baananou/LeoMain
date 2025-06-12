import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="LEOPARTS",
    page_icon="🦁",
    layout="wide"
)

import sys
import os

# Add both project directories to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "Chatbot"))
sys.path.append(os.path.join(os.path.dirname(__file__), "Extraction"))

# Import the main functions from both apps
from Chatbot.chatbot import main as chatbot_main
from Extraction.app import main as extraction_main

def main():
    # Clear session state on page reload
    if 'page_reloaded' not in st.session_state:
        st.session_state.clear()
        st.session_state.page_reloaded = True

    # Initialize session state for navigation
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'home'

    # Navigation sidebar with improved styling
    with st.sidebar:
        st.markdown("<h1 style='text-align: center;'>LEOPARTS</h1>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Navigation buttons with improved styling
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.current_view = 'home'
        if st.button("💬 Ask Questions", use_container_width=True):
            st.session_state.current_view = 'chatbot'
        if st.button("📄 Upload Documents", use_container_width=True):
            st.session_state.current_view = 'extraction'

    # Main content area
    if st.session_state.current_view == 'home':
        # Center the content with improved styling
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

    elif st.session_state.current_view == 'chatbot':
        chatbot_main()

    elif st.session_state.current_view == 'extraction':
        extraction_main()

if __name__ == "__main__":
    main() 