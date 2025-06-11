import streamlit as st
import sys
import os

# Add both project directories to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "Chatbot"))
sys.path.append(os.path.join(os.path.dirname(__file__), "Extraction"))

# Import the main functions from both apps
from Chatbot.chatbot import main as chatbot_main
from Extraction.app import main as extraction_main

def main():
    # Set page config only once at the start
    st.set_page_config(
        page_title="LEOPARTS",
        page_icon="🦁",
        layout="wide"
    )

    # Initialize session state for navigation
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'home'

    # Navigation sidebar
    with st.sidebar:
        st.title("Navigation")
        if st.button("🏠 Home"):
            st.session_state.current_view = 'home'
        if st.button("💬 Chat with Leoparts"):
            st.session_state.current_view = 'chatbot'
        if st.button("📄 Extract a new Part"):
            st.session_state.current_view = 'extraction'

    # Main content area
    if st.session_state.current_view == 'home':
        # Center the content
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("<h1 style='text-align: center;'>LEOPARTS</h1>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>Welcome!</h2>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'>Choose a Tool</h3>", unsafe_allow_html=True)
            
            # Add some spacing
            st.markdown("<br>" * 2, unsafe_allow_html=True)
            
            # Create two columns for the buttons
            col_left, col_right = st.columns(2)
            
            with col_left:
                if st.button("💬 Chat with Leoparts", use_container_width=True):
                    st.session_state.current_view = 'chatbot'
                    st.rerun()
            
            with col_right:
                if st.button("📄 Extract a new Part", use_container_width=True):
                    st.session_state.current_view = 'extraction'
                    st.rerun()

    elif st.session_state.current_view == 'chatbot':
        chatbot_main()

    elif st.session_state.current_view == 'extraction':
        extraction_main()

if __name__ == "__main__":
    main() 