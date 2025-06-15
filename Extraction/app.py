# --- Force python to use pysqlite3 based on chromadb docs ---
# This override MUST happen before any other imports that might import sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- End override ---

# app.py
import streamlit as st
import os
import time
from loguru import logger
import json
import pandas as pd
import re
import asyncio
import subprocess
from typing import List
from langchain.docstore.document import Document
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# --- Install Playwright browsers needed by crawl4ai --- 
def install_playwright_browsers():
    logger.info("Checking and installing Playwright browsers if needed...")
    try:
        process = subprocess.run([sys.executable, "-m", "playwright", "install"], 
                               capture_output=True, text=True, check=False)
        if process.returncode == 0:
            logger.success("Playwright browsers installed successfully (or already exist).")
        else:
            logger.error(f"Playwright browser install command failed with code {process.returncode}.")
            logger.error(f"stdout: {process.stdout}")
            logger.error(f"stderr: {process.stderr}")
    except FileNotFoundError:
        logger.error("Could not find 'playwright' command. Is playwright installed correctly?")
        st.error("Playwright not found. Please ensure 'playwright' is in requirements.txt")
    except Exception as e:
        logger.error(f"An error occurred during Playwright browser installation: {e}", exc_info=True)
        st.warning(f"An error occurred installing Playwright browsers: {e}. Web scraping may fail.")

install_playwright_browsers()

# Import project modules
import config
from pdf_processor import process_uploaded_pdfs, process_pdfs_in_background
from vector_store import (
    get_embedding_function,
    setup_vector_store,
    load_existing_vector_store
)
from llm_interface import (
    initialize_llm,
    create_pdf_extraction_chain,
    create_web_extraction_chain,
    _invoke_chain_and_process,
    scrape_website_table_html
)
from Extraction.extraction_prompts import (
    # Material Properties
    MATERIAL_PROMPT,
    MATERIAL_NAME_PROMPT,
    # Physical / Mechanical Attributes
    PULL_TO_SEAT_PROMPT,
    GENDER_PROMPT,
    HEIGHT_MM_PROMPT,
    LENGTH_MM_PROMPT,
    WIDTH_MM_PROMPT,
    NUMBER_OF_CAVITIES_PROMPT,
    NUMBER_OF_ROWS_PROMPT,
    MECHANICAL_CODING_PROMPT,
    COLOUR_PROMPT,
    COLOUR_CODING_PROMPT,
    # Sealing & Environmental
    WORKING_TEMPERATURE_PROMPT,
    HOUSING_SEAL_PROMPT,
    WIRE_SEAL_PROMPT,
    SEALING_PROMPT,
    SEALING_CLASS_PROMPT,
    # Terminals & Connections
    CONTACT_SYSTEMS_PROMPT,
    TERMINAL_POSITION_ASSURANCE_PROMPT,
    CONNECTOR_POSITION_ASSURANCE_PROMPT,
    CLOSED_CAVITIES_PROMPT,
    # Assembly & Type
    PRE_ASSEMBLED_PROMPT,
    CONNECTOR_TYPE_PROMPT,
    SET_KIT_PROMPT,
    # Specialized Attributes
    HV_QUALIFIED_PROMPT
)
from Extraction.extraction_prompts_web import (
    # Material Properties
    MATERIAL_FILLING_WEB_PROMPT,
    MATERIAL_NAME_WEB_PROMPT,
    # Physical / Mechanical Attributes
    PULL_TO_SEAT_WEB_PROMPT,
    GENDER_WEB_PROMPT,
    HEIGHT_MM_WEB_PROMPT,
    LENGTH_MM_WEB_PROMPT,
    WIDTH_MM_WEB_PROMPT,
    NUMBER_OF_CAVITIES_WEB_PROMPT,
    NUMBER_OF_ROWS_WEB_PROMPT,
    MECHANICAL_CODING_WEB_PROMPT,
    COLOUR_WEB_PROMPT,
    COLOUR_CODING_WEB_PROMPT,
    # Sealing & Environmental
    MAX_WORKING_TEMPERATURE_WEB_PROMPT,
    MIN_WORKING_TEMPERATURE_WEB_PROMPT,
    HOUSING_SEAL_WEB_PROMPT,
    WIRE_SEAL_WEB_PROMPT,
    SEALING_WEB_PROMPT,
    SEALING_CLASS_WEB_PROMPT,
    # Terminals & Connections
    CONTACT_SYSTEMS_WEB_PROMPT,
    TERMINAL_POSITION_ASSURANCE_WEB_PROMPT,
    CONNECTOR_POSITION_ASSURANCE_WEB_PROMPT,
    CLOSED_CAVITIES_WEB_PROMPT,
    # Assembly & Type
    PRE_ASSEMBLED_WEB_PROMPT,
    CONNECTOR_TYPE_WEB_PROMPT,
    SET_KIT_WEB_PROMPT,
    # Specialized Attributes
    HV_QUALIFIED_WEB_PROMPT
)

def initialize_session_state():
    """Initialize session state variables"""
    state_vars = {
        'retriever': None,
        'pdf_chain': None,
        'web_chain': None,
        'processed_files': [],
        'evaluation_results': [],
        'evaluation_metrics': None,
        'extraction_performed': False,
        'scraped_table_html_cache': None,
        'current_part_number_scraped': None,
        'pdf_processing_task': None,
        'pdf_processing_complete': False,
        'pdf_processing_results': None
    }
    
    for var, default_value in state_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

@st.cache_resource
def initialize_embeddings():
    """Initialize embeddings function"""
    embeddings = get_embedding_function()
    return embeddings

@st.cache_resource
def initialize_llm_cached():
    """Initialize LLM function"""
    llm_instance = initialize_llm()
    return llm_instance

def initialize_resources():
    """Initialize or load required resources."""
    try:
        # Initialize embedding function
        embedding_function = get_embedding_function()
        if not embedding_function:
            logger.error("Failed to initialize embedding function")
            return False
            
        # Initialize LLM
        if not initialize_llm():
            logger.error("Failed to initialize LLM")
            return False
            
        # Try to load existing vector store
        vector_store = load_existing_vector_store(embedding_function)
        if not vector_store:
            logger.warning("No existing vector store found, setting up new one...")
            # Create an empty vector store
            vector_store = setup_vector_store([], embedding_function)
            if not vector_store:
                logger.error("Failed to set up vector store")
                return False
                
        # Create extraction chains
        pdf_chain = create_pdf_extraction_chain(vector_store)
        web_chain = create_web_extraction_chain()
        
        if not pdf_chain or not web_chain:
            logger.error("Failed to create extraction chains")
            return False
            
        # Store resources in session state
        st.session_state.vector_store = vector_store
        st.session_state.pdf_chain = pdf_chain
        st.session_state.web_chain = web_chain
        
        return True
        
    except Exception as e:
        logger.error(f"An error has been caught in function 'initialize_resources': {str(e)}")
        return False

def convert_df_to_csv(df):
    """Convert DataFrame to CSV format"""
    return df.to_csv(index=False).encode('utf-8')

def process_documents(part_number):
    """Process uploaded documents"""
    if not st.session_state.uploaded_files:
        st.warning("Please upload at least one PDF file")
        return
    
    with st.spinner("Processing documents..."):
        try:
            # Process the documents
            process_uploaded_pdfs(st.session_state.uploaded_files)
            st.success("Documents processed successfully!")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

def main():
    """Main function to run the Streamlit app"""
    # Initialize session state
    initialize_session_state()
    
    # Set up the page
    st.title("📄 PDF Auto-Extraction with Groq")
    st.markdown("""
    This tool automatically extracts attributes from PDF documents and compares them with web data.
    Upload your PDF files and enter a part number to get started.
    """)
    
    # Check for API key in Streamlit secrets first, then in .env
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        st.error("""
        ⚠️ GROQ API key not found. Please set it in Streamlit secrets or in the .env file.
        
        To set it in Streamlit secrets:
        1. Go to your Streamlit Cloud dashboard
        2. Select your app
        3. Go to Settings > Secrets
        4. Add your GROQ API key:
        ```
        GROQ_API_KEY = "your-api-key-here"
        ```
        """)
        return
    
    # Initialize resources
    if not initialize_resources():
        st.error("Failed to initialize resources. Please check the logs for details.")
        return
    
    # Main area for document upload and processing
    st.markdown("### 📤 Upload Documents")
    st.markdown("Upload your PDF files here. You can upload multiple files at once.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    # Initialize uploaded_files in session state if not present
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    # Check for new files
    if uploaded_files:
        current_files = {f.name for f in st.session_state.uploaded_files}
        new_files = [f for f in uploaded_files if f.name not in current_files]
        
        if new_files:
            st.session_state.uploaded_files.extend(new_files)
            st.success(f"Added {len(new_files)} new file(s)")
    
    # Display currently uploaded files
    if st.session_state.uploaded_files:
        st.markdown("### 📚 Uploaded Files")
        for file in st.session_state.uploaded_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📄 {file.name}")
            with col2:
                if st.button("🗑️ Remove", key=f"remove_{file.name}"):
                    st.session_state.uploaded_files.remove(file)
                    st.rerun()
    
    # Part number input
    st.markdown("### 🔍 Part Number")
    part_number = st.text_input("Enter the part number to search for", key="part_number")
    
    # Process button
    if st.button("Process Documents", type="primary"):
        if not st.session_state.uploaded_files:
            st.warning("Please upload at least one PDF file")
        elif not part_number:
            st.warning("Please enter a part number")
        else:
            with st.spinner("Processing documents..."):
                try:
                    # Process the documents
                    process_documents(part_number)
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
    
    # Display results if available
    if st.session_state.get('evaluation_results'):
        st.markdown("### 📊 Results")
        
        # Prepare data for export
        results_df = pd.DataFrame(st.session_state.evaluation_results)
        summary_dict = st.session_state.evaluation_metrics
        
        # Create download buttons
        csv = convert_df_to_csv(results_df)
        st.download_button(
            label="Download Detailed Results (CSV)",
            data=csv,
            file_name="extraction_results.csv",
            mime="text/csv"
        )
        
        st.download_button(
            label="Download Summary Metrics (JSON)",
            data=json.dumps(summary_dict, indent=2),
            file_name="extraction_metrics.json",
            mime="application/json"
        )
        
        # Display results
        st.dataframe(results_df)
        
        # Display metrics
        st.markdown("### 📈 Metrics")
        st.json(summary_dict)
    elif st.session_state.get('extraction_performed'):
        st.warning("Extraction ran but yielded no valid results. Please check the logs or raw outputs.")

# Only run the main function if this file is executed directly
if __name__ == "__main__":
    main()