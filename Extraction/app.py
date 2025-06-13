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
import json # Import the json library
import pandas as pd # Add pandas import
import re # Import the 're' module for regular expressions
import asyncio # Add asyncio import
from typing import List
from langchain.docstore.document import Document
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# --- Event Loop Setup ---
def get_or_create_eventloop():
    """Get the current event loop or create a new one."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

# Initialize event loop
loop = get_or_create_eventloop()

# Import project modules
import config
from pdf_processor import process_uploaded_pdfs, process_pdfs_in_background
from vector_store import (
    get_embedding_function,
    setup_vector_store,
    load_existing_vector_store
)
# Updated imports from llm_interface
from llm_interface import (
    initialize_llm,
    create_pdf_extraction_chain, # Use PDF chain func
    create_web_extraction_chain, # Use Web chain func
    _invoke_chain_and_process, # Use the helper directly
    scrape_website_table_html
)
# Import the prompts
from extraction_prompts import (
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
# Import the NEW web prompts
from extraction_prompts_web import (
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

async def process_web_urls(part_numbers: List[str]) -> List[Document]:
    """Process part numbers and return documents from web scraping."""
    web_docs = []
    for part_number in part_numbers:
        try:
            # Scrape the website table HTML using part number
            table_html = await scrape_website_table_html(part_number)
            if table_html:
                # Create a document from the scraped HTML
                doc = Document(
                    page_content=table_html,
                    metadata={
                        'source': f'web_{part_number}',
                        'type': 'web_scrape'
                    }
                )
                web_docs.append(doc)
                logger.info(f"Successfully scraped data for part number {part_number}")
            else:
                logger.warning(f"No data found for part number {part_number}")
        except Exception as e:
            logger.error(f"Error processing part number {part_number}: {e}")
    return web_docs

async def process_files(uploaded_files, part_numbers):
    """Process uploaded PDF files and part numbers."""
    try:
        # Initialize required components if not already done
        if 'retriever' not in st.session_state or st.session_state.retriever is None:
            embedding_function = get_embedding_function()
            if embedding_function is None:
                raise ValueError("Failed to initialize embedding function")
            
            # Process PDFs if any are uploaded
            pdf_docs = []
            if uploaded_files:
                temp_dir = os.path.join(os.getcwd(), "temp_pdf_files")
                pdf_docs = await process_uploaded_pdfs(uploaded_files, temp_dir)
            
            # Process part numbers if any are provided
            web_docs = []
            if part_numbers:
                web_docs = await process_web_urls(part_numbers)
            
            # Combine documents, prioritizing web docs if available
            all_docs = web_docs if web_docs else pdf_docs
            
            if all_docs:
                st.session_state.retriever = setup_vector_store(all_docs, embedding_function)
                if st.session_state.retriever:
                    # Create both extraction chains
                    llm = initialize_llm()
                    if llm:
                        st.session_state.pdf_chain = create_pdf_extraction_chain(st.session_state.retriever, llm)
                        st.session_state.web_chain = create_web_extraction_chain(llm)
                        return True
            return False
    except Exception as e:
        logger.error(f"Error in process_files: {e}", exc_info=True)
        raise

def main():
    # Initialize session state
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'pdf_chain' not in st.session_state:
        st.session_state.pdf_chain = None
    if 'web_chain' not in st.session_state:
        st.session_state.web_chain = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'extraction_performed' not in st.session_state:
        st.session_state.extraction_performed = False
    if 'scraped_table_html_cache' not in st.session_state:
        st.session_state.scraped_table_html_cache = None
    if 'current_part_number_scraped' not in st.session_state:
        st.session_state.current_part_number_scraped = None
    if 'pdf_processing_task' not in st.session_state:
        st.session_state.pdf_processing_task = None
    if 'pdf_processing_complete' not in st.session_state:
        st.session_state.pdf_processing_complete = False
    if 'pdf_processing_results' not in st.session_state:
        st.session_state.pdf_processing_results = None

    # Display header
    st.markdown("<h1 style='text-align: center;'>Document Information Extraction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload documents or enter part numbers to extract part information</p>", unsafe_allow_html=True)

    # Create two columns for file upload and part number input
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files containing part information"
        )

    with col2:
        st.markdown("### Enter Part Number")
        part_number = st.text_input(
            "Enter part number",
            help="Enter the part number to extract information from web sources"
        )

    # Process button
    if st.button("Extract Information", type="primary", use_container_width=True):
        if not uploaded_files and not part_number:
            st.warning("Please upload files or enter a part number to extract information.")
            return

        with st.spinner("Processing documents..."):
            try:
                # Initialize required components
                embedding_function = get_embedding_function()
                llm = initialize_llm()

                if embedding_function is None or llm is None:
                    st.error("Unable to initialize required components. Please try again later.")
                    return

                # Process files and part number
                async def process_all():
                    # Process PDFs if any are uploaded
                    pdf_docs = []
                    if uploaded_files:
                        temp_dir = os.path.join(os.getcwd(), "temp_pdf_files")
                        pdf_docs = await process_uploaded_pdfs(uploaded_files, temp_dir)
                    
                    # Process part number if provided
                    web_docs = []
                    if part_number:
                        web_docs = await process_web_urls([part_number])
                    
                    # Combine documents, prioritizing web docs if available
                    all_docs = web_docs if web_docs else pdf_docs
                    
                    if all_docs:
                        st.session_state.retriever = setup_vector_store(all_docs, embedding_function)
                        if st.session_state.retriever:
                            # Create both extraction chains
                            st.session_state.pdf_chain = create_pdf_extraction_chain(st.session_state.retriever, llm)
                            st.session_state.web_chain = create_web_extraction_chain(llm)
                            return True
                    return False

                # Run the async processing
                success = loop.run_until_complete(process_all())

                if success:
                    st.success("Information extraction completed successfully!")
                    st.session_state.extraction_performed = True
                else:
                    st.error("Failed to process documents. Please try again.")

            except Exception as e:
                st.error("An error occurred while processing the documents. Please try again.")
                logger.error(f"Error in main processing: {e}")

    # Display results if available
    if st.session_state.extraction_performed and st.session_state.pdf_processing_results:
        st.markdown("### Extraction Results")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(st.session_state.pdf_processing_results)
        
        # Display results in a table
        st.dataframe(results_df, use_container_width=True)
        
        # Add download button for results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results",
            data=csv,
            file_name="extraction_results.csv",
            mime="text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    main()