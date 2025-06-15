# --- Force python to use pysqlite3 based on chromadb docs ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- End override ---

# app.py
import streamlit as st

# Remove page configuration from here since it's imported
# st.set_page_config(
#     page_title="PDF Auto-Extraction with Groq",
#     page_icon="📄",
#     layout="wide"
# )

import os
import time
from loguru import logger
import json # Import the json library
import pandas as pd # Add pandas import
import re # Import the 're' module for regular expressions
import asyncio # Add asyncio import
import subprocess # To run playwright install
from typing import List
from langchain.docstore.document import Document
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# --- Install Playwright browsers needed by crawl4ai --- 
# This should run on startup in the Streamlit Cloud environment
def install_playwright_browsers():
    logger.info("Checking and installing Playwright browsers if needed...")
    try:
        # Use subprocess to run the command
        # stdout/stderr=subprocess.PIPE can capture output if needed
        # check=True will raise an error if the command fails
        process = subprocess.run([sys.executable, "-m", "playwright", "install"], 
                                 capture_output=True, text=True, check=False) # Use check=False initially to see output
        if process.returncode == 0:
             logger.success("Playwright browsers installed successfully (or already exist).")
        else:
             # Log stdout/stderr for debugging if it failed
             logger.error(f"Playwright browser install command failed with code {process.returncode}.")
             logger.error(f"stdout: {process.stdout}")
             logger.error(f"stderr: {process.stderr}")
             # Optionally raise an error or show a Streamlit warning
             # st.warning("Failed to install necessary Playwright browsers. Web scraping may fail.")
        # Alternative using playwright's internal API (might be cleaner if stable)
        # from playwright.driver import main as playwright_main
        # playwright_main.main(['install']) # Installs default browser (chromium)
        # logger.success("Playwright browsers installed successfully via internal API.")
    except FileNotFoundError:
        logger.error("Could not find 'playwright' command. Is playwright installed correctly?")
        st.error("Playwright not found. Please ensure 'playwright' is in requirements.txt")
    except Exception as e:
        logger.error(f"An error occurred during Playwright browser installation: {e}", exc_info=True)
        st.warning(f"An error occurred installing Playwright browsers: {e}. Web scraping may fail.")

# Import project modules
import config
from Extraction.pdf_processor import process_uploaded_pdfs, process_pdfs_in_background
from Extraction.vector_store import (
    get_embedding_function,
    setup_vector_store,
    load_existing_vector_store
)
# Updated imports from llm_interface
from Extraction.llm_interface import (
    initialize_llm,
    create_pdf_extraction_chain, # Use PDF chain func
    create_web_extraction_chain, # Use Web chain func
    _invoke_chain_and_process, # Use the helper directly
    scrape_website_table_html
)
# Import the prompts
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
# Import the NEW web prompts
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

# Run Playwright browser installation
install_playwright_browsers()

# --- Logging Configuration ---
# Configure Loguru logger (can be more flexible than standard logging)
# logger.add("logs/app_{time}.log", rotation="10 MB", level="INFO") # Example: Keep file logging if desired
# Toasts are disabled as per previous request
# Errors will still be shown via st.error where used explicitly

# --- Application State ---
# Only initialize state variables if they don't exist
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

# Initialize only missing state variables
for var, default_value in state_vars.items():
    if var not in st.session_state:
        st.session_state[var] = default_value

# --- Global Variables / Initialization ---
# Initialize embeddings (this is relatively heavy, do it once)
@st.cache_resource
def initialize_embeddings():
    # Let exceptions from get_embedding_function propagate
    embeddings = get_embedding_function()
    return embeddings

# Initialize LLM (also potentially heavy/needs API key check)
@st.cache_resource
def initialize_llm_cached():
    # logger.info("Attempting to initialize LLM...") # Log before calling if needed
    llm_instance = initialize_llm()
    # logger.success("LLM initialized successfully.") # Log after successful call if needed
    return llm_instance

# --- Wrap the cached function call in try-except ---
embedding_function = None
llm = None

try:
    logger.info("Attempting to initialize embedding function...")
    embedding_function = initialize_embeddings()
    if embedding_function:
         logger.success("Embedding function initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize embeddings: {e}", exc_info=True)
    st.error(f"Fatal Error: Could not initialize embedding model. Error: {e}")
    st.stop()

try:
    logger.info("Attempting to initialize LLM...")
    llm = initialize_llm_cached()
    if llm:
        logger.success("LLM initialized successfully.")
except Exception as e:
     logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
     st.error(f"Fatal Error: Could not initialize LLM. Error: {e}")
     st.stop()

# --- Check if initializations failed ---
if embedding_function is None or llm is None:
     if not st.exception:
        st.error("Core components (Embeddings or LLM) failed to initialize. Cannot continue.")
     st.stop()

# --- Health Check Functions ---
def update_health_check():
    """Update the health check timestamp."""
    st.session_state.last_health_check = time.time()

def check_health_check_timeout():
    """Check if we're approaching the health check timeout."""
    if 'last_health_check' not in st.session_state:
        st.session_state.last_health_check = time.time()
        return False
    
    elapsed = time.time() - st.session_state.last_health_check
    return elapsed > (config.HEALTH_CHECK_TIMEOUT - config.HEALTH_CHECK_GRACE_PERIOD)

# --- Load existing vector store or process uploads ---
# Reset evaluation state when processing new files
def reset_evaluation_state():
    st.session_state.evaluation_results = []
    st.session_state.evaluation_metrics = None
    st.session_state.extraction_performed = False # Reset the flag here too
    st.session_state.scraped_table_html_cache = None # Clear scraped HTML cache
    st.session_state.current_part_number_scraped = None # Clear scraped part number tracker
    # Clear data editor state if it exists
    if 'gt_editor' in st.session_state:
        del st.session_state['gt_editor']

# Try loading existing vector store and create BOTH extraction chains
if st.session_state.retriever is None and config.CHROMA_SETTINGS.is_persistent and embedding_function:
    logger.info("Attempting to load existing vector store...")
    st.session_state.retriever = load_existing_vector_store(embedding_function)
    if st.session_state.retriever:
        logger.success("Successfully loaded retriever from persistent storage.")
        st.session_state.processed_files = ["Existing data loaded from disk"]
        # --- Create BOTH Extraction Chains --- 
        logger.info("Creating extraction chains from loaded retriever...")
        st.session_state.pdf_chain = create_pdf_extraction_chain(st.session_state.retriever, llm)
        st.session_state.web_chain = create_web_extraction_chain(llm)
        if not st.session_state.pdf_chain or not st.session_state.web_chain:
            st.warning("Failed to create one or both extraction chains from loaded retriever.")
        # ------------------------------------
        # Don't reset evaluation if loading existing data, but ensure extraction hasn't run yet
        st.session_state.extraction_performed = False # Ensure flag is false on load
    else:
        logger.warning("No existing persistent vector store found or failed to load.")

# --- UI Layout ---
persistence_enabled = config.CHROMA_SETTINGS.is_persistent
st.title("📄 PDF Auto-Extraction with Groq") # Updated title
st.markdown("Upload PDF documents, process them, and view automatically extracted information.") # Updated description
st.markdown(f"**Model:** `{config.LLM_MODEL_NAME}` | **Embeddings:** `{config.EMBEDDING_MODEL_NAME}` | **Persistence:** `{'Enabled' if persistence_enabled else 'Disabled'}`")

# --- Main Area for Document Upload and Processing ---
st.header("1. Document Upload and Processing")

# Initialize session state for uploaded files if not exists
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'uploaded_files_dict' not in st.session_state:
    st.session_state.uploaded_files_dict = {}

# Create two columns for the upload section
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Upload PDF Files")
    uploaded_files = st.file_uploader(
        "Select one or more PDF files",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader",
        help="Upload PDF documents containing part information"
    )
    
    # Update session state with new uploads
    if uploaded_files:
        # Create a dictionary of current files by name
        current_files_dict = {f.name: f for f in st.session_state.uploaded_files}
        
        # Process new uploads
        new_files = []
        for file in uploaded_files:
            if file.name not in current_files_dict:
                # New file
                new_files.append(file)
                current_files_dict[file.name] = file
            else:
                # File exists, check if content changed
                existing_file = current_files_dict[file.name]
                if file.getvalue() != existing_file.getvalue():
                    # Content changed, update
                    new_files.append(file)
                    current_files_dict[file.name] = file
                else:
                    # Same content, keep existing
                    new_files.append(existing_file)
        
        # Update session state
        st.session_state.uploaded_files = new_files
        if new_files:
            st.success(f"Successfully processed {len(new_files)} file(s)")
    
    # Display currently uploaded files
    if st.session_state.uploaded_files:
        st.write("Currently uploaded files:")
        for file in st.session_state.uploaded_files:
            st.write(f"- {file.name}")

with col2:
    st.subheader("Part Number")
    part_number = st.text_input(
        "Enter Part Number (Optional)",
        key="part_number_input",
        value=st.session_state.get("part_number_input", ""),
        help="Enter the part number to search for additional information online"
    )

# Process button in a full-width container
st.markdown("---")
process_button = st.button("🚀 Process Documents", key="process_button", type="primary", use_container_width=True)

# Check for API Key and show a warning (non-blocking)
if not config.GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY not found! Some features may be limited. Please set the GROQ_API_KEY environment variable for full functionality.", icon="⚠️")
    st.info("You can set the API key by:")
    st.markdown("""
    1. Creating a `.env` file in the root directory with:
       ```
       GROQ_API_KEY=your_api_key_here
       ```
    2. Or setting it as an environment variable:
       ```
       set GROQ_API_KEY=your_api_key_here  # Windows
       export GROQ_API_KEY=your_api_key_here  # Linux/Mac
       ```
    """)

# Process button logic
if process_button and st.session_state.uploaded_files:
    if not embedding_function or not llm:
        st.error("Cannot process documents: Core components (Embeddings or LLM) failed to initialize. Please check the error messages above and ensure your API keys are set correctly.")
    else:
        # Reset state including evaluation and the extraction flag
        st.session_state.retriever = None
        # Reset BOTH chains
        st.session_state.pdf_chain = None
        st.session_state.web_chain = None
        st.session_state.processed_files = []
        reset_evaluation_state() # Reset evaluation results AND extraction_performed flag

        filenames = [f.name for f in st.session_state.uploaded_files]
        logger.info(f"Starting processing for {len(filenames)} files: {', '.join(filenames)}")
        
        # Initialize processed_docs
        processed_docs = []
        
        # --- PDF Processing ---
        with st.spinner("Processing PDFs... Loading, cleaning, splitting..."):
            try:
                start_time = time.time()
                temp_dir = os.path.join(os.getcwd(), "temp_pdf_files")
                
                # Create event loop for async processing
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Start both PDF and web processing in parallel
                async def process_all():
                    """Process PDFs and web data in parallel."""
                    # Start PDF processing
                    pdf_task = asyncio.create_task(
                        process_uploaded_pdfs(st.session_state.uploaded_files, temp_dir)
                    )
                    
                    # Start web processing if part number is provided
                    web_task = None
                    if st.session_state.get("part_number_input"):
                        web_task = asyncio.create_task(
                            process_web_urls([st.session_state.get("part_number_input")])
                        )
                    
                    # Wait for both tasks to complete
                    tasks = [pdf_task]
                    if web_task:
                        tasks.append(web_task)
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results
                    pdf_docs = results[0] if not isinstance(results[0], Exception) else []
                    web_docs = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else []
                    
                    # Combine results, prioritizing web docs if available
                    if web_docs:
                        return web_docs
                    return pdf_docs
                
                # Run the parallel processing
                processed_docs = loop.run_until_complete(process_all())
                
                processing_time = time.time() - start_time
                logger.info(f"Processing took {processing_time:.2f} seconds.")
            except Exception as e:
                logger.error(f"Failed during processing phase: {e}", exc_info=True)
                st.error(f"Error during processing: {e}")
                processed_docs = []  # Ensure it's empty on error

        # --- Vector Store Indexing ---
        if processed_docs and len(processed_docs) > 0:
            logger.info(f"Generated {len(processed_docs)} document chunks.")
            with st.spinner("Indexing documents in vector store..."):
                try:
                    start_time = time.time()
                    st.session_state.retriever = setup_vector_store(processed_docs, embedding_function)
                    indexing_time = time.time() - start_time
                    logger.info(f"Vector store setup took {indexing_time:.2f} seconds.")

                    if st.session_state.retriever:
                        st.session_state.processed_files = filenames # Update list
                        logger.success("Vector store setup complete. Retriever is ready.")
                        # --- Create BOTH Extraction Chains --- 
                        with st.spinner("Preparing extraction engines..."):
                             st.session_state.pdf_chain = create_pdf_extraction_chain(st.session_state.retriever, llm)
                             st.session_state.web_chain = create_web_extraction_chain(llm)
                        if st.session_state.pdf_chain and st.session_state.web_chain:
                            logger.success("Extraction chains created.")
                            # Keep extraction_performed as False here, it will run in the main section
                            st.success(f"Successfully processed {len(filenames)} file(s). Evaluation below.") # Update message
                        else:
                            st.error("Failed to create one or both extraction chains after processing.")
                            # reset_evaluation_state() called earlier is sufficient
                    else:
                        st.error("Failed to setup vector store after processing PDFs.")
                        # reset_evaluation_state() called earlier is sufficient
                except Exception as e:
                     logger.error(f"Failed during vector store setup: {e}", exc_info=True)
                     st.error(f"Error setting up vector store: {e}")
                     # reset_evaluation_state() called earlier is sufficient
        elif not processed_docs and st.session_state.uploaded_files:
            st.warning("No text could be extracted or processed from the uploaded PDFs.")
            # reset_evaluation_state() called earlier is sufficient

        elif process_button and not st.session_state.uploaded_files:
         st.warning("Please upload at least one PDF file before processing.")

    # --- Display processed files status ---
    if st.session_state.processed_files:
        st.info(f"Processed files: {', '.join(st.session_state.processed_files)}")

    # --- Main Area for Displaying Extraction Results ---
    st.header("2. Extracted Information")

    # --- Get current asyncio event loop --- 
    # Needed for both scraping and running the async extraction chain
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # 'RuntimeError: There is no current event loop...'
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    # -------------------------------------

    # Check if BOTH chains are ready before proceeding
    if not st.session_state.pdf_chain or not st.session_state.web_chain:
        st.info("Upload and process documents using the sidebar to see extracted results here.")
        # Ensure evaluation state is also clear if no chain
        if not st.session_state.evaluation_results and not st.session_state.extraction_performed:
            reset_evaluation_state()  # Ensure reset if no chain and extraction not done
        else:
            # --- Block 1: Run Extraction (if needed) --- 
            if (st.session_state.pdf_chain and st.session_state.web_chain) and not st.session_state.extraction_performed:
                # --- Get Part Number --- 
                part_number = st.session_state.get("part_number_input", "").strip()
                # ---------------------

                # Define the prompts (attribute keys mapped to PDF and WEB instructions)
                prompts_to_run = { 
                    # Material Properties
                    "Material Filling": {"pdf": MATERIAL_PROMPT, "web": MATERIAL_FILLING_WEB_PROMPT},
                    "Material Name": {"pdf": MATERIAL_NAME_PROMPT, "web": MATERIAL_NAME_WEB_PROMPT},
                    # Physical / Mechanical Attributes
                    "Pull-to-Seat": {"pdf": PULL_TO_SEAT_PROMPT, "web": PULL_TO_SEAT_WEB_PROMPT},
                    "Gender": {"pdf": GENDER_PROMPT, "web": GENDER_WEB_PROMPT},
                    "Height [MM]": {"pdf": HEIGHT_MM_PROMPT, "web": HEIGHT_MM_WEB_PROMPT},
                    "Length [MM]": {"pdf": LENGTH_MM_PROMPT, "web": LENGTH_MM_WEB_PROMPT},
                    "Width [MM]": {"pdf": WIDTH_MM_PROMPT, "web": WIDTH_MM_WEB_PROMPT},
                    "Number of Cavities": {"pdf": NUMBER_OF_CAVITIES_PROMPT, "web": NUMBER_OF_CAVITIES_WEB_PROMPT},
                    "Number of Rows": {"pdf": NUMBER_OF_ROWS_PROMPT, "web": NUMBER_OF_ROWS_WEB_PROMPT},
                    "Mechanical Coding": {"pdf": MECHANICAL_CODING_PROMPT, "web": MECHANICAL_CODING_WEB_PROMPT},
                    "Colour": {"pdf": COLOUR_PROMPT, "web": COLOUR_WEB_PROMPT},
                    "Colour Coding": {"pdf": COLOUR_CODING_PROMPT, "web": COLOUR_CODING_WEB_PROMPT},
                    # Sealing & Environmental
                    "Working Temperature": {"pdf": WORKING_TEMPERATURE_PROMPT, "web": WORKING_TEMPERATURE_WEB_PROMPT},
                    "Housing Seal": {"pdf": HOUSING_SEAL_PROMPT, "web": HOUSING_SEAL_WEB_PROMPT},
                    "Wire Seal": {"pdf": WIRE_SEAL_PROMPT, "web": WIRE_SEAL_WEB_PROMPT},
                    "Sealing": {"pdf": SEALING_PROMPT, "web": SEALING_WEB_PROMPT},
                    "Sealing Class": {"pdf": SEALING_CLASS_PROMPT, "web": SEALING_CLASS_WEB_PROMPT},
                    # Terminals & Connections
                    "Contact Systems": {"pdf": CONTACT_SYSTEMS_PROMPT, "web": CONTACT_SYSTEMS_WEB_PROMPT},
                    "Terminal Position Assurance": {"pdf": TERMINAL_POSITION_ASSURANCE_PROMPT, "web": TERMINAL_POSITION_ASSURANCE_WEB_PROMPT},
                    "Connector Position Assurance": {"pdf": CONNECTOR_POSITION_ASSURANCE_PROMPT, "web": CONNECTOR_POSITION_ASSURANCE_WEB_PROMPT},
                    "Closed Cavities": {"pdf": CLOSED_CAVITIES_PROMPT, "web": CLOSED_CAVITIES_WEB_PROMPT},
                    # Assembly & Type
                    "Pre-Assembled": {"pdf": PRE_ASSEMBLED_PROMPT, "web": PRE_ASSEMBLED_WEB_PROMPT},
                    "Connector Type": {"pdf": CONNECTOR_TYPE_PROMPT, "web": CONNECTOR_TYPE_WEB_PROMPT},
                    "Set Kit": {"pdf": SET_KIT_PROMPT, "web": SET_KIT_WEB_PROMPT},
                    # Specialized Attributes
                    "HV Qualified": {"pdf": HV_QUALIFIED_PROMPT, "web": HV_QUALIFIED_WEB_PROMPT}
                }

                # Call the async function
                asyncio.run(process_attributes_main())

    # --- Block 2: Display Results (if available) ---
    if st.session_state.evaluation_results:
        # Prepare data for export
        export_df = pd.DataFrame(st.session_state.evaluation_results)
        export_summary = st.session_state.evaluation_metrics if st.session_state.evaluation_metrics else {}

        # Convert DataFrame to CSV
        @st.cache_data # Cache the conversion
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv_data = convert_df_to_csv(export_df)

        # Convert summary dict to JSON
        json_summary_data = json.dumps(export_summary, indent=2).encode('utf-8')

        export_cols = st.columns(2)
        with export_cols[0]:
            st.download_button(
                label="📥 Download Detailed Results (CSV)",
                data=csv_data,
                file_name='detailed_extraction_results.csv',
                mime='text/csv',
                key='download_csv'
            )
        with export_cols[1]:
            st.download_button(
                label="📥 Download Summary Metrics (JSON)",
                data=json_summary_data,
                file_name='evaluation_summary.json',
                mime='application/json',
                key='download_json'
            )
    else:
        st.info("Process documents and calculate metrics to enable export.")

    # --- Block 3: Handle cases where extraction ran but yielded nothing ---
    if (st.session_state.pdf_chain or st.session_state.web_chain) and st.session_state.extraction_performed and not st.session_state.evaluation_results:
        st.warning("Extraction process completed, but no valid results were generated for some fields. Check logs or raw outputs if available.")

async def process_attributes_main():
    """Main async function for processing attributes."""
    if (st.session_state.pdf_chain and st.session_state.web_chain) and not st.session_state.extraction_performed:
        # Initialize health check
        update_health_check()
        
        # Get part number and scrape web data
        part_number = st.session_state.get("part_number_input", "").strip()
        scraped_table_html = None
        
        if part_number:
            if (st.session_state.current_part_number_scraped == part_number and 
                st.session_state.scraped_table_html_cache is not None):
                scraped_table_html = st.session_state.scraped_table_html_cache
            else:
                with st.spinner("Scraping web data..."):
                    try:
                        scraped_table_html = await scrape_website_table_html(part_number)
                        if scraped_table_html:
                            st.session_state.scraped_table_html_cache = scraped_table_html
                            st.session_state.current_part_number_scraped = part_number
                    except Exception as e:
                        logger.error(f"Web scraping error: {e}")
        
        # Process all attributes
        with st.spinner("Processing attributes..."):
            try:
                intermediate_results = await process_all_attributes(
                    prompts_to_run,
                    st.session_state.web_chain,
                    st.session_state.pdf_chain,
                    scraped_table_html
                )
                
                # Convert results to list and update session state
                extraction_results_list = list(intermediate_results.values())
                st.session_state.evaluation_results = extraction_results_list
                st.session_state.extraction_performed = True
                
                st.success("Extraction complete!")
            except Exception as e:
                logger.error(f"Error during attribute processing: {e}")
                st.error("Error during processing. Please try again.")

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
    
    # Check for API key
    if not st.session_state.get('GROQ_API_KEY'):
        st.error("Please set your GROQ API key in the .env file")
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

if __name__ == "__main__":
    main()