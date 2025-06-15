# llm_interface.py
import requests
import json
from typing import List, Dict, Optional
from loguru import logger
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.docstore.document import Document

# Recommended: Use LangChain's Groq integration
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

import config # Import configuration
import asyncio # Need asyncio for crawl4ai
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from bs4 import BeautifulSoup # Import BeautifulSoup
import re # Import re for regular expressions
import time # Import time for timing

class LLMInterface:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMInterface, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the LLM interface with website configurations."""
        self.website_configs = [
            {
                "name": "TE Connectivity",
                "base_url_template": "https://www.te.com/en/product-{part_number}.html",
                "pre_extraction_js": (
                    "(async () => {"
                    "    const expandButtonSelector = '#pdp-features-expander-btn';"
                    "    const featuresPanelSelector = '#pdp-features-tabpanel';"
                    "    const expandButton = document.querySelector(expandButtonSelector);"
                    "    const featuresPanel = document.querySelector(featuresPanelSelector);"
                    "    if (expandButton && expandButton.getAttribute('aria-selected') === 'false') {"
                    "        console.log('Features expand button indicates collapsed state, clicking...');"
                    "        expandButton.click();"
                    "        await new Promise(r => setTimeout(r, 1500));"
                    "        console.log('Expand button clicked and waited.');"
                    "    } else if (expandButton) {"
                    "        console.log('Features expand button already indicates expanded state.');"
                    "    } else {"
                    "        console.log('Features expand button selector not found:', expandButtonSelector);"
                    "        if (featuresPanel && !featuresPanel.offsetParent) {"
                    "           console.warn('Button not found, but panel seems hidden. JS might need adjustment.');"
                    "        } else if (!featuresPanel) {"
                    "           console.warn('Neither expand button nor features panel found.');"
                    "        }"
                    "    }"
                    "})();"
                ),
                "table_selector": "#pdp-features-tabpanel",
                "part_number_pattern": r"^\d{7}-\d$"
            },
            {
                "name": "Molex",
                "base_url_template": "https://www.molex.com/en-us/products/part-detail/{part_number}#part-details",
                "pre_extraction_js": None,
                "table_selector": "body",
                "part_number_pattern": r"^\d{9}$"
            },
            {
                "name": "TraceParts",
                "base_url_template": "https://www.traceparts.com/en/search?CatalogPath=&KeepFilters=true&Keywords={part_number}&SearchAction=Keywords",
                "pre_extraction_js": None,
                "table_selector": ".technical-data",
                "part_number_pattern": None
            }
        ]

    @staticmethod
    @logger.catch(reraise=True)
    def initialize_llm():
        """Initializes and returns the Groq LLM client."""
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in the environment variables.")

        try:
            llm = ChatGroq(
                temperature=config.LLM_TEMPERATURE,
                groq_api_key=config.GROQ_API_KEY,
                model_name=config.LLM_MODEL_NAME,
                max_tokens=config.LLM_MAX_OUTPUT_TOKENS
            )
            return llm
        except Exception as e:
            raise ConnectionError(f"Could not initialize Groq LLM: {e}")

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """Formats retrieved documents into a string for the prompt."""
        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            start_index = doc.metadata.get('start_index', None)
            chunk_info = f"Chunk {i+1}" + (f" (starts at char {start_index})" if start_index is not None else "")
            context_parts.append(
                f"{chunk_info} from '{source}' (Page {page}):\\n{doc.page_content}"
            )
        return "\\n\\n---\\n\\n".join(context_parts)

    def clean_scraped_html(self, html_content: str, site_name: str) -> Optional[str]:
        """Parses scraped HTML using BeautifulSoup and extracts key-value pairs."""
        if not html_content:
            return None

        logger.debug(f"Cleaning HTML content from {site_name}...")
        soup = BeautifulSoup(html_content, 'html.parser')
        extracted_texts = []

        try:
            if site_name == "TE Connectivity":
                feature_items = soup.find_all('li', class_='product-feature')
                if not feature_items:
                    panel = soup.find(id='pdp-features-tabpanel')
                    if panel:
                        feature_items = panel.find_all('li', class_='product-feature')
                
                if feature_items:
                    for item in feature_items:
                        title_span = item.find('span', class_='feature-title')
                        value_em = item.find('em', class_='feature-value')
                        if title_span and value_em:
                            title = title_span.get_text(strip=True).replace(':', '').strip()
                            value = value_em.get_text(strip=True)
                            if title and value:
                                extracted_texts.append(f"{title}: {value}")
                    logger.info(f"Extracted {len(extracted_texts)} features from TE Connectivity HTML.")
                else:
                    logger.warning(f"Could not find 'li.product-feature' items in the TE Connectivity HTML provided.")

            elif site_name == "Molex":
                tables = soup.find_all('table')
                for table in tables:
                    section_title = "General"
                    prev_element = table.find_previous(['h4', 'h3', 'h2'])
                    if prev_element:
                        section_title = prev_element.get_text(strip=True)

                    rows = table.find_all('tr')
                    for row in rows:
                        headers = row.find_all('th')
                        data_cells = row.find_all('td')
                        
                        for i in range(0, len(headers), 2):
                            if i < len(headers):
                                label = headers[i].get_text(strip=True).replace(':', '').strip()
                                if i < len(data_cells):
                                    value = data_cells[i].get_text(strip=True)
                                    if label and value:
                                        extracted_texts.append(f"{section_title} - {label}: {value}")
                                
                                if i + 1 < len(headers) and i + 1 < len(data_cells):
                                    label2 = headers[i + 1].get_text(strip=True).replace(':', '').strip()
                                    value2 = data_cells[i + 1].get_text(strip=True)
                                    if label2 and value2:
                                        extracted_texts.append(f"{section_title} - {label2}: {value2}")

                if extracted_texts:
                    logger.info(f"Extracted {len(extracted_texts)} specifications from Molex HTML.")
                else:
                    logger.warning("No specifications found in Molex HTML tables.")
                    for element in soup.find_all(['div', 'p', 'span']):
                        text = element.get_text(strip=True)
                        if ':' in text:
                            parts = text.split(':', 1)
                            if len(parts) == 2:
                                label = parts[0].strip()
                                value = parts[1].strip()
                                if label and value:
                                    extracted_texts.append(f"{label}: {value}")

            elif site_name == "TraceParts":
                logger.warning(f"HTML cleaning logic for TraceParts is not implemented yet.")
                pass

            else:
                logger.warning(f"No specific HTML cleaning logic defined for site: {site_name}. Returning raw text content as fallback.")
                return soup.get_text(separator=' ', strip=True)

            if not extracted_texts:
                logger.warning(f"HTML cleaning for {site_name} resulted in no text extracted.")
                return None

            return "\\n".join(extracted_texts)

        except Exception as e:
            logger.error(f"Error cleaning HTML for {site_name}: {e}", exc_info=True)
            return None

    async def scrape_website_table_html(self, part_number: str) -> Optional[str]:
        """Attempts to scrape the outer HTML of a features table, then cleans it."""
        if not part_number:
            logger.debug("Web scraping skipped: No part number provided.")
            return None

        logger.info(f"Attempting web scrape for features table / Part#: '{part_number}'...")

        matching_site = None
        for site_config in self.website_configs:
            pattern = site_config.get("part_number_pattern")
            if pattern and re.match(pattern, part_number):
                matching_site = site_config
                break

        sites_to_try = [matching_site] if matching_site else self.website_configs

        for site_config in sites_to_try:
            selector = site_config.get("table_selector")
            site_name = site_config.get("name", "Unknown Site")
            if not selector:
                logger.warning(f"No table_selector defined for {site_name}. Skipping.")
                continue

            target_url = site_config["base_url_template"].format(part_number=part_number)
            js_code = site_config.get("pre_extraction_js")
            logger.debug(f"Attempting scrape on {site_name} ({target_url}) for table selector '{selector}'")

            extraction_schema = {
                "name": "TableHTML",
                "baseSelector": "html",
                "fields": [
                    {"name": "html_content", "selector": selector, "type": "html"}
                ]
            }
            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                js_code=[js_code] if js_code else None,
                page_timeout=20000,
                verbose=False,
                extraction_strategy=JsonCssExtractionStrategy(extraction_schema)
            )
            browser_config = BrowserConfig(verbose=False)

            try:
                async with AsyncWebCrawler(config=browser_config) as crawler:
                    results = await crawler.arun_many(urls=[target_url], config=run_config)
                    result = results[0]

                    if result.success and result.extracted_content:
                        raw_html = None
                        try:
                            extracted_data_list = json.loads(result.extracted_content)
                            if extracted_data_list and isinstance(extracted_data_list, list) and len(extracted_data_list) > 0:
                                first_item = extracted_data_list[0]
                                if isinstance(first_item, dict) and "html_content" in first_item:
                                    raw_html = str(first_item["html_content"]).strip()
                            else:
                                logger.debug(f"Extraction strategy did not find or extract HTML for selector '{selector}' on {site_name}.")

                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON from crawl4ai extraction result for table HTML on {site_name}: {result.extracted_content[:100]}...")
                        except Exception as parse_error:
                            logger.error(f"Error processing extracted JSON for {site_name}: {parse_error}", exc_info=True)

                        if raw_html:
                            cleaned_text = self.clean_scraped_html(raw_html, site_name)
                            if cleaned_text:
                                logger.success(f"Successfully scraped and cleaned features table from {site_name}.")
                                return cleaned_text
                            else:
                                logger.warning(f"HTML was scraped from {site_name}, but cleaning failed or yielded no text.")

                    elif result.error_message:
                        logger.warning(f"Scraping page failed for {site_name} ({target_url}): {result.error_message}")
                    else:
                        logger.debug(f"Scraping attempt for {site_name} yielded no extracted content or error message.")

            except asyncio.TimeoutError:
                logger.warning(f"Scraping timed out for {site_name} ({target_url})")
            except Exception as e:
                logger.error(f"Unexpected error during web scraping for {site_name} ({target_url}): {e}", exc_info=True)

        logger.info(f"Web scraping finished for features table. No usable cleaned text found across configured sites.")
        return None

    @staticmethod
    def create_pdf_extraction_chain(retriever, llm):
        """Creates a RAG chain that uses ONLY PDF context and detailed instructions."""
        logger.info("Starting PDF extraction chain creation...")
        if retriever is None or llm is None:
            logger.error("Retriever or LLM is not initialized for PDF extraction chain.")
            return None

        logger.info("Creating PDF extraction prompt template...")
        template = """
You are an expert data extractor. Your goal is to extract a specific piece of information based on the Extraction Instructions provided below, using ONLY the Document Context from PDFs.

Part Number Information (if provided by user):
{part_number}

--- Document Context (from PDFs) ---
{context}
--- End Document Context ---

Extraction Instructions:
{extraction_instructions}

---
IMPORTANT: Respond with ONLY a single, valid JSON object containing exactly one key-value pair.
- The key for the JSON object MUST be the string: "{attribute_key}"
- The value MUST be the extracted result determined by following the Extraction Instructions using the Document Context provided above.
- Provide the value as a JSON string. Examples: "GF, T", "none", "NOT FOUND", "Female", "7.2", "999".
- Do NOT include any explanations, reasoning, or any text outside of the single JSON object in your response.

Example Output Format:
{{"{attribute_key}": "extracted_value_from_pdf"}}

Output:
"""
        prompt = PromptTemplate.from_template(template)
        logger.info("PDF extraction prompt template created")

        logger.info("Building PDF extraction chain...")
        pdf_chain = (
            RunnableParallel(
                context=RunnablePassthrough() | (lambda x: retriever.invoke(f"Extract information about {x['attribute_key']} for part number {x.get('part_number', 'N/A')}")) | LLMInterface.format_docs,
                extraction_instructions=RunnablePassthrough(),
                attribute_key=RunnablePassthrough(),
                part_number=RunnablePassthrough()
            )
            .assign(
                extraction_instructions=lambda x: x['extraction_instructions']['extraction_instructions'],
                attribute_key=lambda x: x['attribute_key']['attribute_key'],
                part_number=lambda x: x['part_number'].get('part_number', "Not Provided")
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("PDF Extraction RAG chain created successfully with Mistral integration.")
        return pdf_chain

    @staticmethod
    def create_web_extraction_chain(llm):
        """Creates a simpler chain that uses ONLY cleaned website data."""
        logger.info("Starting web extraction chain creation...")
        if llm is None:
            logger.error("LLM is not initialized for Web extraction chain.")
            return None

        logger.info("Creating web extraction prompt template...")
        template = """
You are an expert data extractor. Your goal is to answer a specific piece of information by applying the logic described in the 'Extraction Instructions' to the 'Cleaned Scraped Website Data' provided below. Use ONLY the provided website data as your context.

--- Cleaned Scraped Website Data ---
{cleaned_web_data}
--- End Cleaned Scraped Website Data ---

Extraction Instructions:
{extraction_instructions}

---
IMPORTANT: Follow the Extraction Instructions carefully using the website data.
Respond with ONLY a single, valid JSON object containing exactly one key-value pair.
- The key for the JSON object MUST be the string: "{attribute_key}"
- The value MUST be the result obtained by applying the Extraction Instructions to the Cleaned Scraped Website Data.
- Provide the value as a JSON string.
- If the information cannot be determined from the Cleaned Scraped Website Data based on the instructions, the value MUST be "NOT FOUND".
- Do NOT include any explanations or reasoning outside the JSON object.

Example Output Format:
{{"{attribute_key}": "extracted_value_based_on_instructions"}}

Output:
"""
        prompt = PromptTemplate.from_template(template)
        logger.info("Web extraction prompt template created")

        logger.info("Building web extraction chain...")
        web_chain = (
            RunnableParallel(
                cleaned_web_data=RunnablePassthrough(),
                extraction_instructions=RunnablePassthrough(),
                attribute_key=RunnablePassthrough()
            )
            .assign(
                cleaned_web_data=lambda x: x['cleaned_web_data']['cleaned_web_data'],
                extraction_instructions=lambda x: x['extraction_instructions']['extraction_instructions'],
                attribute_key=lambda x: x['attribute_key']['attribute_key']
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("Web Data Extraction chain created successfully with Mistral integration.")
        return web_chain

    @staticmethod
    async def _invoke_chain_and_process(chain, input_data, attribute_key):
        """Helper to invoke chain, handle errors, and clean response."""
        logger.info(f"Invoking chain for attribute: {attribute_key}")
        logger.debug(f"Input data for {attribute_key}: {input_data}")
        
        try:
            start_time = time.time()
            response = await chain.ainvoke(input_data)
            processing_time = time.time() - start_time
            
            log_msg = f"Chain invoked successfully for '{attribute_key}' in {processing_time:.2f} seconds."
            if response:
                log_msg += f" Response length: {len(response)}"
            logger.info(log_msg)
            logger.debug(f"Raw response for {attribute_key}: {response}")

            if response is None:
                logger.error(f"Chain invocation returned None for '{attribute_key}'")
                return json.dumps({"error": f"Chain invocation returned None for {attribute_key}"})

            cleaned_response = response
            
            # 1. Remove <think> tags
            think_start_tag = "<think>"
            think_end_tag = "</think>"
            start_index_think = cleaned_response.find(think_start_tag)
            end_index_think = cleaned_response.find(think_end_tag)
            if start_index_think != -1 and end_index_think != -1 and end_index_think > start_index_think:
                cleaned_response = cleaned_response[end_index_think + len(think_end_tag):].strip()
                logger.debug(f"Removed think tags for {attribute_key}")

            # 2. Remove ```json ... ``` markdown
            if cleaned_response.strip().startswith("```json"):
                cleaned_response = cleaned_response.strip()[7:]
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
                logger.debug(f"Removed markdown for {attribute_key}")

            # 3. Find the first '{' and the last '}' to isolate the JSON object
            try:
                first_brace = cleaned_response.find('{')
                last_brace = cleaned_response.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    potential_json = cleaned_response[first_brace : last_brace + 1]
                    json.loads(potential_json)  # Validate JSON
                    cleaned_response = potential_json
                    logger.debug(f"Successfully isolated JSON for '{attribute_key}': {cleaned_response}")
                else:
                    logger.warning(f"Could not find clear JSON braces {{...}} in response for '{attribute_key}'. Using original cleaned response.")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse isolated JSON for '{attribute_key}'. Using original cleaned response. Raw: {cleaned_response}")
            except Exception as e:
                logger.error(f"Unexpected error during JSON isolation for '{attribute_key}': {e}")

            return cleaned_response
        except Exception as e:
            logger.error(f"Error in chain invocation for {attribute_key}: {e}", exc_info=True)
            return json.dumps({"error": f"Chain invocation error: {str(e)}"})

# Create singleton instance
llm_interface = LLMInterface()

# Export commonly used functions
initialize_llm = LLMInterface.initialize_llm
create_pdf_extraction_chain = LLMInterface.create_pdf_extraction_chain
create_web_extraction_chain = LLMInterface.create_web_extraction_chain
scrape_website_table_html = llm_interface.scrape_website_table_html
_invoke_chain_and_process = LLMInterface._invoke_chain_and_process