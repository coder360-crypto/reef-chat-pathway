# Import necessary libraries for file handling and system operations
import os
import sys

# Add parent directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import standard libraries
import json
import logging
import time
from typing import Dict, List, Optional

# Import third-party libraries
import fitz  # PyMuPDF2
import pytesseract
import requests
from bs4 import BeautifulSoup
from config import AgentsConfig as Config
from json_repair import repair_json
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from PIL import Image
from pydantic import BaseModel, Field

# Set up project root path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

config = Config()


class FileParser:
    """
    A class for parsing different types of files and extracting text content.

    """
    def __init__(self):
        pass

    def parse_pdf_to_text(self, file_path):
        """
        Extracts text from a PDF file using OCR.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            str: Extracted text from the PDF.
        """
        text = ""
        try:
            document = fitz.open(file_path)
            for page_num in range(len(document)):
                page = document.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img)
        except Exception as e:
            logger.error(
                f"An error occurred while extracting text from the PDF: {str(e)}"
            )
        return text


class PatentSearch:
    """
    A class for searching and retrieving patent information from various APIs.

    """
    def __init__(self):
        self.google_api_key = config.GOOGLE_API_KEY
        self.cx = config.CUSTOM_SEARCH_ENGINE_ID
        self.serp_api_key = config.SERP_API_KEY

    def call_serp_api(self, url, params=None):
        try:
            time.sleep(0.5)  # Add a delay of 500ms between requests
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred: {str(e)}")
            return None

    def call_google_api(self, query):
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "cx": self.cx,
            "key": self.google_api_key,
            "num": 10,
        }
        try:
            time.sleep(0.5)  # Add a delay of 500ms between requests
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API Error: {str(e)}")
            return None

    def scrape_patents_info(self, patent_ids: list):
        url = "https://serpapi.com/search"
        contract_summary = []

        for patent_id in patent_ids:
            params = {
                "q": patent_id,
                "api_key": self.serp_api_key,
                "engine": "google_patents",
            }
            response = self.call_serp_api(url, params)

            new_response = self.call_serp_api(
                response["organic_results"][0].get("serpapi_link", []),
                {"api_key": params["api_key"]},
            )
            title = new_response["title"]

            html_link = new_response["description_link"]
            page = requests.get(html_link)
            page.raise_for_status()

            soup = BeautifulSoup(page.content, "html.parser")
            text = soup.get_text(separator=" ", strip=True)

            start_index = text.find("SUMMARY")
            end_index = text.find("DESCRIPTION")
            if start_index != -1 and end_index != -1:
                text = text[start_index:end_index]
            else:
                text = "Relevant section not found."

            response_dict = {
                "summary": text,
                "patent_id": patent_id,
                "title": title,
                "patent_link": html_link,
            }
            contract_summary.append(response_dict)
        return contract_summary

    def get_patents_info(self, queries: List[str] | str):
        """
        Searches and retrieves patent information for given queries.

        Args:
            queries (Union[List[str], str]): Search queries for patents.

        Returns:
            list: List of dictionaries containing patent information.
        """
        if isinstance(queries, str):
            queries = [queries]

        patent_ids = set()
        for query in queries:
            api_response = self.call_google_api(query)
            pdf_urls = [item["link"] for item in api_response.get("items", [])]
            for url in pdf_urls:
                patent_id = url.split("/")[-2].split(".")[0]
                if patent_id.startswith("US"):
                    patent_ids.add(patent_id)

        return self.scrape_patents_info(list(patent_ids))


class TrademarkSearch:
    """
    A class for searching and processing trademark information from APIs.
    
    Args:
        base_url (str): Base URL for the trademark API
        headers (dict): Headers required for API authentication
        
    Returns:
        None
    """
    def __init__(self, base_url, headers):
        """
        Initialize the TrademarkSearch with base API URL and headers.

        Parameters:
            base_url (str): The base URL of the API endpoint.
            headers (dict): The headers required for the API request.
        """
        self.base_url = base_url
        self.headers = headers

    def fetch_data(self, query):
        """
        Fetch data from the trademark API based on a query parameter.

        Parameters:
            query (str): The query string to search for trademarks.

        Returns:
            dict: The response data from the API.
        """
        url = f"{self.base_url}/{query}/active"
        try:
            time.sleep(0.5)  # Add a delay of 500ms between requests
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred: {str(e)}")
            return None

    def process_data(self, data):
        """
        Process the raw data from the API into a structured format.

        Parameters:
            data (dict): The raw data fetched from the API.

        Returns:
            list: A list of structured trademark data.
        """
        processed_data = []
        for item in data.get("items", []):
            result = {
                "keyword": item.get("keyword"),
                "registration_number": item.get("registration_number"),
                "serial_number": item.get("serial_number"),
                "status_label": item.get("status_label"),
                "registration_date": item.get("registration_date"),
                "expiration_date": item.get("expiration_date"),
                "description": item.get("description"),
                "owner_details": [
                    self._process_owner(owner) for owner in item.get("owners", [])
                ],
            }
            processed_data.append(result)
        return processed_data[:40]

    @staticmethod
    def _process_owner(owner):
        """
        Process owner details.

        Parameters:
            owner (dict): Owner details from the API response.

        Returns:
            dict: Processed owner details.
        """
        return {
            "name": owner.get("name"),
            "address1": owner.get("address1"),
            "address2": owner.get("address2"),
            "city": owner.get("city"),
            "state": owner.get("state"),
            "country": owner.get("country"),
            "postcode": owner.get("postcode"),
        }

    def get_processed_trademark_data(self, query):
        """
        Fetch and process trademark data based on a query parameter.

        Parameters:
            query (str): The query string to search for trademarks.

        Returns:
            list: A list of structured trademark data.
        """
        raw_data = self.fetch_data(query)
        return self.process_data(raw_data)


class IPIdentifierSchema(BaseModel):
    """Schema for IP Identifier Tool input."""

    query: str = Field(
        ..., description="The invention or concept to analyze for IP rights"
    )
    context: Optional[str] = Field(
        None, description="Additional context about the invention"
    )


class IPIdentifierTool(BaseTool):
    """
    A tool for identifying potential intellectual property types for inventions.
    
    Args:
        llm (BaseChatModel): Language model for analysis
        
    Returns:
        None
    """
    name: str = "ip_identifier_tool"
    description: str = """
    Identify potential intellectual property types for inventions and concepts. This is the first tool that should be called to perform an IPR analysis.
    Use this tool when you need to:
    1. Identify which types of IP protection might be applicable
    2. Get initial guidance on IP strategy
    3. Understand what IP types to investigate further
    """
    args_schema: type[BaseModel] = IPIdentifierSchema
    llm: BaseChatModel
    file_parser: FileParser

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize the IP identifier tool."""
        file_parser = FileParser()
        super().__init__(llm=llm, file_parser=file_parser)

    def _run(
        self, query: str, context: Optional[str] = None, file_path: Optional[str] = None
    ) -> str:
        """Run the IP identifier tool."""
        logger.info("IPIdentifierTool Execution")
        try:
            # If query is a file path, parse it
            if file_path and file_path.endswith(".pdf"):
                with open(file_path, "rb") as file:
                    file_content = file.read()
                text = self.file_parser.parse_pdf_to_text(file_content)
                logger.info(f"Text extracted from PDF: {text[:100]}")
                query = text

            ip_types = self._identify_ip_types(query, context)
            logger.info(f"IP Types Identified: {json.dumps(ip_types, indent=2)}")
            return ip_types
        except Exception as e:
            logger.error(f"Error in IP identifier tool: {str(e)}")
            return f"Error identifying IP types: {str(e)}"

    async def _arun(
        self, query: str, context: Optional[str] = None, file_path: Optional[str] = None
    ) -> str:
        """Async implementation of the tool."""
        return self._run(query, context, file_path)

    def _identify_ip_types(self, query: str, context: Optional[str] = None) -> dict:
        """Identify potential IP types for the invention."""
        prompt = f"""You are a highly specialized Intellectual Property Rights (IPR) analyst. Your task is to identify
        the IPR status of an invention based on a provided checklist.

        Here is the invention idea you need to analyze:
        <invention>
        {query}
        </invention>

        Carefully review the invention and compare it against the following checklist:

        1. Does the invention involve a brand name, slogan, logo, or other distinctive methods to
        differentiate it from competitors?
        If yes, then it is trademarked.

        2. Does the invention include a newly designed machine, tool, medicine, or a novel process for
        creating one of these?
        If yes, it might have a utility patent and a design patent.

        3. Does the invention pertain to the design or creation of computer hardware or software?
        If yes, it might have a utility patent.

        4. Does the invention involve the development of new plant or seed varieties?
        If yes, it might have a plant patent.

        Analyze the invention idea against each point in the checklist. Consider all aspects of the
        invention and how they relate to different types of intellectual property protection.

        After your analysis, provide your conclusion in the following JSON format:

        <output>
        {{{{
            "utility_patent": boolean,
            "design_patent": boolean,
            "plant_patent": boolean,
            "trademark": boolean
        }}}}
        </output>

        Set each value to True if the invention likely qualifies for that type of protection, and False if
        it does not.

        Before providing the JSON output, explain your reasoning for each decision in a brief paragraph.
        Then, present the JSON output with your final assessment."""
        response = self.llm.invoke(
            [{"role": "system", "content": prompt, "temperature": 0.0}]
        )
        unparsed_json = response.content.replace("```json", "").replace("```", "")
        parsed_json = repair_json(unparsed_json)
        return json.loads(parsed_json)


class PatentNoveltySchema(BaseModel):
    """Schema for Patent Novelty Tool input."""

    query: str = Field(..., description="The invention to analyze for patent novelty")
    context: Optional[str] = Field(
        None, description="Additional context about the invention"
    )


class PatentNoveltyTool(BaseTool):
    """
    A tool for analyzing patent novelty and comparing with existing patents.
    
    Args:
        llm (BaseChatModel): Language model for analysis
        
    Returns:
        None
    """
    name: str = "patent_novelty_tool"
    description: str = """
    Analyze patent novelty and existing patents for inventions.
    Use this tool when:
    1. IPIdentifierTool returns true for utility_patent, design_patent or plant_patent
    2. You need to search for similar existing patents
    3. You need to analyze patent novelty
    4. You need to get recommendations for patent protection
    """
    args_schema: type[BaseModel] = PatentNoveltySchema
    llm: BaseChatModel
    patent_search: PatentSearch

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize the patent novelty tool."""
        patent_search = PatentSearch()
        super().__init__(llm=llm, patent_search=patent_search)

    def _run(self, query: str, context: Optional[str] = None) -> dict:
        """Run the patent novelty tool."""
        logger.info("PatentNoveltyTool Execution")
        try:
            logger.info(f"Analyzing patents for query: {query[:100]}...")
            patent_analysis = self._analyze_patents(query)
            logger.info(
                f"Patent Analysis Results: {json.dumps(patent_analysis, indent=2)}"
            )
            return patent_analysis
        except Exception as e:
            logger.error(f"Error in patent novelty tool: {str(e)}")
            return {"error": str(e)}

    async def _arun(self, query: str, context: Optional[str] = None) -> dict:
        """Async implementation of the tool."""
        return self._run(query, context)

    def get_main_topics(self, patent_text) -> dict:
        # prompt = get_prompt("agents/utils/ipr_agent.py", "get_main_topics")
        # prompt = prompt.format(patent_text = patent_text)
        prompt = f"""You are a highly specialized patent analyst with expertise in identifying and summarizing the core concepts of inventions. Your task is to analyze a given patent text and generate concise main topics or titles that capture the essence of the invention.

Here is the patent text you will be analyzing:

<patent_text>
{patent_text}
</patent_text>

Your goal:

*First, check if the patent text already mentions a title for the invention. If a clear and concise title is provided in the text, include it directly in the output.*

Then, regardless of whether a title exists, generate three additional variations of a title based on the core concept and essence of the invention. Each title should be distinct and professional, capturing different perspectives on the invention.

When generating titles:

Keep each title concise, ideally less than 7 words.
Highlight the unique value or novel aspect of the invention.
Use technical language appropriate for a patent, avoiding overly complex jargon.
Focus on the problem the invention solves or the improvement it offers over existing technologies.
Output:
If a title is found in the patent text:
<output>
{{{{
"title": [
"Provided Title",
"Generated Title 1",
"Generated Title 2",
"Generated Title 3"
]
}}}}
</output>

If no title is provided in the patent text:
<output>
{{{{
"title": [
"Generated Title 1",
"Generated Title 2",
"Generated Title 3"
]
}}}}
</output>

Your primary objective is to ensure the titles accurately reflect the invention's focus, whether by including the provided title or generating new ones. Each title should immediately convey the invention's core concept to someone familiar with the field."""
        response = self.llm.invoke(
            [{"role": "system", "content": prompt, "temperature": 0.0}]
        )

        unparsed_json = response.content
        parsed_json = unparsed_json.replace("```json", "").replace("```", "")
        output_json = repair_json(parsed_json)
        return json.loads(output_json)

    def _analyze_patents(self, query: str) -> dict:
        """Analyze patent novelty and existing patents."""
        try:
            patent_titles = self.get_main_topics(query)
            logger.info(f"Patent Titles: {str(patent_titles)}")
            similar_patents = self.patent_search.get_patents_info(
                patent_titles["title"][0:1]
            )
            return self._generate_recommendation(query, similar_patents)
        except Exception as e:
            logger.error(f"Error in patent analysis: {str(e)}")
            return {"error": str(e)}

    def _generate_recommendation(self, query: str, similar_patents: List[Dict]) -> dict:
        """Generate recommendations based on patent analysis."""
        #
        # prompt = get_prompt("agents/utils/ipr_agent.py", "_generate_recommendation")
        # prompt = prompt.format(query = query, similar_patents = similar_patents)
        prompt = f"""You are an AI assistant tasked with analyzing a product or innovation idea to determine its novelty.
        You will compare the given idea with summaries of similar patents and provide a detailed analysis.
        Follow these instructions carefully:

        1. First, you will be presented with the text of a product or innovation idea. Read it carefully and
        understand its key concepts, features, and potential applications.

        <idea_text>
        {query}
        </idea_text>

        2. Next, you will be given a list of similar patents in JSON format. Each patent entry contains a
        summary, patent ID, title, and patent link. Carefully review these patents and their summaries.

        <similar_patents>
        {similar_patents}
        </similar_patents>

        3. Analyze the novelty of the idea by comparing it with the provided similar patents. Consider the
        following aspects:
        a. Unique features or concepts in the idea
        b. Similarities and differences with existing patents
        c. Potential improvements or innovations over existing patents
        d. Any gapsor areas not covered by existing patents that the idea addresses

        4. Compare the idea with each similar patent, noting specific points of similarity or difference.
        Pay attention to the core concepts, methodologies, and applications.

        5. Prepare your analysis and format the output as a JSON object with the following structure:
        {{
        "Similar Patents": [
        {{
        "patent_id": "ID of the most similar patent",
        "title": "Title of the most similar patent",
        "patent_link": "Link to the most similar patent"
        "Detailed Analysis":{{
        "Similarities": "List of similarities between the idea and the similar patent under analysis",
        "Similarity Score": "High/Medium/Low",
        "Differences": "List of key differences between the idea and the similar patent under analysis",
        "Potential Improvements": "List of potential improvements or innovations over the similar patent under analysis",
        "Gaps": "List of gaps or areas not covered by the similar patent under analysis that the idea addresses"
        }}
        }}
        ],
        "Information Gap": "If you need more information to conduct a better novelty research, mention it here"
        }}

        6. In the "Detailed Analysis" section, provide a comprehensive evaluation of the idea's novelty.
        Include:
        a. Similarities and differences with the most relevant patent and a similarity score
        b. Potential areas of innovation or improvement over the most relevant patents
        c. Any gaps or areas not covered by existing patents that the idea addresses

        7. In the "Similar Patents" array, include the three most similar patents from the provided list. Use
        the exact patent_id, title, and patent_link as given in the input.

        8. Ensure your analysis is objective, thorough, and based solely on the information provided in the
        idea text and similar patents. Do not make assumptions or introduce external information.

        9. Write your complete response, including the JSON output, within <response> tags.

        Remember to maintain a professional and analytical tone throughout your response"""
        unparsed_json = self.llm.invoke(
            [{"role": "system", "content": prompt, "temperature": 0.0}]
        ).content
        if "<response>" in unparsed_json:
            unparsed_json = unparsed_json.split("<response>")[1].split("</response>")[0]
        parsed_json = unparsed_json.replace("```json", "").replace("```", "")
        try:
            return json.loads(parsed_json)
        except json.JSONDecodeError:
            # If parsing fails, use json_repair as fallback
            try:
                return json.loads(repair_json(parsed_json))
            except:
                # If all parsing fails, return a structured error response
                return {
                    "Detailed Analysis": "Error parsing response",
                    "Similar Patents": [],
                }


class TrademarkNoveltySchema(BaseModel):
    """Schema for Trademark Novelty Tool input."""

    query: str = Field(
        ..., description="The mark to analyze for trademark availability"
    )
    context: Optional[str] = Field(
        None, description="Additional context about the mark"
    )


class TrademarkNoveltyTool(BaseTool):
    """
    A tool for analyzing trademark availability and potential conflicts.
    
    Args:
        llm (BaseChatModel): Language model for analysis
        
    Returns:
        None
    """
    name: str = "trademark_novelty_tool"
    description: str = """
    Analyze trademark availability and potential conflicts.
    Use this tool when:
    1. IPIdentifierTool returns true for trademark or false for other IPR types
    2. You need to search for similar existing trademarks
    3. You need to analyze trademark distinctiveness
    4. You need to get recommendations for trademark protection
    """
    args_schema: type[BaseModel] = TrademarkNoveltySchema
    llm: BaseChatModel
    trademark_search: TrademarkSearch

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize the trademark novelty tool."""
        trademark_search = TrademarkSearch(
            "https://uspto-trademark.p.rapidapi.com/v1/trademarkSearch",
            {
                "x-rapidapi-key": config.RAPID_API_KEY,
                "x-rapidapi-host": "uspto-trademark.p.rapidapi.com",
            },
        )
        super().__init__(llm=llm, trademark_search=trademark_search)

    def _run(
        self, query: str, similar_trademarks: Optional[list] = None, **kwargs
    ) -> str:
        """Run the trademark analysis tool."""
        try:
            logger.info(f"[TrademarkAnalysisTool] Processing trademark data: {query}")
            if similar_trademarks is None:
                similar_trademarks = []

            analysis_results = self.run_analysis_for_keywords(query)

            return analysis_results
        except Exception as e:
            logger.error(f"Error in trademark analysis tool: {str(e)}")
            return f"Error analyzing trademark: {str(e)}"

    async def _arun(
        self, query: str, similar_trademarks: Optional[list] = None, **kwargs
    ) -> str:
        """Async implementation of the tool."""
        return self._run(query, similar_trademarks, **kwargs)

    def generate_keywords(self, query: str):
        """
        Generate keywords from the trademark data string by analyzing its key attributes
        and producing relevant search terms.

        Parameters:
            trademark_data_str (str): Trademark information provided as a string.

        Returns:
            list: A list of keywords derived from the trademark data.
        """
        # prompt = get_prompt("agents/utils/ipr_agent.py", "generate_keywords")
        # prompt = prompt.format(query = query)
        prompt = f"""Given the following trademark information, generate a list of relevant keywords for the trademark data:

        {query}

        The keywords should be derived from the title, proprietor name, owner, and address. 
        Provide only the top 2 most relevant keywords that are a single word and specific to the trademark data.
        Please return the keywords in the following format (at most 2 keywords):

        <START>['keyword1', 'keyword2']<END>
        <START> is the starting token and <END> is the ending token, you must include them in your response as substrings.

        The list should only contain words, with no commas or additional text."""
        response = self.llm.invoke(
            [{"role": "system", "content": prompt, "temperature": 0.0}]
        )

        keywords_str = response.content.strip()

        start_token = "<START>"
        end_token = "<END>"

        start_index = keywords_str.find(start_token)
        end_index = keywords_str.find(end_token)

        if start_index != -1 and end_index != -1:
            keywords_substring = keywords_str[
                start_index + len(start_token) : end_index
            ].strip()

            try:
                keywords = eval(keywords_substring)
                if isinstance(keywords, list):
                    return keywords
                else:
                    return []
            except (SyntaxError, NameError):
                return []
        else:
            return []

    def analyze_trademark(self, query: str, similar_trademarks=[]):
        """
        Analyze a trademark for distinctiveness and compare it with similar trademarks.
        """

        # prompt = get_prompt("agents/utils/ipr_agent.py", "analyze_trademark")
        # prompt = prompt.format(
        #     query=query,
        #     similar_trademarks=similar_trademarks
        # )
        prompt = f"""You are an AI trademark analyst with expertise in evaluating trademark applications for distinctiveness
and analyzing their similarity to existing trademarks. Below are the details of a trademark for analysis:

<Trademark Details>
{query}
</Trademark Details>

In addition, you are provided with a list of similar trademarks in JSON format:

<Similar Trademarks>
{similar_trademarks}
</Similar Trademarks>

Your task is to:
1. Evaluate the distinctiveness of the trademark based on:
    - Visual similarity: Does the title or description include design elements or symbols that resemble those in the similar trademarks?
    - Phonetic similarity: Do the names sound alike when spoken aloud, increasing the likelihood of confusion?
    - Conceptual similarity: Do the trademarks convey a similar idea, theme, or meaning?
2. Compare the trademark with the provided similar trademarks based on:
    - Consumer perception: How might the average consumer interpret these trademarks in a real-world context?
    - Overall impression: Does the trademark create an impression that is too similar to the listed similar trademarks, even if minor differences exist?
    - Market context: Are the goods or services offered under these trademarks in overlapping or related industries, or do they target the same audience?
3. Identify any overlaps or unique features that enhance or diminish the trademark's distinctiveness.
4. Assess potential risks, including:
    - Likelihood of confusion: Could consumers reasonably assume the trademarks are from the same source or affiliated?
    - Deceptive similarity: Does the trademark create a false association with existing trademarks, intentionally or unintentionally?
    - Legal challenges: How likely is it that the trademark will face opposition or litigation based on its similarity to existing trademarks?
5. Provide recommendations to improve the trademark's distinctiveness or mitigate risks.
6. Provide a final conclusion on whether the trademark is unique and distinctive.

Format your response as follows:
<response>
{{
    "Detailed Analysis": "Comprehensive analysis of the trademark's distinctiveness and comparisons, highlighting visual, phonetic, and conceptual aspects, consumer perception, and market context.",
    "Similar Trademarks": [
        {{
            "title": "Title of the most similar trademark",
            "serial_number": "Serial number of the most similar trademark",
            "description": "Description of the most similar trademark",
            "owner": "Owner of the most similar trademark",
            "Similarity Assessment": {{
                "Visual": "High/Medium/Low",
                "Phonetic": "High/Medium/Low",
                "Conceptual": "High/Medium/Low"
            }}
        }}
    ], 
    "Conclusion": "The trademark is unique and distinctive / The trademark is not unique and may face challenges."
}}
</response>

Ensure your analysis is objective, thorough, and professional. Highlight specific features that enhance
or diminish the trademark's distinctiveness. Avoid assumptions or introducing external information."""

        response = self.llm.invoke(
            [{"role": "system", "content": prompt, "temperature": 0.0}]
        )

        unparsed_json = response.content

        if "<response>" in unparsed_json:
            unparsed_json = unparsed_json.split("<response>")[1].split("</response>")[0]

        parsed_json = unparsed_json.strip().replace("```json", "").replace("```", "")

        try:
            return json.loads(parsed_json)
        except json.JSONDecodeError:
            try:
                return json.loads(repair_json(parsed_json))
            except Exception as e:
                return {
                    "Detailed Analysis": "Error parsing response",
                    "Similar Trademarks": [],
                }

    def run_analysis_for_keywords(self, trademark_data):
        """
        Generate keywords and run analysis for each keyword using the search tool.
        """
        keywords = self.generate_keywords(trademark_data)
        all_analysis_results = []

        for keyword in keywords:
            similar_trademarks = self.trademark_search.get_processed_trademark_data(
                keyword
            )
            similar_trademarks_json = json.dumps(similar_trademarks, indent=4)
            logger.info(f"Analyzing for keyword: {keyword}")
            analysis_result = self.analyze_trademark(
                trademark_data, similar_trademarks_json
            )
            all_analysis_results.append(
                {"keyword": keyword, "analysis": analysis_result}
            )
        return all_analysis_results


class IPRAgent:
    """
    Main agent class for coordinating IPR analysis using various tools.
    
    Args:
        llm (BaseChatModel): Language model for analysis
        temp_files (Optional[List[str]]): List of temporary files to process
        
    Returns:
        None
    """
    def __init__(self, llm: BaseChatModel, temp_files: Optional[List[str]] = []):
        if not isinstance(llm, BaseChatModel):
            raise ValueError("llm must be an instance of BaseChatModel")
        self.llm = llm
        self.temp_files = [f for f in temp_files if f.endswith(".pdf")]
        logger.info(f"ipr_agent temp_files: {self.temp_files}")
        self.file_parser = FileParser()
        # Initialize the tools
        self.ip_identifier_tool = IPIdentifierTool(llm)
        self.patent_novelty_tool = PatentNoveltyTool(llm)
        self.trademark_novelty_tool = TrademarkNoveltyTool(llm)

    def get_ipr_analysis_tool(self):
        """
        Creates a structured tool for IPR analysis.

        Returns:
            StructuredTool: Tool for analyzing IPR aspects of inventions.
        """

        agent_executor = self.create_ipr_analysis_agent()

        def analyze_ipr_content(query: str, context: Optional[str] = None):
            # Check if query is requesting analysis of a specific PDF
            pdf_files = [f for f in self.temp_files if f.endswith(".pdf")]
            if pdf_files:
                for pdf_file in pdf_files:
                    try:
                        # Use the FileParser directly to get text
                        text = self.file_parser.parse_pdf_to_text(
                            pdf_file
                        )  # Changed this line
                        if text:
                            logger.info(
                                f"Successfully extracted text from PDF: {pdf_file}"
                            )
                            query = (
                                f"Analyze the following content for IPR aspects: {text}"
                            )
                        else:
                            return {"error": "No text could be extracted from the PDF"}
                    except Exception as e:
                        return {"error": f"Error processing PDF file: {str(e)}"}

            chain_input = {"query": query, "temp_files": self.temp_files}
            if context:
                chain_input["context"] = context

            return agent_executor.invoke(chain_input)["output"]

        logger.info(f"temp_files: {self.temp_files}")

        return StructuredTool.from_function(
            name="ipr_analyst",
            func=analyze_ipr_content,
            description="""Tool for analyzing Intellectual Property Rights (IPR) status of inventions and documents.
            Can analyze both text descriptions and PDF documents for IPR implications.""",
        )

    def create_ipr_analysis_agent(self):
        """
        Creates an agent specifically for IPR analysis.

        Returns:
            AgentExecutor: Configured agent for IPR analysis.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a specialized Intellectual Property Rights (IPR) analysis assistant.
            If your final response is from Patent Novelty Tool or Trademark Novelty Tool, ensure that you cover all the sections of that response.""",
                ),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # Use the BaseTool implementations
        tools = [
            self.ip_identifier_tool,
            self.patent_novelty_tool,
            self.trademark_novelty_tool,
        ]

        agent = create_tool_calling_agent(self.llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools)


def format_ipr_response(response: str) -> None:
    """
    Format and print the IPR analysis response in a readable way.

    Args:
        response (dict): The response dictionary from the IPR analysis
    """
    try:
        # Extract and clean the output text
        output_text = response.get("output", "")
        logger.info(output_text)
        # Split the text into sections based on common patterns

    except Exception as e:
        logger.error(f"\nError formatting response: {str(e)}")
        logger.info("\nRaw response:")
        logger.info(str(json.dumps(response, indent=2)))


if __name__ == "__main__":
    logger.info("\n=== Starting IPR Analysis ===")
    # Initialize the IPR agent with GPT-4o model
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=config.OPENAI_API_KEY)
    logger.info("Initialized LLM and IPR Agent")

    # List of available PDF files
    # 20240384871-nn
    temp_files = ["data/Trademark_Spects.pdf"]
    ipr_agent = IPRAgent(llm, temp_files)

    # Get the IPR analysis tool
    ipr_tool = ipr_agent.get_ipr_analysis_tool()

    logger.info("\nAnalyzing PDF document...")
    results = ipr_tool.invoke(
        {
            "query": f"Analyze the IPR aspects of the uploaded document ",
            "temperature": 0.0,
        }
    )

    # Format and display results
    logger.info("\n=== Final Results ===")
    format_ipr_response(results)
