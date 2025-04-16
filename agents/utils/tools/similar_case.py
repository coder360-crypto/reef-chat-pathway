# Import required libraries
import json
import os
import sys

# Add parent directories to system path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import dependencies
import logging
import random
from json.decoder import JSONDecodeError
from typing import Dict, List

from config import AgentsConfig as Config
from langchain.base_language import BaseLanguageModel
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from similar_case_agent.ik import IKApi
from utils.file_parser import FileParser

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
config = Config()


class SimilarCaseToolSchema(BaseModel):
    """Schema for the similar case tool response"""
    links: List[str] = Field(description="The links to the similar cases.")


prompt = """
You are a legal expert. You are given a case description. You need to come up with phrases that can be used to find similar cases.
For each phrase, provide reasoning for the choice of phrase.

**Output JSON Format:**
{{
    "phrases": ["phrase1", "phrase2", "phrase3"],
    "reasoning": "Reasoning for the choice of phrases"
}}

**Guidelines:**
- The phrases should be unique and not overlap with each other.
- The phrases should be relevant to the case description.
- If the document does seem like a case description, give an empty list for phrases


<case_description>
{case_description}
</case_description>
"""

similarity_prompt = """
You are a legal expert. You are given a case description and a potential precedent. You need to judge whether or not \
the precedent will be helpful in the case. Explain your reasoning.

**Output JSON Format:**
{{
    "helpful": (boolean),
    "reasoning": (string) "Reasoning for the choice of phrases"
}}

**Case Description:**
{case_description}

**Precedent:**
{case_text}

**Guidelines:**
- A precedent is helpful if it discusses a similar issue as the case description.
- A precedent is helpful only if the same side of the argument is being made in the case.
- A precedent is helpful only if it can be used to support the case.
- The reasoning should be concise and to the point.
"""


class SimilarCaseAgent:
    """
    Agent for finding and analyzing similar legal cases
    """
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", api_key=config.OPENAI_API_KEY)
        self.scraper = IKApi(token=config.IK_API_KEY)

    def get_queries(self, case_description):
        """
        Generate search queries based on case description
        
        Args:
            case_description: Description of the case to analyze
            
        Returns:
            List of search phrases
        """
        messages = [
            {
                "role": "system",
                "content": prompt.format(case_description=case_description),
            }
        ]
        response = self.llm.invoke(messages)

        try:
            content = json.loads(
                response.content.replace("```json", "").replace("```", "").strip()
            )
        except JSONDecodeError:
            logger.error(f"Failed to parse JSON response, retrying: {response.content}")
            # Retry once
            response = self.llm.invoke(messages)
            content = json.loads(
                response.content.replace("```json", "").replace("```", "").strip()
            )

        logger.info(f"Phrases: {content['phrases']}")
        return content["phrases"]

    def get_similar_cases_ikanoon(self, case_description, num_docs=10):
        """
        Fetch similar cases from IKanoon
        
        Args:
            case_description: Description of the case
            num_docs: Number of documents to retrieve
            
        Returns:
            List of similar cases with their details
        """
        queries = self.get_queries(case_description)
        cases = []
        urls = set()
        for query in queries[:3]:
            ret = self.scraper.scrape_document_details(
                query, doctypes="judgments", max_documents=5
            )
            if ret["status"] == "success" and len(ret["data"]) > 0:
                for case in ret["data"]:
                    if case["url"] not in urls:
                        cases.append(case)
                        urls.add(case["url"])

        return random.sample(cases, min(num_docs, len(cases)))

    def judge_similarity_ikanoon(self, case_description, docs):
        helpful_docs = []
        not_helpful_docs = []
        for doc in docs:
            messages = [
                {
                    "role": "system",
                    "content": similarity_prompt.format(
                        case_description=case_description,
                        case_text=doc["background"][:10000]
                        + "\n"
                        + doc["conclusion"][:10000],
                    ),
                }
            ]
            response = self.llm.invoke(messages)

            try:
                content = json.loads(
                    response.content.replace("```json", "").replace("```", "").strip()
                )
            except JSONDecodeError:
                logger.error(
                    f"Failed to parse JSON response, retrying: {response.content}"
                )
                # Retry once
                response = self.llm.invoke(messages)
                content = json.loads(
                    response.content.replace("```json", "").replace("```", "").strip()
                )

            doc["reasoning"] = content["reasoning"]
            logger.info(doc["url"])
            logger.info(content["reasoning"])
            if content["helpful"]:
                helpful_docs.append(doc)
            else:
                not_helpful_docs.append(doc)
        return helpful_docs, not_helpful_docs


class SimilarCaseToolHelper:
    """
    Helper class to create and manage the similar case search tool
    
    Args:
        llm: Language model instance
    """
    def __init__(self, llm):
        self.llm = llm

    @staticmethod
    @tool
    def get_similar_cases_ikanoon(file_path: str) -> List[Dict]:
        """
        Analyzes a PDF document and returns similar cases under the Indian Law.

        Args:
            file_path (str): The path to the PDF file to analyze.
            num_docs (int, optional): The number of documents to retrieve. Defaults to 5.

        Returns:
            List[Dict]: A list of dictionaries containing the similar cases.
        """
        agent = SimilarCaseAgent()
        with open(file_path, "rb") as file:
            file_content = file.read()
        case_description = FileParser().parse_pdf_to_text(file_content)
        docs = agent.get_similar_cases_ikanoon(case_description)
        helpful_docs, not_helpful_docs = agent.judge_similarity_ikanoon(
            case_description, docs
        )
        ret = []
        for doc in helpful_docs:
            ret.append(
                {
                    "url": doc["url"],
                    "title": doc["doc_title"],
                    "reasoning": doc["background"][:800]
                    + "\n"
                    + doc["background"][-800],
                    "analysis": doc["reasoning"],
                }
            )

        return ret

    def get_similar_case_tool(self, temp_files: list):
        self.temp_files = temp_files

        model = self.llm
        model = model.bind_tools([self.get_similar_cases_ikanoon])

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are a legal expert. You are given a list of documents. Based on the query, search for the relevant cases and provide \
                        the links to the cases.
            Available documents for analysis: {[f for f in self.temp_files if f.endswith(".pdf")]}
            
            Analyze documents thoroughly and provide detailed compliance reports.""",
                ),
                ("human", "{query}"),
            ]
        )

        chain = prompt | model

        def get_similar_cases(query: str) -> dict:
            response = chain.invoke({"query": query})
            print(response)
            tool_calls = response.tool_calls
            try:
                tool_call = tool_calls[0]
                tool_input = tool_call["args"].get("file_path")
            except Exception as e:
                return {"message": f"Error: File for compliance not available."}
            try:
                case_info = self.get_similar_cases_ikanoon(tool_input)
                if len(case_info) == 0:
                    raise Exception("No similar cases found")
                message = f"The similar cases based on the given file have been found. The links lead you to the Indian Kanoon website. \
                    Follow the links for more information about the cases. Here is the document analysiis: \n{str(case_info)}"
                return {
                    "message": message,
                    "metadata": {"links": [doc["url"] for doc in case_info]},
                }
            except Exception as e:
                logger.error(f"Error in similar cases: {str(e)}", exc_info=True)
                return {"message": f"Error: {str(e)}"}

        return StructuredTool.from_function(
            name="similar_case_finder",
            func=get_similar_cases,
            description="Searches for similar previous cases judged under the Indian Law.",
        )


if __name__ == "__main__":
    # Create and test the similar case tool
    my_tool = SimilarCaseToolHelper(
        llm=ChatOpenAI(model="gpt-4o-mini", api_key=config.OPENAI_API_KEY)
    ).get_similar_case_tool(temp_files=["dummy.pdf"])
    print(my_tool.invoke({"query": "Get similar cases to the given case"}))
