# Import standard libraries
import json
import os
import sys

# Import third-party libraries
import requests

# Add parent directory to system path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import logging
from typing import List

# Import project dependencies
from config import AgentsConfig as Config
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from utils.file_parser import FileParser
from utils.pdf_highlighter import add_highlights

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ComplianceToolSchema(BaseModel):
    """
    Schema for the compliance checking tool parameters.
    
    Args:
        file_path (str): Path to the highlighted PDF file
        k (int): Number of documents to retrieve
    """
    file_path: str = Field(
        description="The path to the highlighted PDF file containing compliance violations. defaults to empty string."
    )
    k: int = Field(
        description="The number of documents to retrieve for compliance checking. defaults to 5."
    )


class ComplianceChecker:
    """
    A class to check document compliance against legal requirements.
    
    Args:
        llm: Language model instance
        temp_files (List[str], optional): List of temporary file paths
    """

    def __init__(self, llm, temp_files=None):
        self.config = Config()
        self.retriever_url = self.config.RETRIEVER_API_URL
        self.retriever_client = requests.Session()
        self.llm = llm
        self.temp_files = temp_files or []

    def check_document_compliance(self, file_path: str, k: int = 5) -> dict:
        """
        Analyzes a PDF document for compliance and returns the path to a highlighted version of the document.

        Args:
            file_path (str): The path to the PDF file to analyze out of the temp files.
            k (int, optional): The number of documents to retrieve for compliance checking. Defaults to 5.

        Returns:
            str: The path to the highlighted PDF file.
        """

        def retrieve_documents(query: str, k: int) -> List[dict]:
            response = self.retriever_client.post(
                self.retriever_url,
                json={
                    "query": query,
                    "k": k,
                    "metadata_filter": f"contains(parents, `{self.config.US_LAWS_FOLDER_ID}`)",
                },
            )
            return response.json()

        def get_title(chunk_content: str) -> str:
            prompt = f"""
            This is the starting content of a legal document:
            {chunk_content}

            Extract the title of the document from the content.
            """
            response = self.llm.invoke(
                input=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.content.strip()

        def check_contract_compliance(
            retrieved_documents: List[str], query: str, title: str
        ):
            """
            Checks for contract compliance by analyzing retrieved and context documents.

            Args:
                retrieved_documents (List[dict]): A list of retrieved document metadata and texts.
                documents (List[str]): A list of context document texts.
                query (str): The query used to retrieve the documents.
            Returns:
                List[dict]: A list of identified rule violations, if any.
            """
            if not retrieved_documents:
                logger.warning("No retrieved documents provided for compliance check.")
                return []

            retrieved_context = "\n".join(retrieved_documents)
            all_violations = []

            if retrieved_context:
                llm_prompt = f"""You are two agents Mr.Stringent and Mr.Lenient. 
                
                    Mr.Stringent is strict and points out all mistakes,
                    even if he is unsure, the meaning is ambiguous or if the issue is not malicious.

                    Mr.Lenient is more exact about his reasoning. He points out mistakes only if they are 
                    definitely illegal. He does not flag mistakes that may be ambiguous or where the intent to harm is absent.
                    
                    Both are now tasked with reviewing a section of a legal document. The legal document has certain
                    laws to follow, as mentioned below:

                    First, review the relevant laws pertaining to this section of the legal document:

                    <laws>
                    {retrieved_context}
                    </laws>

                    This is the title of the document that you are analyzing:
                    <title_of_document>
                    {title}
                    </title_of_document>
                    Assume that both parties are aware of this title and its implications

                    Now, examine the following section of the legal document:

                    <legal_document_section>
                    {query}
                    </legal_document_section>

                    To complete this task, both Mr. Stringent and Mr. Lenient will follow these steps:

                    1. Mr. Stringent will meticulously read through the provided section of the legal document and point out all potential mistakes, regardless of ambiguity or intent.
                    2. Mr. Lenient will carefully analyze the same section, identifying only those instances of language that are definitively illegal or harmful, avoiding any ambiguous interpretations.
                    3. For each instance of potentially malicious language identified by either agent:
                    a. Note the exact phrase or sentence
                    b. Specify which law it violates
                    c. Explain why it might be considered malicious or problematic

                    After your analysis, format your findings as a JSON object. Use the following structure:

                    <json_format>
                    {{"Stringent": [
                        {{"rule": "The specific rule or principle that could be violated",
                        "violating_text": "The exact text or phrase that causes the violation",
                        "explanation": "A concise explanation of why this may violates the rule or is legally problematic",
                        "harmfulness": "high, medium or low"}}
                        ],
                    "Lenient": [
                        {{"rule": "The specific rule or principle that is violated",
                        "violating_text": "The exact text or phrase that causes the violation",
                        "explanation": "A explanation of why this absolutely violates the rule and why are we sure about it",
                        "harmfulness": "high, medium or low"}}
                        ]
                    }}
                    </json_format>

                    Ensure that your JSON output is properly formatted and contains all the required information. If no
                    malicious language is found, return a json of the given format with empty values. Assume that misinterpretations are few.

                    Begin your analysis now, and provide your output in the specified JSON format within <json_output>
                    tags."""
                try:
                    response = self.llm.invoke(
                        input=[{"role": "system", "content": llm_prompt}],
                        temperature=0,
                    )

                    contract_compliance_json = response.content.strip()
                    start_index = contract_compliance_json.find("<json_output>") + len(
                        "<json_output>"
                    )
                    end_index = contract_compliance_json.find("</json_output>")
                    contract_compliance_json = contract_compliance_json[
                        start_index:end_index
                    ].strip()

                    if (
                        contract_compliance_json == "[]"
                        or contract_compliance_json == ""
                    ):
                        violations = []
                    else:
                        violations = json.loads(contract_compliance_json)

                    stringent_violations = []
                    medium_violations = []

                    for violation in violations["Stringent"]:
                        violation["user_id"] = "Stringent"
                        if violation["harmfulness"] == "high":
                            stringent_violations.append(violation)
                        elif violation["harmfulness"] == "medium":
                            medium_violations.append(violation)

                    for violation in violations["Lenient"]:
                        violation["user_id"] = "Lenient"

                    all_violations.extend(stringent_violations)
                    all_violations.extend(violations["Lenient"])

                    if len(all_violations) < 3:
                        all_violations.extend(medium_violations)

                except json.JSONDecodeError:
                    logger.error("Error parsing LLM response")
                except Exception as e:
                    logger.error(f"Error querying OpenAI API: {str(e)}")

            return all_violations

        # Main execution logic
        try:

            with open(file_path, "rb") as file:
                file_content = file.read()

            logger.info(f"File content: {file_content[:100]}")
            file_parser = FileParser()

            text = file_parser.parse_pdf_to_text(file_content)

            logger.info(f"Text: {text[:100]}")

            final_answer = []
            query = ""
            title = None

            split_text = text.split(".")

            for line in split_text:
                query += line + ". "
                if len(query) < 500 and line != split_text[-1]:
                    continue

                if not title:
                    title = get_title(query)

                documents = retrieve_documents(query, k)
                documents = [doc["text"] for doc in documents]
                # print(documents)
                compliance_judgement = check_contract_compliance(
                    documents, query, title
                )
                logger.info(f"Compliance judgement: {compliance_judgement}")
                final_answer.append((compliance_judgement, documents))

                query = ""

            highlight_data = []
            for answers in final_answer:
                for identified_violation in answers[0]:
                    highlight_data.append(
                        {
                            "content": identified_violation["violating_text"],
                            "suggested_content": identified_violation["explanation"],
                        }
                    )
            highlighted_file_path = add_highlights(highlight_data, file_path)
            return {
                "message": f"The file has been checked for compliance and the highlighted file is available at <filename>{highlighted_file_path}</filename>",
                "metadata": {"file_path": highlighted_file_path},
            }

            # return f"The file has been checked for compliance and the highlighted file is available at <filename>{highlighted_file_path}</filename>"

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return {"message": f"Error: {str(e)}"}

    def get_compliance_tool(self):

        # Create a wrapper function that doesn't rely on self
        @tool
        def compliance_check(file_path: str, k: int = 5) -> dict:
            """
            Analyzes a PDF document for compliance and returns the path to a highlighted version of the document.

            Args:
                file_path (str): The path to the PDF file to analyze.
                k (int, optional): The number of documents to retrieve for compliance checking. Defaults to 5.

            Returns:
                dict: Results of the compliance check including highlighted file path.
            """
            return self.check_document_compliance(file_path, k)

        model = ChatOpenAI(
            model="gpt-4o", api_key=Config().OPENAI_API_KEY, temperature=0
        )
        model = model.bind_tools([compliance_check])

        pdf_files = [f for f in self.temp_files if f.endswith(".pdf")]

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are a compliance assistant. You are given a query and a list of documents.
                Perform the compliance check on the relevant document, and format the response.
                If the query is not related to the documents, you should return a message indicating that the query is not related to the documents.
                
                Available documents for analysis: {pdf_files}
                
                Analyze documents thoroughly and provide detailed compliance reports.""",
                ),
                ("human", "{query}"),
            ]
        )

        chain = prompt | model

        def analyze_compliance(query: str) -> dict:
            response = chain.invoke({"query": query})
            tool_calls = response.tool_calls
            try:
                tool_call = tool_calls[0]
                tool_input = tool_call["args"].get("file_path")
            except Exception as e:
                return {"message": f"Error: File for compliance not available."}
            try:
                compliance_result = self.check_document_compliance(tool_input)
                return compliance_result
            except Exception as e:
                logger.error(f"Error analyzing compliance: {str(e)}")
                return {"message": f"Error analyzing compliance: {str(e)}"}

        return StructuredTool.from_function(
            name="compliance_checker",
            func=analyze_compliance,
            description="Analyzes text or documents for legal and regulatory compliance issues.",
            return_direct=True,
        )


if __name__ == "__main__":

    # Initialize the language model (replace with actual initialization if needed)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=Config().OPENAI_API_KEY)

    print("Starting compliance check")

    # Create an instance of ComplianceChecker
    compliance_checker = ComplianceChecker(llm)

    # Define a test PDF file path (replace with an actual file path)
    test_file_path = "nda.pdf"

    # Call the check_document_compliance method
    result = compliance_checker.check_document_compliance(test_file_path, k=5)

    # Print the result
    print("Compliance Check Result:", result)
