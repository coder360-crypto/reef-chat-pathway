# Import required libraries
import os
import sys

# Add parent directory to system path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import logging

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader

# Initialize logger
logger = logging.getLogger(__name__)


class PDFQuerySchema(BaseModel):
    """Schema for PDF query tool input.
    
    Args:
        pdf_name (str): Name of the PDF file to read
    """

    pdf_name: str = Field(..., description="Name of the PDF file to read")


class PDFQueryTool(BaseTool):
    """Tool for querying PDF documents.
    
    Args:
        name (str): Name of the tool
        description (str): Description of the tool's functionality
        args_schema (BaseModel): Schema for tool's input validation
    
    Returns:
        PDFQueryTool: An instance of the PDF query tool
    """

    name: str = "pdf_query_tool"
    description: str = """
    This tool is used to query a PDF document.
    
    Do NOT use this tool if:
    - No PDF documents have been uploaded yet
    - You need to analyze a new document that hasn't been uploaded
    """
    args_schema: type[BaseModel] = PDFQuerySchema

    def _run(self, pdf_name: str) -> str:
        """Run the PDF read tool.
        
        Args:
            pdf_name (str): Name of the PDF file to read
            
        Returns:
            str: Extracted text from the PDF file (first 10 pages)
        """
        try:
            # Remove .pdf extension if present
            if ".pdf" in pdf_name:
                pdf_name = pdf_name.replace(".pdf", "")
            # Initialize empty text string
            text = ""
            # Create PDF reader object
            pdf_reader = PdfReader(f"{pdf_name}.pdf")
            # Get total number of pages
            number_of_pages = len(pdf_reader.pages)
            # Extract text from first 10 pages (or less if PDF is shorter)
            for i in range(min(10, number_of_pages)):
                page = pdf_reader.pages[i]
                text += page.extract_text()
            return text
        except Exception as e:
            # Log and return error message if PDF reading fails
            logger.error(f"Error in PDF read tool: {str(e)}")
            return f"Error reading PDF: {str(e)}"

    async def _arun(self, pdf_name: str) -> str:
        """Async implementation of the tool.
        
        Args:
            pdf_name (str): Name of the PDF file to read
            
        Returns:
            str: Extracted text from the PDF file
        """
        return self._run(pdf_name)


# Main execution block
if __name__ == "__main__":
    # Initialize the PDF query tool
    pdf_tool = PDFQueryTool()

    # Example usage
    try:
        # Replace with your PDF filename
        sample_pdf = "dummy.pdf"

        # Query the PDF
        result = pdf_tool.run(sample_pdf)

        print("PDF Content (first 10 pages):")
        print("-" * 50)
        print(result[:100])

    except Exception as e:
        print(f"An error occurred: {str(e)}")
