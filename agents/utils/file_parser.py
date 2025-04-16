# Import required libraries
import os
import sys

# Add parent directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

# Import custom file extractor
from utils.file_extractor import FileExtractor


class FileParser:
    """
    A class to handle parsing of different file formats into text.

    Attributes:
        extractor (FileExtractor): Instance of FileExtractor to handle file processing
    """
    def __init__(self):
        # Initialize file extractor instance
        self.extractor = FileExtractor()

    def parse_pdf_to_text(self, byte_data: bytes) -> str:
        """
        Parses PDF byte data into text using the FileExtractor.

        Args:
            byte_data (bytes): Raw bytes of the PDF file

        Returns:
            str: Extracted text content from the PDF
        """
        return self.extractor.extract_text_from_pdf(byte_data)
