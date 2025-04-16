import csv
import json
import os
import tempfile

import PyPDF2
from docx import Document


class FileExtractor:
    """
    Class responsible for extracting text from various file types (.docx, .pdf, .txt).

    Methods:
        extract_text_from_file(byte_data: bytes, file_extension: str) -> str
        extract_text_from_docx(byte_data: bytes) -> str
        extract_text_from_pdf(byte_data: bytes) -> str
        extract_text_from_txt(byte_data: bytes) -> str
        extract_text_from_csv(byte_data: bytes) -> str
        extract_text_from_json(byte_data: bytes) -> str
    """

    @staticmethod
    def extract_text_from_file(byte_data: bytes, file_extension: str) -> str:
        # Creates a temporary file with the given extension and writes byte data to it
        with tempfile.NamedTemporaryFile(
            suffix=file_extension, delete=False, mode="wb"
        ) as temp_file:
            temp_file.write(byte_data)
            temp_file_path = temp_file.name

        return temp_file_path

    @staticmethod
    def extract_text_from_docx(byte_data: bytes) -> str:
        """
        Extracts text from a .docx file.

        Args:
            byte_data (bytes): Byte data of the .docx file.

        Returns:
            str: Extracted text from the .docx file with paragraphs joined by newlines.
        """
        # Create temporary file
        temp_file_path = FileExtractor.extract_text_from_file(byte_data, ".docx")

        try:
            # Extract text from each paragraph
            doc = Document(temp_file_path)
            full_text = [para.text for para in doc.paragraphs]
            return "\n".join(full_text)
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)

    @staticmethod
    def extract_text_from_pdf(byte_data: bytes) -> str:
        """
        Extracts text from a PDF file.

        Args:
            byte_data (bytes): Byte data of the PDF file.

        Returns:
            str: Extracted text from the PDF file.
        """
        temp_file_path = FileExtractor.extract_text_from_file(byte_data, ".pdf")

        try:
            with open(temp_file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                full_text = [page.extract_text() for page in reader.pages]
            return "\n".join(full_text)
        finally:
            os.unlink(temp_file_path)

    @staticmethod
    def extract_text_from_txt(byte_data: bytes) -> str:
        """
        Extracts text from a .txt file.

        Args:
            byte_data (bytes): Byte data of the .txt file.

        Returns:
            str: Extracted text from the .txt file.
        """
        temp_file_path = FileExtractor.extract_text_from_file(byte_data, ".txt")

        try:
            with open(temp_file_path, "r", encoding="utf-8") as file:
                text = file.read()
            return text
        finally:
            os.unlink(temp_file_path)

    @staticmethod
    def extract_text_from_csv(byte_data: bytes) -> str:
        """
        Extracts text from a CSV file.

        Args:
            byte_data (bytes): Byte data of the CSV file.

        Returns:
            str: Extracted text from the CSV file.
        """
        temp_file_path = FileExtractor.extract_text_from_file(byte_data, ".csv")

        try:
            with open(temp_file_path, "r", encoding="utf-8") as file:
                csv_reader = csv.reader(file)
                full_text = [",".join(row) for row in csv_reader]
            return "\n".join(full_text)
        finally:
            os.unlink(temp_file_path)

    @staticmethod
    def extract_text_from_json(byte_data: bytes) -> str:
        """
        Extracts text from a JSON file.

        Args:
            byte_data (bytes): Byte data of the JSON file.

        Returns:
            str: Extracted text from the JSON file.
        """
        temp_file_path = FileExtractor.extract_text_from_file(byte_data, ".json")

        try:
            with open(temp_file_path, "r", encoding="utf-8") as file:
                json_data = json.load(file)
            return json.dumps(json_data, indent=2)
        finally:
            os.unlink(temp_file_path)
