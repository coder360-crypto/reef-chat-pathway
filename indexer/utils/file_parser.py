# Import required libraries for file handling and hashing
import hashlib
import os
import tempfile
from pathlib import Path
import json

# Import libraries for data processing
import htmltabletomd
import unstructured_client
from diskcache import Cache
from unstructured_client.models import operations, shared
from utils.enhance_metadata import MetadataEnhancerForDoc
from config import IndexerConfig as Config
import logging


# Get the logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler("parser.log", mode="a"))


class FileParser:
    """Base class for parsing files into text format
    
    Args:
        None
        
    Returns:
        FileParser: Instance of FileParser class
    """
    def __init__(self):
        self.cache = Cache(".cache/parser-cache")
        self.config = Config()
        self.unstructured_client = unstructured_client.UnstructuredClient(
            api_key_auth=self.config.UNSTRUCTURED_API_KEY,
            server_url=self.config.UNSTRUCTURED_SERVER_URL,
        )

    def _get_cache_key(self, file_name: str, byte_data: bytes) -> str:
        """Generate a unique cache key based on file name and content
        
        Args:
            file_name (str): Name of the file
            byte_data (bytes): Binary content of the file
            
        Returns:
            str: Unique cache key
        """
        content_hash = hashlib.md5(byte_data).hexdigest()
        return f"{file_name}_{content_hash}"

    def obj_to_text(self, file_name: str, byte_data: bytes) -> str:
        """Convert file object to text using unstructured API
        
        Args:
            file_name (str): Name of the file
            byte_data (bytes): Binary content of the file
            
        Returns:
            str: Extracted text from the file
        """
        # Strip quotes if present
        file_name = file_name.strip('"')

        # Check cache
        cache_key = self._get_cache_key(file_name, byte_data)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for key: {cache_key}")
            if cached_result == "":
                logger.info(f"Cache hit for key: {cache_key} but empty")
            else:
                return cached_result

        logger.info(f"Processing new text for key: {cache_key}")

        # Use tempfile for safer temporary file handling
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(byte_data)
            temp_file_path = temp_file.name

        try:
            req = operations.PartitionRequest(
                partition_parameters=shared.PartitionParameters(
                    files=shared.Files(
                        content=open(temp_file_path, "rb"),
                        file_name=file_name,
                    ),
                    strategy=shared.Strategy.HI_RES,
                    chunking_strategy=shared.ChunkingStrategy.BASIC,
                    languages=["eng"],
                    split_pdf_page=True,  # If True, splits the PDF file into smaller chunks of pages.
                    split_pdf_allow_failed=True,  # If True, the partitioning continues even if some pages fail.
                    split_pdf_concurrency_level=15,  # Set the number of concurrent request to the maximum value: 15.
                ),
            )

            res = self.unstructured_client.general.partition(request=req)

            # Process the response
            elements = []
            for element in res.elements:
                if element["type"] == "table":
                    elements.append(
                        htmltabletomd.convert_table(element["metadata"]["text_as_html"])
                    )
                else:
                    elements.append(element["text"])

            result_text = "\n".join(elements)
            self.cache.set(cache_key, result_text)
            return result_text

        except Exception as e:
            logger.error(f"Error processing document {file_name}: {e}")
            raise
        finally:
            # Clean up temp file
            Path(temp_file_path).unlink(missing_ok=True)

    def parse_to_byte_array(self, byte_string: bytes) -> bytearray:
        """
        Converts a string representation of bytes (e.g., b'...') into a byte array.

        Args:
            byte_string (bytes): Input byte string.

        Returns:
            bytearray: Byte array representation.
        """
        return bytearray(byte_string)


class SubDocumentParser(FileParser):
    """Enhanced parser that splits documents into subdocuments and adds metadata
    
    Args:
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments
        
    Returns:
        SubDocumentParser: Instance of SubDocumentParser class
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata_enhancer = MetadataEnhancerForDoc()

    def obj_to_text(self, file_name: str, byte_data: bytes, **kwargs) -> str:
        """Convert file object to text using unstructured API
        
        Args:
            file_name (str): Name of the file
            byte_data (bytes): Binary content of the file
            
        Returns:
            str: Extracted text from the file
        """
        # Strip quotes if present
        file_name = file_name.strip('"')
        existing_metadata = kwargs.get("existing_metadata", None)

        # Check cache
        cache_key = self._get_cache_key(file_name, byte_data)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for key: {cache_key}")
            if cached_result == "":
                logger.info(f"Cache hit for key: {cache_key} but empty")
            else:
                return cached_result

        logger.info(f"Processing new text for key: {cache_key}")

        # Use tempfile for safer temporary file handling
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(byte_data)
            temp_file_path = temp_file.name

        try:
            req = operations.PartitionRequest(
                partition_parameters=shared.PartitionParameters(
                    files=shared.Files(
                        content=open(temp_file_path, "rb"),
                        file_name=file_name,
                    ),
                    strategy=shared.Strategy.HI_RES,
                    chunking_strategy=shared.ChunkingStrategy.BY_PAGE,
                    max_characters=6000,
                    new_after_n_chars=5000,
                    languages=["eng"],
                    split_pdf_page=True,  # If True, splits the PDF file into smaller chunks of pages.
                    split_pdf_allow_failed=True,  # If True, the partitioning continues even if some pages fail.
                    split_pdf_concurrency_level=15,  # Set the number of concurrent request to the maximum value: 15.
                ),
            )

            res = self.unstructured_client.general.partition(request=req)

            def html_to_md(text_as_html):
                try:
                    return htmltabletomd.convert_table(text_as_html)
                except Exception as e:
                    logger.error(f"Error converting HTML to MD: {e}")
                    return text_as_html

            page_based_list = []
            if file_name.endswith(".pdf"):
                current_page = 1
                current_text = ""
                for chunk in res.elements:
                    text = (
                        html_to_md(chunk["metadata"]["text_as_html"])
                        if chunk["type"] == "Table"
                        else chunk["text"]
                    )
                    if chunk["metadata"]["page_number"] == current_page:
                        current_text += text
                    else:
                        page_based_list.append(current_text)
                        current_page = chunk["metadata"]["page_number"]
                        current_text = text
                if current_text:
                    page_based_list.append(current_text)
            else:
                page_based_list = [
                    (
                        html_to_md(chunk["metadata"]["text_as_html"])
                        if chunk["type"] == "Table"
                        else chunk["text"]
                    )
                    for chunk in res.elements
                ]

            # merge into sublist of 6 pages with one page over
            merged_list = []
            for i in range(0, len(page_based_list), 6):
                merged_list.append("\n".join(page_based_list[i : i + 6]))

            logger.info(f"Merged list size: {len(merged_list)}")

            metadata = self.metadata_enhancer.enhance(merged_list, file_name)

            metadata["name"] = file_name
            if existing_metadata:
                metadata.update(existing_metadata)

            merged_list.append(json.dumps(metadata))
            merged_list_str = str(merged_list)

            # Cache the result
            self.cache.set(cache_key, merged_list_str)

            return merged_list_str

        except Exception as e:
            logger.error(f"Error processing document {file_name}: {e}")
            raise
        finally:
            # Clean up temp file
            Path(temp_file_path).unlink(missing_ok=True)
