# Import standard libraries
import os
import re
import sys
from pathway import Json

# Import pathway library for data processing
import pathway as pw
import logging

# Add parent directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom utility modules
from utils.file_parser import FileParser, SubDocumentParser
from utils.enhance_metadata import MetadataEnhancer
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DriveConnector:
    """
    Class responsible for interacting with Google Drive, fetching, and processing file content.

    Args:
        folder_id (str): The ID of the Google Drive folder to connect to
        mode (str, optional): Connection mode. Defaults to "streaming"
        refresh_interval (int, optional): Refresh interval in seconds. Defaults to 30

    Returns:
        None
    """

    def __init__(
        self, folder_id: str, mode: str = "streaming", refresh_interval: int = 30
    ):
        self.folder_id = folder_id
        self.table = self._drive_connector(
            folder_id=folder_id, mode=mode, refresh_interval=refresh_interval
        )

    def _drive_connector(
        self,
        folder_id: str,
        mode: str = "streaming",
        refresh_interval: int = 30,
    ):
        """
        Streams file content from Google Drive based on the folder ID.

        Args:
            folder_id (str): Google Drive folder ID.
            mode (str): Connection mode
            refresh_interval (int): Refresh interval in seconds

        Returns:
            pw.Table: The table containing file content from Google Drive.
        """

        table = pw.io.gdrive.read(
            object_id=folder_id,
            service_user_credentials_file=os.path.join(
                os.path.dirname(__file__), "credentials.json"
            ),
            mode=mode,
            refresh_interval=refresh_interval,
            with_metadata=True,
            
            
        )
        return table

    def get_parsed_table(self):
        """
        Processes the file data by extracting its type and content.

        Args:
            table (pw.Table): Table containing file metadata and data.

        Returns:
            pw.Table: Table with processed file content including metadata and parsed data.
        """

        # UDF to extract and process file content with metadata
        @pw.udf
        def get_file_content(metadata, data):
            file_parser = SubDocumentParser()
            byte_data = file_parser.parse_to_byte_array(data)

            file_name = str(metadata["name"])

            existing_metadata = str(metadata)
            existing_metadata = json.loads(existing_metadata)

            extracted_text_with_metadata = file_parser.obj_to_text(
                file_name, byte_data, existing_metadata=existing_metadata
            ).encode("utf-8")

            return extracted_text_with_metadata

        # Add processed data column to table
        result_table = self.table.with_columns(
            data=get_file_content(self.table._metadata, self.table.data),
        )

        # UDF to extract metadata from processed data
        @pw.udf
        def get_metadata(data):
            data = data.decode("utf-8")
            return json.loads(eval(data)[-1])

        # Add metadata column to processed table
        processed_table = result_table.with_columns(
            _metadata=get_metadata(result_table.data),
        )

        return processed_table
