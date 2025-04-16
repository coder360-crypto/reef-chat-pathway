# Import required libraries
import os
import re
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import Google Drive and authentication related libraries
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseUpload
from fastapi import UploadFile
import io
import json
from config import IndexerConfig as Config
from pymongo import MongoClient


class DriveService:
    """
    Service class to handle Google Drive operations including file uploads and folder management
    """
    SCOPES = ["https://www.googleapis.com/auth/drive"]
    SERVICE_ACCOUNT_FILE = "credentials.json"

    def __init__(self):
        self.config = Config()
        self.credentials = service_account.Credentials.from_service_account_file(
            self.SERVICE_ACCOUNT_FILE, scopes=self.SCOPES
        )
        self.service = build("drive", "v3", credentials=self.credentials)

        # MongoDB connection
        self.client = MongoClient(self.config.DB_URL)
        self.db = self.client["default"]
        self.folder_mapping = self.db["drive_folder_mappings"]

        self.parent_folder_id = self.config.GDRIVE_LINK

        # Create default entry for test@pipe.ai if it doesn't exist
        test_mapping = self.folder_mapping.find_one({"gmail": "test@pipe.ai"})
        if not test_mapping:
            self.folder_mapping.insert_one(
                {"gmail": "test@pipe.ai", "folder_id": self.parent_folder_id}
            )

    def get_user_folder(self, gmail: str) -> str:
        """
        Get folder ID for user, create if doesn't exist
        
        Args:
            gmail (str): User's gmail address
            
        Returns:
            str: Folder ID associated with the user
        """
        mapping = self.folder_mapping.find_one({"gmail": gmail})
        if not mapping:
            folder_id = self.create_user_folder(gmail)
            self.folder_mapping.insert_one({"gmail": gmail, "folder_id": folder_id})
            return folder_id
        return mapping["folder_id"]

    def create_user_folder(self, gmail: str) -> str:
        """
        Create a new folder for user and share it with them
        
        Args:
            gmail (str): User's gmail address
            
        Returns:
            str: ID of the newly created folder
        """
        folder_metadata = {
            "name": f"{gmail}",
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [self.parent_folder_id],
        }

        folder = (
            self.service.files().create(body=folder_metadata, fields="id").execute()
        )

        folder_id = folder.get("id")

        # Share folder with user
        permission = {"type": "user", "role": "writer", "emailAddress": gmail}

        self.service.permissions().create(
            fileId=folder_id, body=permission, sendNotificationEmail=True
        ).execute()

        return folder_id

    async def upload_file(self, file: UploadFile, folder_id: str) -> dict:
        """
        Upload file to specific folder
        
        Args:
            file (UploadFile): File to be uploaded
            folder_id (str): ID of the folder where file will be uploaded
            
        Returns:
            dict: Contains file_id and drive_link of the uploaded file
        """
        file_content = await file.read()

        file_metadata = {"name": file.filename, "parents": [folder_id]}

        fh = io.BytesIO(file_content)
        media = MediaIoBaseUpload(fh, mimetype=file.content_type, resumable=True)

        file = (
            self.service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )

        file_id = file.get("id")
        return {
            "file_id": file_id,
            "drive_link": f"https://drive.google.com/file/d/{file_id}/view",
        }

    def _list_folders(self, parent_folder_id: str = None, depth: int = -1) -> list:
        """
        Recursively list all folders within a specified folder and their subfolders

        Args:
            parent_folder_id: ID of the parent folder (None uses default parent folder)
            depth: Maximum depth to traverse (-1 for unlimited)

        Returns: List of dictionaries containing folder info (id, name, path, subfolders)
        """
        if parent_folder_id is None:
            parent_folder_id = self.parent_folder_id

        if depth == 0:
            return []

        query = f"'{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"

        results = (
            self.service.files()
            .list(q=query, spaces="drive", fields="files(id, name)", pageSize=1000)
            .execute()
        )

        folders = []
        for folder in results.get("files", []):
            folder_info = {
                "id": folder["id"],
                "name": folder["name"],
                "subfolders": self._list_folders(
                    folder["id"], depth - 1 if depth > 0 else -1
                ),
            }
            folders.append(folder_info)

        return folders

    def get_folder_ids(self, parent_folder_id: str = None, depth: int = -1) -> list:
        """
        Get all folder IDs recursively including subfolders
        
        Args:
            parent_folder_id (str, optional): ID of the parent folder. Defaults to None
            depth (int, optional): Maximum depth to traverse. Defaults to -1
            
        Returns:
            list: List of all folder IDs found
        """
        folders = self._list_folders(parent_folder_id, depth)
        all_ids = []

        def extract_ids(folder_list):
            for folder in folder_list:
                all_ids.append(folder["id"])
                extract_ids(folder["subfolders"])

        extract_ids(folders)
        return all_ids


# Main execution block
if __name__ == "__main__":
    drive_service = DriveService()
    print(drive_service.get_folder_ids(parent_folder_id=""))
