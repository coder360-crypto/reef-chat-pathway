import os
import sys
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, File, UploadFile, Form, Header, HTTPException
from fastapi.responses import JSONResponse
from config import IndexerConfig as Config
from services.drive_service import DriveService
import jwt
import requests

# from core.indexer_service import IndexerService


class IndexerRouter:
    def __init__(self, router: APIRouter):
        self.router = router
        # self.indexer_service = indexer_service
        self.drive_service = DriveService()
        self._init_routes()
        self.config = Config()

    def _init_routes(self):
        self.router.add_api_route("/upload/", self.upload_pdf, methods=["POST"])
        self.router.add_api_route(
            "/initialize-storage/", self.initialize_storage, methods=["POST"]
        )
        self.router.add_api_route(
            "/get-user-doc-metadata/",
            self.get_user_doc_metadata,
            methods=["GET"],
        )
        self.router.add_api_route(
            "/generate-token/",
            self.generate_token,
            methods=["POST"],
        )

    def _validate_token(self, authorization: Optional[str] = Header(None)) -> str:
        if not authorization:
            raise HTTPException(
                status_code=401, detail="No authorization token provided"
            )

        try:
            # Remove 'Bearer ' prefix if present
            token = authorization.replace("Bearer ", "")
            payload = jwt.decode(token, self.config.JWT_SECRET, algorithms=["HS256"])
            return payload.get("email")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

    async def initialize_storage(self, authorization: Optional[str] = Header(None)):
        gmail = self._validate_token(authorization)
        folder_id = self.drive_service.get_user_folder(gmail)
        return {
            "status": "success",
            "folder_id": folder_id,
            "drive_link": f"https://drive.google.com/drive/folders/{folder_id}",
        }

    async def upload_pdf(
        self, file: UploadFile = File(...), authorization: Optional[str] = Header(None)
    ):
        gmail = self._validate_token(authorization)
        folder_id = self.drive_service.get_user_folder(gmail)

        # Upload to Google Drive
        drive_result = await self.drive_service.upload_file(file, folder_id)

        return {
            "status": "success",
            "drive_info": drive_result,
        }

    async def get_user_doc_metadata(self, authorization: Optional[str] = Header(None)):
        gmail = self._validate_token(authorization)
        folder_id = self.drive_service.get_user_folder(gmail)
        all_folder_ids = self.drive_service.get_folder_ids(folder_id)
        all_folder_ids.append(folder_id)
        deafault_list = [
            "19w71MDMazfq3xMSCZ5GwB1w52_K790Pm",
            "1v_Qi8qcIbH81KscHpAr_Efoar2wIoMGo",
        ]
        for default_folder_id in deafault_list:
            if default_folder_id not in all_folder_ids:
                all_folder_ids.append(default_folder_id)
        url = f"{self.config.VECTOR_DB_HOST}:{self.config.VECTOR_DB_PORT}/v1/inputs"

        response = requests.get(url)
        metadatas = []
        for metadata in response.json():
            if metadata["parents"][0] in all_folder_ids:
                metadatas.append(metadata)
        return JSONResponse(
            content={"metadatas": metadatas, "folder_ids": all_folder_ids}
        )

    async def generate_token(self, email: str = Form(...)):
        try:
            token = jwt.encode(
                {"email": email}, self.config.JWT_SECRET, algorithm="HS256"
            )
            return {"token": token}
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Token generation failed: {str(e)}"
            )
