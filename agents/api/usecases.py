import os
import sys

sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import logging
from typing import List, Optional

from config import AgentsConfig as Config
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from langchain_community.chat_models import ChatOpenAI
from services.moray_services import MORAY
from utils.equity_generation_utils.equity_research_generator import EquityResearchTool
from utils.financial_generation_utils.financial_report_generator import ReportGenerator

config = Config()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UseCasesRouter:
    def __init__(self, router: APIRouter):
        self.router = router
        self.setup_routes()

        # Initialize the equity research tool with proper LLM
        self.equity_research_tool = EquityResearchTool()
        self.financial_report_generator = ReportGenerator()

    def setup_routes(self):
        @self.router.post("/equity-research")
        async def generate_equity_research_report(query: str = Form(...)):
            try:
                # Extract company name from query using properly formatted messages
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    api_key=config.OPENAI_API_KEY,
                    base_url=config.OPENAI_API_BASE,
                )
                messages = [
                    {
                        "role": "system",
                        "content": "Extract only the company name from the given query. Return just the company name.",
                    },
                    {"role": "user", "content": query},
                ]
                response = await llm.ainvoke(messages)

                company_name = response.content.strip()
                logger.info(f"Company name extracted from query: {company_name}")

                result = self.equity_research_tool._run(company_name=company_name)
                return result
            except Exception as e:
                logger.error(f"Error in equity research: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/financial-report")
        async def generate_financial_report(query: str = Form(...)):
            try:
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    api_key=config.OPENAI_API_KEY,
                    base_url=config.OPENAI_API_BASE,
                )
                # Extract company name from query using properly formatted messages
                messages = [
                    {
                        "role": "system",
                        "content": "Extract only the company name from the given query. Return just the company name.",
                    },
                    {"role": "user", "content": query},
                ]
                response = await llm.ainvoke(input=messages)
                company_name = response.content.strip()
                logger.info(f"Company name extracted from query: {company_name}")
                result = self.financial_report_generator.generate_report(company_name)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/compliance-check")
        async def check_compliance(
            file: UploadFile = File(...), query: str = Form(...)
        ):
            # save file to temp folder
            file_path = file.filename
            with open(file_path, "wb") as f:
                f.write(await file.read())

            try:
                query = f"Strictly use Compliance Checker on the following file path: {file_path} to answer the following query: {query}"

                moray_service = MORAY()
                result = moray_service.process_user_query(
                    query=query, temp_files=[file_path], jwt_token=config.JWT_TOKEN
                )
                pdf_path = ""
                for step in result["results"]:
                    if "plan_and_schedule" in step:
                        for message in step["plan_and_schedule"]["messages"]:
                            if message.name == "compliance_checker":
                                logger.info(f"Message: {str(message)}")
                                file_path = message.response_metadata["file_path"]
                                if os.path.exists(file_path):
                                    pdf_path = file_path
                                    break

                return {
                    "message": result,
                    "metadata": {
                        "pdf_path": pdf_path,
                    },
                }
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)

        @self.router.post("/ipr-agent")
        async def ipr_agent(
            file: Optional[UploadFile] = File(None),
            query: str = Form(...),
        ):
            if file:
                # save file to temp folder
                file_path = file.filename
                with open(file_path, "wb") as f:
                    f.write(await file.read())
            else:
                file_path = None

            try:
                query = f"Strictly use IPR Analyst to answer the following query, Do not use any other tools: {query}"

                moray_service = MORAY()
                response = moray_service.process_user_query(
                    query=query,
                    temp_files=[file_path] if file_path else [],
                    jwt_token=config.JWT_TOKEN,
                )

                return {"message": response}
            finally:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)

        @self.router.post("/valuation-agent")
        async def valuation_agent(query: str = Form(...)):
            query = f"The use wants to perform valuation analysis of the company, therefore STRICTLY  use valuation_tool to answer the following query, Do not use any other tools: {query}"

            moray_service = MORAY()
            response = moray_service.process_user_query(
                query=query, temp_files=[], jwt_token=config.JWT_TOKEN
            )

            return {"message": response}

        @self.router.post("/similar-case-agent")
        async def similar_case_agent(
            file: UploadFile = File(...),
            query: str = Form(...),
        ):
            # save file to temp folder
            file_path = file.filename
            with open(file_path, "wb") as f:
                f.write(await file.read())

            try:
                query = f"The user wants to perform find the similar case related to the , therefore STRICTLY use similar_case_tool to answer the following query, Do not use any other tools: {query}"

                moray_service = MORAY()
                response = moray_service.process_user_query(
                    query=query, temp_files=[file_path], jwt_token=config.JWT_TOKEN
                )
                return {
                    "message": response,
                }
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)

        @self.router.post("/esg-comparison-agent")
        async def esg_comparison_agent(
            files: List[UploadFile] = File(...),
            query: str = Form(...),
        ):
            # save file to temp folder
            file_paths = []
            for file in files:
                file_path = file.filename
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                file_paths.append(file_path)

            try:
                query = f"The user wants to perform ESG comparison analysis, therefore STRICTLY use ESG Analyst to answer the following query, Do not use any other tools: {query}"

                moray_service = MORAY()
                response = moray_service.process_user_query(
                    query=query, temp_files=file_paths, jwt_token=config.JWT_TOKEN
                )

                pdf_path = ""

                for step in response["results"]:
                    if "plan_and_schedule" in step:
                        for message in step["plan_and_schedule"]["messages"]:
                            if message.name == "esg_analyst":
                                try:
                                    file_path = message.response_metadata["pdf_path"]
                                    if os.path.exists(file_path):
                                        pdf_path = file_path
                                        break
                                except Exception as e:
                                    logger.error(
                                        "Tool failure detected. File path for esg analyst not found.",
                                        exc_info=True,
                                    )

                return {
                    "message": response,
                    "metadata": {
                        "pdf_path": pdf_path,
                    },
                }
            finally:
                for file_path in file_paths:
                    if os.path.exists(file_path):
                        os.remove(file_path)

        # @self.router.post()

        def __call__(self):
            return self.router
