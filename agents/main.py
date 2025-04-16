import asyncio
import json
import logging
import os
import sys
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, List, Literal, Optional, Union

import jwt
import requests
import uvicorn
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import services.carp_services as carp_service
import services.moray_services as moray_service
import services.squid_services as squid_service
from api.usecases import UseCasesRouter

# from api.usecases import UsecasesRouter
from config import AgentsConfig as Config
from fastapi import APIRouter
from router_utilities import analyze_query, evaluate_response

config = Config()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Explicitly set the level for router_utilities logger
router_logger = logging.getLogger("router_utilities")
router_logger.setLevel(logging.INFO)

# Set logging level for all modules
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)

MAX_CONCURRENT_REQUESTS = 5  # Adjust based on server capacity
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
request_queue = deque()


# Create a factory class for services
class ServiceFactory:
    @staticmethod
    def create_service(service_type: str, *args, **kwargs):
        services = {
            "squid": squid_service.SQUID,
            "moray": moray_service.MORAY,
            "carp": carp_service.CARP,
        }

        # Create service instance with all kwargs
        service = services[service_type](*args, **kwargs)
        return service


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Create files directory if it doesn't exist
        os.makedirs(os.path.join(os.path.dirname(__file__), "files"), exist_ok=True)

        # Only store the factory in app
        app.service_factory = ServiceFactory()
        app.request_semaphore = request_semaphore
        app.request_queue = request_queue

        logger.info("Service factory initialized successfully")
    except Exception as e:
        logger.error("Error in initializing Service Factory: %s", str(e))
    yield
    app.request_queue.clear()


app = FastAPI(lifespan=lifespan)


app_router = APIRouter()
usecases_router = UseCasesRouter(app_router)

# Include the usecases router
app.include_router(app_router, prefix="/use-cases")

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class AgentPayload(BaseModel):
    query: str
    context: list[str] = None
    agent: str = Literal["moray", "squid", "auto", "carp"]
    use_internet: bool = True
    llm: str = "gpt-4o"
    guardrails: bool = False
    selected_docs: Optional[List[str]] = None
    jwt_token: Optional[str] = None


class AgentResponse(BaseModel):
    result: Union[dict, list]
    files: Optional[list[str]] = None
    time_taken: Optional[float] = None
    agent: Optional[str] = None


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


def _validate_token(authorization: Optional[str] = Header(None)) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="No authorization token provided")

    try:
        # Remove 'Bearer ' prefix if present
        token = authorization.replace("Bearer ", "")
        # Verify and decode JWT - replace with your actual secret key and algorithm
        payload = jwt.decode(token, config.JWT_SECRET, algorithms=["HS256"])
        return payload.get("email")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def process_request(
    agent_payload: AgentPayload, files: List[UploadFile], app: FastAPI
):
    start_time = asyncio.get_event_loop().time()
    async with app.request_semaphore:
        temp_files = []
        try:
            agent_payload.agent = agent_payload.agent.strip()

            agent_payload.query = agent_payload.query.strip()
            guarded_query = agent_payload.query
            if agent_payload.guardrails:
                url = config.GUARDRAILS_API_URL
                response = requests.post(
                    url, json={"query": agent_payload.query}
                ).json()
                guarded_query = response.get("guarded_query", agent_payload.query)

                if not response.get("verdict"):
                    return AgentResponse(
                        result={
                            "result": "I apologize, but I cannot process your request as it appears to contain inappropriate language, vulgar content, or attempts at jailbreaking. Please rephrase your query in a more appropriate and professional manner."
                        },
                        time_taken=asyncio.get_event_loop().time() - start_time,
                        agent="clarification_response",
                    )

            logger.info("Guarded query: %s", guarded_query)
            agent_payload.query = guarded_query

            # Check if it's a conversational query first
            # Single analysis call to determine handling strategy
            analysis = await analyze_query(
                agent_payload.query, agent_payload.agent, agent_payload.llm
            )

            # Handle conversational queries directly
            if (
                analysis["agent"] == "conversational"
                or analysis["agent"] == "clarification_response"
            ):
                return AgentResponse(
                    result={"result": analysis["response"]},
                    time_taken=asyncio.get_event_loop().time() - start_time,
                    agent=analysis["agent"],
                )

            # Handle file uploads if present
            if files:
                for file in files:
                    # Create temporary file
                    temp_file_path = file.filename
                    content = await file.read()
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(content)
                    logger.info(f"Saved file to {temp_file_path}")
                    temp_files.append(temp_file_path)

            logger.info("Received query: %s", agent_payload.query)
            logger.info("Using LLM: %s", agent_payload.llm)

            context = agent_payload.context or []

            context = "\n".join(context)
            if context:
                logger.info("Context prepared: %s", context)
            else:
                logger.info("No context provided")

            # For agent-based processing, use the selected agent
            if agent_payload.agent == "auto":
                if analysis["agent"] == "moray":
                    agent_payload.agent = "moray"
                    service = app.service_factory.create_service("moray")
                    # Add file information to query if present
                    if temp_files:
                        guarded_query += "\n\nThese are the PDFs uploaded by the user to analyse and process:\n"
                        for file in temp_files:
                            guarded_query += f"- {file}\n"
                    result = service.process_user_query(
                        query=guarded_query,
                        context=context,
                        llm=agent_payload.llm,
                        temp_files=temp_files,
                        selected_docs=agent_payload.selected_docs,
                        jwt_token=agent_payload.jwt_token,
                        use_internet=agent_payload.use_internet,
                    )

                    is_valid, error_message = evaluate_response(
                        result, "moray", agent_payload.llm
                    )
                    if not is_valid:
                        logger.info(
                            f"Moray response inadequate: {error_message}, trying squid"
                        )
                        squid_service = app.service_factory.create_service(
                            "squid",
                            jwt_token=agent_payload.jwt_token,
                            selected_docs=agent_payload.selected_docs,
                        )
                        result = await squid_service.process_user_query(
                            guarded_query,
                            context,
                            agent_payload.llm,
                            agent_payload.use_internet,
                            temp_files,
                        )

                elif analysis["agent"] == "squid":
                    agent_payload.agent = "squid"
                    service = app.service_factory.create_service(
                        "squid",
                        jwt_token=agent_payload.jwt_token,
                        selected_docs=agent_payload.selected_docs,
                    )
                    # Add file information to query if present
                    if temp_files:
                        guarded_query += "\n\nThese are the PDFs uploaded by the user to analyse and process:\n"
                        for file in temp_files:
                            guarded_query += f"- {file}\n"
                    result = await service.process_user_query(
                        guarded_query,
                        context,
                        agent_payload.llm,
                        agent_payload.use_internet,
                        temp_files,
                    )
                    is_valid, error_message = evaluate_response(
                        result, "squid", agent_payload.llm
                    )
                    if not is_valid:
                        logger.info(
                            f"Squid response inadequate: {error_message}, trying Moray"
                        )
                        service = app.service_factory.create_service("moray")
                        result = service.process_user_query(
                            query=guarded_query,
                            context=context,
                            llm=agent_payload.llm,
                            temp_files=temp_files,
                            selected_docs=agent_payload.selected_docs,
                            jwt_token=agent_payload.jwt_token,
                            use_internet=agent_payload.use_internet,
                        )

            elif agent_payload.agent == "carp":
                agent_payload.agent = "carp"
                service = app.service_factory.create_service(
                    "carp",
                    llm=agent_payload.llm,
                    jwt_token=agent_payload.jwt_token,
                    selected_docs=agent_payload.selected_docs,
                )
                if temp_files:
                    guarded_query += "\n\nThese are the PDFs uploaded by the user to analyse and process:\n"
                    for file in temp_files:
                        guarded_query += f"- {file}\n"

                result = service.process_user_query(
                    query=guarded_query,
                    context=context,
                    temp_files=temp_files,
                )

            # Create and execute appropriate service
            elif analysis["agent"] == "moray":
                agent_payload.agent = analysis["agent"]
                service = app.service_factory.create_service("moray")
                # Add file information to query if present
                if temp_files:
                    guarded_query += "\n\nThese are the PDFs uploaded by the user to analyse and process:\n"
                    for file in temp_files:
                        guarded_query += f"- {file}\n"
                result = service.process_user_query(
                    query=guarded_query,
                    context=context,
                    llm=agent_payload.llm,
                    temp_files=temp_files,
                    selected_docs=agent_payload.selected_docs,
                    jwt_token=agent_payload.jwt_token,
                    use_internet=agent_payload.use_internet,
                )

            elif analysis["agent"] == "squid":
                agent_payload.agent = analysis["agent"]
                service = app.service_factory.create_service(
                    "squid",
                    jwt_token=agent_payload.jwt_token,
                    selected_docs=agent_payload.selected_docs,
                )
                # Add file information to query if present
                if temp_files:
                    guarded_query += "\n\nThese are the PDFs uploaded by the user to analyse and process:\n"
                    for file in temp_files:
                        guarded_query += f"- {file}\n"
                result = await service.process_user_query(
                    guarded_query,
                    context,
                    agent_payload.llm,
                    agent_payload.use_internet,
                    temp_files,
                )

            elif agent_payload.agent == "carp":
                agent_payload.agent = "carp"
                service = app.service_factory.create_service(
                    "carp",
                    llm=agent_payload.llm,
                    jwt_token=agent_payload.jwt_token,
                    selected_docs=agent_payload.selected_docs,
                )

            logger.info(f"Using agent: {agent_payload.agent}")

            # Cleanup temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

            files = []
            if agent_payload.agent == "moray":
                for step in result["results"]:
                    if "plan_and_schedule" in step:
                        for message in step["plan_and_schedule"]["messages"]:
                            if message.name == "compliance_checker":
                                try:
                                    file_path = message.response_metadata["file_path"]
                                    if os.path.exists(file_path):
                                        files.append(file_path)
                                except Exception as e:
                                    logger.error(
                                        "Tool failure detected. File path for compliance checker not found."
                                    )
                            elif message.name == "graph_tool":
                                try:
                                    file_path = message.response_metadata["image_path"]
                                    if os.path.exists(file_path):
                                        files.append(file_path)
                                except Exception as e:
                                    logger.error(
                                        "Tool failure detected. File path for graph tool not found."
                                    )
                            elif message.name == "equity_research_tool":
                                try:
                                    file_path = message.response_metadata["pdf_path"]
                                    if os.path.exists(file_path):
                                        files.append(file_path)
                                except Exception as e:
                                    logger.error(
                                        "Tool failure detected. File path for equity research tool not found."
                                    )
                            elif message.name == "esg_analyst":
                                try:
                                    file_path = message.response_metadata["pdf_path"]
                                    if os.path.exists(file_path):
                                        files.append(file_path)
                                except Exception as e:
                                    logger.error(
                                        "Tool failure detected. File path for esg analyst not found.",
                                        exc_info=True,
                                    )

            elif agent_payload.agent == "squid":
                task_flow_data = result.get("task_flow_data", {})
                specialist_tasks = task_flow_data.get("specialist_tasks", {})
                specialist_list = [
                    "compliance_specialist",
                    "legal_researcher",
                    "contract_analyst",
                ]
                for specialist in specialist_list:
                    specialist_data = specialist_tasks.get(specialist, {})
                    tool_usage = specialist_data.get("tool_usage", [])

                    for tool in tool_usage:
                        if tool.get("tool_name") == "compliance_checker":
                            try:
                                output = tool.get("output", "")
                                parsed_output = json.loads(output)
                                file_path = parsed_output.get("metadata", {}).get(
                                    "file_path", ""
                                )
                                if file_path and os.path.exists(file_path):
                                    files.append(file_path)
                                    logger.info(
                                        f"Added compliance checker file from {specialist}: {file_path}"
                                    )
                                    break
                                elif file_path:
                                    logger.warning(
                                        f"File path found but does not exist from {specialist}: {file_path}"
                                    )
                                else:
                                    logger.warning(
                                        f"File path not found in the compliance checker output from {specialist}."
                                    )
                            except json.JSONDecodeError as jde:
                                logger.error(
                                    f"Error decoding JSON output from {specialist}: {str(jde)}",
                                    exc_info=True,
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error processing compliance checker result from {specialist}: {str(e)}",
                                    exc_info=True,
                                )

            elif agent_payload.agent == "carp":
                pass

            time_taken = asyncio.get_event_loop().time() - start_time

            return AgentResponse(
                result=result,
                files=files,
                time_taken=round(time_taken, 2),
                agent=(
                    agent_payload.agent
                    if agent_payload.agent != "auto"
                    else analysis["agent"]
                ),
            )
        except Exception as e:
            # Cleanup temporary files in case of error
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            logger.error("Error processing request: %s", str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


async def process_request_with_retry(
    agent_payload: AgentPayload,
    files: List[UploadFile],
    app: FastAPI,
    max_retries: int = 3,
):
    for attempt in range(max_retries):
        try:
            result = await process_request(agent_payload, files, app)
            return result
        except HTTPException as he:
            # Re-raise HTTP exceptions immediately
            raise he
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed with exception: {str(e)}")

        if attempt < max_retries - 1:
            await asyncio.sleep(1 * (attempt + 1))

    raise HTTPException(status_code=500, detail="Max retries exceeded")


@app.get("/files/{filename}")
async def get_file(filename: str):
    file_path = Path("files") / filename
    if file_path.is_file():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")


@app.post("/query")
async def execute_agent(
    query: Annotated[str, Form()],
    agent: Annotated[str, Form()] = "moray",
    llm: Annotated[str, Form()] = "gpt-4o",
    guardrails: Annotated[bool, Form()] = False,
    context: Annotated[List[str] | None, Form()] = None,
    files: Annotated[List[UploadFile] | None, File()] = None,
    selected_docs: Annotated[List[str] | None, Form()] = None,
    background_tasks: BackgroundTasks = None,
    authorization: Optional[str] = Header(None),
    use_internet: Annotated[bool, Form()] = True,
):
    if not authorization:
        authorization = config.JWT_TOKEN
    gmail = _validate_token(authorization)
    selected_docs = (
        selected_docs[0].split(",") if selected_docs else []
    )  # Remove the first and last characters and split by comma and space
    logger.info(f"Gmail: {gmail}")
    logger.info(f"Selected docs: {selected_docs}")

    try:
        payload = AgentPayload(
            query=query,
            context=context or [],
            agent=agent,
            llm=llm,
            guardrails=guardrails,
            selected_docs=selected_docs,
            jwt_token=authorization,
            use_internet=use_internet,
        )

        result = await process_request_with_retry(payload, files or [], app)
        return result
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    config = Config()
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
