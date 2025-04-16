import json
import logging
import os
import sys
from asyncio import Semaphore
from collections import deque
from contextlib import asynccontextmanager
from typing import Annotated, List, Literal, Optional

import aiohttp
import uvicorn
from config import InterfaceConfig as Config
from fastapi import BackgroundTasks, FastAPI, File, Form, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.conversation_handler import ConversationHandler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


config = Config()


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add these after the config setup
MAX_CONCURRENT_REQUESTS = 5  # Adjust based on your server capacity
request_semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
request_queue = deque()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.request_semaphore = request_semaphore
    app.reformulation_agent = ConversationHandler()
    yield
    # Cleanup queued requests on shutdown
    app.request_queue.clear()


app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


class AgentPayload(BaseModel):
    query: str
    context: list[str] = None
    agent: str = Literal["moray", "auto", "squid"]
    use_internet: bool = True
    llm: str = "gpt-4o"
    guardrails: bool = False
    selected_docs: Optional[List[str]] = None
    jwt_token: Optional[str] = None
    files: Annotated[List[UploadFile] | None, File()] = None


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


async def process_request(
    agent_payload: AgentPayload,
    app: FastAPI,
    background_tasks: BackgroundTasks,
    uid: str | None = None,
    useHistory: bool = True,
):
    original_query = agent_payload.query
    async with app.request_semaphore:
        try:
            if useHistory:
                query_result = await app.reformulation_agent.process_query(
                    agent_payload.query, uid, agent_payload.llm
                )

                # if query_result["is_unclear"]:
                #     logger.info(
                #         f"Query is unclear: {query_result['clarification_response']}"
                #     )
                #     background_tasks.add_task(
                #         app.reformulation_agent.add_to_database,
                #         uid,
                #         agent_payload.agent,
                #         agent_payload.query,
                #         agent_payload.query,
                #         query_result["clarification_response"],
                #         {},
                #     )
                #     return {
                #         "message": query_result["clarification_response"],
                #         "status": "needs_clarification",
                #     }

                reformulated_query = (
                    query_result["reformulated_query"] or agent_payload.query
                )

                logger.info(f"Reformulated query: {reformulated_query}")
            else:
                reformulated_query = agent_payload.query

        except Exception as e:
            logger.error("Error processing query: %s", str(e), exc_info=True)
            reformulated_query = agent_payload.query

        try:
            # Create form data with the correct line ending handling
            form_data = aiohttp.FormData(quote_fields=False)

            form_data.add_field("query", reformulated_query, content_type="text/plain")
            form_data.add_field("agent", agent_payload.agent, content_type="text/plain")

            form_data.add_field("llm", agent_payload.llm, content_type="text/plain")

            form_data.add_field(
                "guardrails", str(agent_payload.guardrails), content_type="text/plain"
            )

            if agent_payload.context:

                for ctx in agent_payload.context:

                    # Convert to JSON string to preserve formatting
                    ctx_str = (
                        json.dumps(ctx) if isinstance(ctx, (dict, list)) else str(ctx)
                    )
                    form_data.add_field("context", ctx_str)
                form_data.add_field("context", "End of context.\n")

            # if useHistory:
            #     chat_memory = await app.reformulation_agent.get_recent_chat_history(uid)

            #     mem_use_prompt = "\nGiven below is the memory from your previous conversations. This MAY OR MAY NOT be relevant to the current query. Make an informed decision using only the most relevant context and IGNORE the rest of the conversation history.If you do however find relevant context, consider it to be the ground truth that NEED NOT be verified.\n<conversation_history>\n"
            #     form_data.add_field("context", mem_use_prompt)
            #     for mem in chat_memory:
            #         # Convert to JSON string to preserve formatting
            #         mem_str = (
            #             json.dumps(mem) if isinstance(mem, (dict, list)) else str(mem)
            #         )
            #         form_data.add_field("context", mem_str)
            #     form_data.add_field("context", "</conversation_history>")

            # Add files if they exist
            if agent_payload.files:
                for file in agent_payload.files:
                    content = await file.read()
                    form_data.add_field(
                        "files",
                        content,
                        filename=file.filename,
                        content_type=file.content_type,
                    )

            # Add use_internet field to form data
            form_data.add_field(
                "use_internet",
                str(agent_payload.use_internet),
                content_type="text/plain",
            )

            # Use aiohttp for async HTTP request
            async with aiohttp.ClientSession() as session:
                logger.info("Sending request to agents API")
                async with session.post(
                    f"{config.AGENTS_API_URL}/query",
                    data=form_data,
                    headers=(
                        {"Authorization": agent_payload.jwt_token}
                        if agent_payload.jwt_token
                        else {}
                    ),
                ) as response:
                    logger.info(f"Received response with status: {response.status}")

                    if response.status != 200:
                        error_data = await response.json()
                        logger.error(f"API error response: {error_data}")
                        return {
                            "error": error_data.get("error", "Unknown API error"),
                            "status": "failed",
                            "place1": "",
                        }

                    response_data = await response.json()
                    logger.info("Successfully parsed response JSON")

                    if not response_data:
                        logger.error("Empty response data received")
                        return {"error": "Empty response from API", "status": "failed"}

                    result = response_data.get("result")
                    if not result:
                        logger.error(f"No result in response data: {response_data}")
                        return {"error": "No result in response", "status": "failed"}

                    # print(result)
                    # input("Continue?")

                    try:
                        if useHistory:
                            if "moray" in agent_payload.agent:
                                background_tasks.add_task(
                                    app.reformulation_agent.add_to_database,
                                    uid,
                                    agent_payload.agent,
                                    original_query,
                                    reformulated_query,
                                    result["results"][-1]["join"]["messages"][-1][
                                        "content"
                                    ],
                                    result,
                                )
                            elif "squid" in agent_payload.agent:
                                background_tasks.add_task(
                                    app.reformulation_agent.add_to_database,
                                    uid,
                                    agent_payload.agent,
                                    original_query,
                                    reformulated_query,
                                    result["task_flow_data"]["final_compilation"],
                                    result,
                                )
                            elif "carp" in agent_payload.agent:
                                background_tasks.add_task(
                                    app.reformulation_agent.add_to_database,
                                    uid,
                                    agent_payload.agent,
                                    original_query,
                                    reformulated_query,
                                    result["final_response"],
                                    result,
                                )
                    except Exception as e:
                        try:
                            background_tasks.add_task(
                                app.reformulation_agent.add_to_database,
                                uid,
                                agent_payload.agent,
                                original_query,
                                reformulated_query,
                                result["result"],
                                result,
                            )
                        except Exception as e:
                            logger.error(f"Error adding to database: {str(e)}")

                return response_data
        except aiohttp.ClientError as e:
            logger.error(f"Network error while processing request: {str(e)}")
            return {"error": f"Network error: {str(e)}", "status": "failed"}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            return {"error": "Invalid JSON response from API", "status": "failed"}
        except Exception as e:
            logger.error(
                f"Unexpected error processing request: {str(e)}", exc_info=True
            )
            return {"error": str(e), "status": "failed"}


@app.post("/query")
async def execute_agent(
    query: Annotated[str, Form()],
    agent: Annotated[str, Form()] = "moray",
    llm: Annotated[str, Form()] = "gpt-4o",
    guardrails: Annotated[bool, Form()] = False,
    context: Annotated[List[str] | None, Form()] = None,
    files: Annotated[List[UploadFile] | None, File()] = None,
    selected_docs: Annotated[List[str] | None, Form()] = None,
    uid: Annotated[str | None, Form()] = None,
    useHistory: Annotated[bool, Form()] = True,
    use_internet: Annotated[bool, Form()] = True,
    background_tasks: BackgroundTasks = None,
    authorization: Annotated[str | None, Header()] = None,
):
    try:
        # Update payload creation to include new fields
        payload = AgentPayload(
            query=query,
            context=context or [],
            agent=agent,
            llm=llm,
            guardrails=guardrails,
            files=files,
            selected_docs=selected_docs,
            use_internet=use_internet,
            jwt_token=authorization,
        )

        # Rest of the function remains the same
        result = await process_request(payload, app, background_tasks, uid, useHistory)
        return result
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg)
        return {"error": str(e), "status": "failed"}


if __name__ == "__main__":
    config = Config()
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
