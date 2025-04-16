import hashlib
import os
import sys

from motor.motor_asyncio import AsyncIOMotorClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging

import openai
from config import InterfaceConfig as Config

logger = logging.getLogger(__name__)

# Replace the existing template and vagueness_prompt with this new all-in-one prompt
combined_prompt = """
    You are a context imputing agent. You wil be given a query and a conversation history.
    You should attach all relevant context to the query and return the refined query.
    If no relevant context is found, return the query as is.

    Example 1:
        <context>
            <user>Give me the financial statements for Apple Inc. for the year 2023.</user>
            <bot>Sure, here are the financial statements for Apple Inc. for the year 2023....</bot>
        </context>
        <query>
            What are the revenue figures for this company?
        </query>
        <refined_query>
            What are the revenue figures for Apple Inc. for the year 2023?
        </refined_query>

    Example 2:
        <context>
            <user>Give me the revenue figures of Best Buy for the year 2023.</user>
            <bot>Sure, here are the revenue figures of Best Buy for the year 2023....</bot>
            <user>Give me the revenue figures of AMCOR for the year 2023.</user>
            <bot>Sure, here are the revenue figures of AMCOR for the year 2023....</bot>
        </context>
        <query>
            Compare the revenue figures of the two companies for the year 2023.
        </query>
        <refined_query>
            Sure, here are the revenue figures of Best Buy for the year 2023....
            Sure, here are the revenue figures of AMCOR for the year 2023....
            Compare the revenue figures of Best Buy and AMCOR for the year 2023.
        </refined_query>

    Here is the conversation history:
    <context>{context}</context>

    Here is the query:
    <query>{query}</query>

    Return the refined query in the format:
    <refined_query>{{refined_query}}</refined_query>
"""


class ConversationHandler:
    def __init__(
        self,
        db_name: str = "default",
        collection_name: str = "default",
        chat_memory_size: int = 5,
    ):

        self.conversations = {}
        self.config = Config()
        client = AsyncIOMotorClient(self.config.DB_URL)
        db = client[db_name]
        self.collection = db[collection_name]
        self.chat_mem_size = chat_memory_size

    async def process_query(
        self, query: str, uid: str, llm: str = "gpt-4o-mini"
    ) -> dict:
        """Combined method for vagueness check and query reformulation in one API call"""
        history = await self._get_conversation(uid, k=2)

        if history:
            history = "\n".join(
                [
                    f"Query: {entry['reformed_query']}\nResponse: {entry['response']}"
                    for entry in history
                ]
            )
        else:
            history = ""

        logger.info(f"History: {history}")

        prompt = combined_prompt.format(context=history, query=query)
        logger.info(f"Prompt: {prompt} \n\n")

        model = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
        response = model.chat.completions.create(
            model=llm,
            messages=[{"role": "system", "content": prompt}],
        )

        # Add error handling for JSON parsing
        try:
            content = response.choices[0].message.content.strip()

            result = content[
                content.find("<refined_query>")
                + len("<refined_query>") : content.find("</refined_query>")
            ].strip()

            return {
                "is_unclear": False,
                "clarification_response": "",
                "reformulated_query": result,
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing response: {e}")
            # Return a default response in case of error
            return {
                "is_unclear": True,
                "clarification_response": "I apologize, but I encountered an error processing your request. Could you please rephrase your question?",
                "reformulated_query": "",
            }

    # Hash function for chat ID
    def hash_chat_id(self, chat_id):
        """
        Hashes the chat ID using SHA-256.

        Args:
            chat_id (str): The chat ID to be hashed.

        Returns:
            str: The hexadecimal hash of the chat ID.
        """
        return hashlib.sha256(chat_id.encode()).hexdigest()

    # Store query, agent, and response in MongoDB
    async def store_chat_data(
        self, chat_id, agent, query, reformed_query, response, metadata
    ):
        """
        Searches for or creates a JSON document for the hashed chat ID in MongoDB,
        generates an agent response, and stores query, agent name, and response.

        Args:
            chat_id (str): The chat ID.
            query (str): The query input.
            agent (str): The name of the agent.
        """
        # Hash the chat ID
        hashed_id = self.hash_chat_id(chat_id)

        # Check if a JSON document for this chat ID already exists
        existing_doc = await self.collection.find_one({"chat_id": hashed_id})

        if not existing_doc:
            # If it doesn't exist, create a new document
            try:
                new_doc = {
                    "chat_id": hashed_id,
                    "history": [],  # Initialize an empty list to store chat history
                    "clarity_count": 0,
                }
                await self.collection.insert_one(new_doc)
                print(f"Created new JSON for chat ID: {chat_id} (Hash: {hashed_id})")
            except Exception as e:
                print(f"Error creating new JSON: {e}")
                return
        else:
            print(f"Found existing JSON for chat ID: {chat_id} (Hash: {hashed_id})")

        # Create the entry to be added
        chat_entry = {
            "agent": agent,
            "query": query,
            "reformed_query": reformed_query,
            "response": response,
            "metadata": metadata,
            # "timestamp": datetime.utcnow()
        }

        # Add the entry to the chat history
        try:
            await self.collection.update_one(
                {"chat_id": hashed_id}, {"$push": {"history": chat_entry}}
            )
        except Exception as e:
            print(f"Error storing chat entry: {e}")
            return

        print(f"Stored chat entry in JSON for chat ID: {chat_id}")

    async def increase_clarity_count(self, chat_id):
        try:
            hashed_id = self.hash_chat_id(chat_id)
            await self.collection.update_one(
                {"chat_id": hashed_id}, {"$inc": {"clarity_count": 1}}
            )
        except Exception as e:
            logger.error(f"Error increasing clarity count: {e}")

    async def reset_clarity_count(self, chat_id):
        try:
            hashed_id = self.hash_chat_id(chat_id)
            await self.collection.update_one(
                {"chat_id": hashed_id}, {"$set": {"clarity_count": 0}}
            )
        except Exception as e:
            logger.error(f"Error resetting clarity count: {e}")

    async def get_clarity_count(self, chat_id):
        try:
            hashed_id = self.hash_chat_id(chat_id)
            doc = await self.collection.find_one({"chat_id": hashed_id})
            return doc["clarity_count"] if doc else 0
        except Exception as e:
            logger.error(f"Error getting clarity count: {e}")
            return 0

    async def get_recent_chat_history(self, chat_id, limit=1):
        """
        Retrieves the most recent chat history entries for a given chat ID,
        returning only the reformed query and response fields.
        """
        hashed_id = self.hash_chat_id(chat_id)

        try:
            doc = await self.collection.find_one(
                {"chat_id": hashed_id}, {"history": {"$slice": -limit}}
            )
            if doc and "history" in doc:
                simplified_history = [
                    {
                        "reformed_query": entry["reformed_query"],
                        "response": entry["response"],
                    }
                    for entry in doc["history"]
                ]
                return simplified_history
            else:
                try:
                    new_doc = {
                        "chat_id": hashed_id,
                        "history": [],
                        "clarity_count": 0,
                    }
                    await self.collection.insert_one(new_doc)
                    logger.info(
                        f"Created new JSON for chat ID: {chat_id} (Hash: {hashed_id})"
                    )
                except Exception as e:
                    logger.error(f"Error creating new JSON: {e}")
                return []
        except Exception as e:
            logger.error(f"Error retrieving chat history: {e}")
            return []

    async def _get_conversation(self, uid: str, k: int = 1) -> list:
        return await self.get_recent_chat_history(uid, k)

    async def get_chat_memory(self, uid):
        chat_id = self.hash_chat_id(uid)
        try:
            temp = await self.collection.find_one({"chat_id": chat_id})
            if temp:
                return temp["chat_memory"]
            else:
                try:
                    new_doc = {
                        "chat_id": self.hash_chat_id(uid),
                        "history": [],
                        "clarity_count": 0,
                        "chat_memory": [],
                        "recent_index": 0,
                    }
                    await self.collection.insert_one(new_doc)
                    logger.info(f"Created new JSON for chat ID: {chat_id}")
                    return []
                except Exception as e:
                    logger.error(f"Error creating new JSON: {e}")
                    return []
        except Exception as e:
            logger.error(f"Error getting chat memory: {e}")
            return []

    async def add_to_database(
        self,
        uid: str,
        agent: str,
        query: str,
        reformed_query: str,
        response: str,
        metadata: dict,
    ) -> None:
        await self.store_chat_data(
            uid, agent, query, reformed_query, response, metadata
        )
