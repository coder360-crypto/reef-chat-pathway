# Import standard libraries
import hashlib
import json
import logging
import os
import sys
import anthropic
import diskcache
import pathway as pw
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

# Add parent directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
from tqdm import tqdm

from config import IndexerConfig as Config

config = Config()


class ContextualSplitter(pw.UDF):
    """A text splitter that adds contextual information to each chunk using Claude.
    
    Args:
        chunk_size (int): Size of each text chunk
        chunk_overlap (int): Number of characters to overlap between chunks
        separators (list[str]): List of separators to use for splitting
        api_key (str): Anthropic API key
        cache_dir (str): Directory to store the disk cache
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] = None,
        api_key: str = None,
        cache_dir: str = ".cache/contextual-splitter",
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        # Initialize LLM client
        self.api_key = api_key or config.ANTHROPIC_API_KEY
        print("ANTHROPIC_API_KEY: %s", self.api_key)
        self.client = anthropic.Client(api_key=self.api_key)
        self.model = "claude-3-haiku-20240307"

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

        # Get the logger
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.FileHandler("splitter.log", mode="a"))
        # Initialize disk cache
        self.cache = diskcache.Cache(cache_dir)

        self.token_counts = {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_creation": 0,
        }

    # Generate cache key from input text
    def _get_cache_key(self, txt: str) -> str:
        """Generate a cache key based on input text."""
        return hashlib.md5(txt.encode()).hexdigest()

    # Get context for a chunk using Claude
    # Args: doc (str): Full document text, chunk (str): Text chunk to contextualize
    # Returns: tuple[str, any, str]: Contextualized text, usage stats, and original chunk
    def situate_context(self, doc: str, chunk: str) -> tuple[str, any, str]:
        DOCUMENT_CONTEXT_PROMPT = f"""
        <document>
        {doc}
        </document>
        """

        CHUNK_CONTEXT_PROMPT = f"""
        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """

        response = self.client.beta.prompt_caching.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": DOCUMENT_CONTEXT_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": CHUNK_CONTEXT_PROMPT,
                        },
                    ],
                },
            ],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        # log the response.usage
        logging.info(f"Response usage: {response.usage}")
        return response.content[0].text, response.usage, chunk

    # Main splitting function with caching
    # Args: txt (str): Input text to split, **kwargs: Additional arguments
    # Returns: list[tuple[str, dict]]: List of chunks with their metadata
    def __wrapped__(self, txt: str, **kwargs) -> list[tuple[str, dict]]:
        """Split the text into chunks using LangChain's splitter and add file context."""
        chunk_size = kwargs.get("chunk_size", self.chunk_size)
        chunk_overlap = kwargs.get("chunk_overlap", self.chunk_overlap)

        # Update splitter if parameters changed
        if chunk_size != self.chunk_size or chunk_overlap != self.chunk_overlap:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=self.separators,
            )

        # Check cache first
        cache_key = self._get_cache_key(txt)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.logger.info("Cache hit for key: %s", cache_key)
            if len(cached_result) == 0:
                self.logger.info("Cache hit for key: %s but empty", cache_key)
            else:
                return cached_result

        self.logger.info("Processing new text for key: %s", cache_key)

        # Process each chunk with document context
        original_chunks = self.splitter.split_text(txt)

        def process_chunk(doc, chunk):
            # for each chunk, produce the context
            contextualized_text, usage, _ = self.situate_context(doc, chunk)

            self.token_counts["input"] += usage.input_tokens
            self.token_counts["output"] += usage.output_tokens
            self.token_counts["cache_read"] += usage.cache_read_input_tokens
            self.token_counts["cache_creation"] += usage.cache_creation_input_tokens

            return f"{chunk}\n\n{contextualized_text}"

        contextualized_chunks = []

        for chunk in original_chunks:
            contextualized_chunk = process_chunk(txt, chunk)
            contextualized_chunks.append((contextualized_chunk, {}))
        # add metadata to each chunk

        # combine all output
        self.logger.info(
            "Input tokens: %s, Output tokens: %s, Cache read tokens: %s, Cache creation tokens: %s",
            self.token_counts["input"],
            self.token_counts["output"],
            self.token_counts["cache_read"],
            self.token_counts["cache_creation"],
        )

        # Cache the result before returning
        self.cache.set(cache_key, contextualized_chunks)

        return contextualized_chunks

    # Public interface for splitting
    def __call__(self, text: pw.ColumnExpression, **kwargs) -> pw.ColumnExpression:
        """Split given text into overlapping chunks with file context."""
        return super().__call__(text, **kwargs)


class ContextualMetadataSplitter(ContextualSplitter):
    """Extended splitter that handles documents with metadata.
    
    Inherits from ContextualSplitter and adds capability to process metadata
    along with the text chunks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __wrapped__(self, txt: str, **kwargs) -> list[tuple[str, dict]]:

        sub_docs_with_metadata = eval(txt)
        sub_docs = sub_docs_with_metadata[:-1]
        full_metadata = json.loads(sub_docs_with_metadata[-1])

        # add metadata to the chunks
        chunk_size = kwargs.get("chunk_size", self.chunk_size)
        chunk_overlap = kwargs.get("chunk_overlap", self.chunk_overlap)

        # Update splitter if parameters changed
        if chunk_size != self.chunk_size or chunk_overlap != self.chunk_overlap:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=self.separators,
            )

        # Check cache first
        cache_key = self._get_cache_key(str(sub_docs))
        cached_result = self.cache.get(cache_key)
        cached_chunks = []
        if cached_result is not None:
            self.logger.info("Cache hit for key: %s", cache_key)
            if len(cached_result) == 0:
                self.logger.info("Cache hit for key: %s but empty", cache_key)
            else:
                cached_chunks = cached_result

        def process_chunk(doc, chunk):
            # for each chunk, produce the context
            contextualized_text, usage, _ = self.situate_context(doc, chunk)

            self.token_counts["input"] += usage.input_tokens
            self.token_counts["output"] += usage.output_tokens
            self.token_counts["cache_read"] += usage.cache_read_input_tokens
            self.token_counts["cache_creation"] += usage.cache_creation_input_tokens

            return f"{chunk}\n\n{contextualized_text}"

        if not cached_chunks:
            original_chunks = []
            for sub_doc in sub_docs:
                # Process each chunk with document context
                original_chunks.append(self.splitter.split_text(sub_doc))

            contextualized_chunks = []
            for i, sub_doc_chunks in enumerate(original_chunks):
                for chunk in tqdm(sub_doc_chunks):
                    contextualized_chunk = process_chunk(sub_docs[i], chunk)
                    contextualized_chunks.append((contextualized_chunk, i))

            self.cache.set(cache_key, contextualized_chunks)
        else:
            contextualized_chunks = cached_chunks

        contextualized_chunks_with_metadata = []
        for chunk, i in contextualized_chunks:
            contextualized_chunks_with_metadata.append(
                (
                    chunk,
                    {"sub_doc_id": i, "topics": full_metadata["topics"][str(i)]},
                )  # str needed since json has str keys
            )

        # combine all output
        self.logger.info(
            "Input tokens: %s, Output tokens: %s, Cache read tokens: %s, Cache creation tokens: %s",
            self.token_counts["input"],
            self.token_counts["output"],
            self.token_counts["cache_read"],
            self.token_counts["cache_creation"],
        )

        return contextualized_chunks_with_metadata
