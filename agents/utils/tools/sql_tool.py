# Import system libraries
import os
import sys

# Add parent directory to system path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import required libraries
import logging
from typing import Optional

import psycopg2
from config import AgentsConfig as Config
from langchain.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel, ConfigDict, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SQLQuerySchema(BaseModel):
    """Schema for SQL Query Tool input.
    
    Args:
        query (str): The natural language query to be processed
        
    Returns:
        SQLQuerySchema: An instance of the schema
    """

    query: str = Field(
        ...,
        description="The natural language question to be converted into SQL query",
        examples=[
            "How many users registered last month?",
            "What is the total revenue for Q1 2024?",
            "List all products with price above $100",
        ],
    )


class SQLQueryTool(BaseTool):
    """Tool for generating and executing SQL queries.
    
    Args:
        llm (BaseLanguageModel): Language model for query generation
        db_schema (Optional[str]): Database schema, if None will be fetched
        
    Returns:
        SQLQueryTool: An instance of the SQL query tool
    """

    name: str = "sql_database"
    description: str = """
    Generate and execute SQL queries based on user questions.
    Input should be a natural language question about the data.
    The tool will generate and execute appropriate SQL query.
    """
    args_schema: type[BaseModel] = SQLQuerySchema

    # Add model config
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Define class attributes with default values
    llm: Optional[BaseLanguageModel] = None
    db_schema: Optional[str] = None
    config: Config = Config()

    def __init__(self, llm: BaseLanguageModel, db_schema: Optional[str] = None):
        """Initialize the SQL Query Tool."""
        # Initialize base class first
        super().__init__()

        # Set instance attributes
        self.llm = llm
        self.db_schema = db_schema or self._fetch_db_schema()

        # Validate required fields
        if not self.llm:
            raise ValueError("LLM is required for SQL Query Tool")

        # Update description with schema
        self.description = f"""
        Generate and execute SQL queries based on user questions.
        Input should be a natural language question about the data.
        
        Database Schema:
        {self.db_schema}
        
        Example:
        Question: "How many users registered last month?"
        Generated SQL: "SELECT COUNT(*) FROM users WHERE DATE_TRUNC('month', registration_date) = DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')"
        """

    def _get_db_connection(self):
        """Create and return a database connection."""
        return psycopg2.connect(
            dbname=self.config.DB_NAME,
            user=self.config.DB_USER,
            password=self.config.DB_PASSWORD,
            host=self.config.DB_HOST,
            port=self.config.DB_PORT,
        )

    def _fetch_db_schema(self) -> str:
        """Fetch database schema from PostgreSQL."""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Query to get table and column information
            schema_query = """
                SELECT 
                    t.table_name,
                    array_agg(c.column_name || ' ' || c.data_type) as columns
                FROM 
                    information_schema.tables t
                    JOIN information_schema.columns c 
                        ON t.table_name = c.table_name
                WHERE 
                    t.table_schema = 'public'
                GROUP BY 
                    t.table_name;
            """

            cursor.execute(schema_query)
            results = cursor.fetchall()

            # Format schema information
            schema = []
            for table_name, columns in results:
                schema.append(f"Table: {table_name}")
                schema.append("Columns:")
                for column in columns:
                    schema.append(f"  - {column}")
                schema.append("")

            return "\n".join(schema)

        except Exception as e:
            logger.error(f"Error fetching schema: {str(e)}")
            return "Schema unavailable"
        finally:
            if conn:
                conn.close()

    def _generate_sql_query(self, question: str) -> str:
        """Generate SQL query from natural language question."""
        prompt = f"""
        Given the following question: "{question}"
        And the database schema:
        {self.db_schema}
        
        Generate a PostgreSQL query that would answer this question.
        Return only the SQL query, nothing else.
        Use appropriate PostgreSQL functions and syntax.
        """

        response = self.llm.predict(prompt)
        return response.strip()

    def _execute_query(self, query: str) -> str:
        """Execute SQL query and return results."""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query)
            results = cursor.fetchall()
            return str([dict(row) for row in results])

        except Exception as e:
            return f"Error executing query: {str(e)}"
        finally:
            if conn:
                conn.close()

    def _run(self, query: str) -> str:
        """Run the tool."""
        try:
            sql_query = self._generate_sql_query(query)
            logger.info(f"Generated SQL query: {sql_query}")
            results = self._execute_query(sql_query)
            return f"Results: {results}"

        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async implementation of the tool."""
        return self._run(query)


if __name__ == "__main__":
    from langchain_openai import ChatOpenAI

    # Initialize LLM
    llm = ChatOpenAI(modeltemperature=0)

    # Create SQL tool instance
    sql_tool = SQLQueryTool(llm=llm)

    # Test questions
    test_questions = [
        "How many users are in the system?",
        "What are the top 5 most expensive products?",
    ]

    # Run tests
    for question in test_questions:
        print(f"\nQuestion: {question}")
        result = sql_tool._run(question)
        print(f"Result: {result}")
