TOOL_DOCSTRINGS = [
    """
        def retriever_tool(query: str) -> str:
        \"\"\"
        Purpose: Fetch information from the document corpus. Should be used whenever relevant sections of documents related to the query are needed.
        Returns a string with the requested information, if found. THIS TOOL MUST BE USED.

        Example:
        <query> What are the terms of renewal of the contract between Company X and Company Y? </query>
        <response> 
        ....
        retrieved_document_text = retriever_tool(query='Renewal, Company X and Company Y')
        ....
        </response>
        \"\"\"
    """,
    """
    def logic_tool(input_string: str, request: str, output_type: str, return_explanation: bool = False) -> str:
        \"\"\"
        Purpose: Use this tool to perform logical reasoning on the input string based on the request, or to convert information in the input string to the output type.
        Output type can be int, float, str, bool, list, dict, tuple, set, etc.
        This tool is overshadowed by specialized tools if present in the pipeline.
        If return_explanation is True, return a tuple with the first element being the output and the second element being the explanation.
        \"\"\"
    """,
    """
        def solve_tool(expression: str) -> str:
        \"\"\"      
        Purpose: Perform calculations, numerical analysis, or unit conversions.
        **Key Use Cases**:
        - Solving equations or performing arithmetic operations.
        - Supporting numerical reasoning or data manipulation.
        - The input expression should follow the rules of python's numexpr library.
        **Pipeline Integration**:
        - Use as a secondary tool in conjunction with decomposer_tool or multihop_tool for numerical queries.
        **Usage**:
        solve_tool(expression='Mathematical expression to evaluate')
        \"\"\"
    """,
    """
        def legal_analyst(query: str, context: str) -> dict:
        \"\"\"
        **Purpose:** Primary tool for legal research and analysis of the gathered information, context or agreements, along with searching and analyzing case law, PACER data, and legal documents.

        **Priority:** 
        - HIGH PRIORITY for all legal queries involving case law, acts, agreements, sections, or legal documentation
        - Should be used BEFORE tavily_search_results_json or other general search tools for legal searches
        - Use after multihop_tool for legal domain queries if uploaded documents are related to the key entities in the query.

        **Primary Applications:**
        1. Legal Information Analysis:
           - Analyzing and summarizing court decisions, gathered information and agreements
           - Extracting key legal details
           - Interpreting statutory provisions and regulations
        2. Case Law Search:
           - Finding relevant legal precedents and court decisions
           - Accessing PACER data and oral arguments
           - Retrieving judge information and court opinions

         The format of the dictionary is:
         {
            "message": str
         }

        **Usage:**
        legal_analyst(
            query='Legal search query or analysis request',
            context='Optional additional context or specific requirements'
        )
        \"\"\"  
    """,
    """
        def technical_indicator_analyst(query: str, context: str) -> str:
        \"\"\"
        Purpose:** Specialized tool for technical analysis of financial markets, focusing on indicators, patterns, and market trends.

        **Priority:** 
        - HIGH PRIORITY for queries specifically involving technical analysis or market indicators
        - Should be used AFTER multihop_tool if specific company or market data is needed
        - Use when detailed technical analysis is required beyond basic market data

        **Primary Applications:**
        1. Technical Analysis:
           - Calculating and interpreting technical indicators (RSI, MACD, EMA)
           - Pattern recognition and trend analysis
           - Support and resistance level identification

        2. Market Trend Analysis:
           - Price movement analysis
           - Volume analysis
           - Momentum indicators evaluation

        3. Trading Signal Generation:
           - Buy/sell signal identification
           - Trend reversal detection
           - Market sentiment analysis

        **Integration Points:**
        1. Data Requirements:
           - Historical price data
           - Volume information
           - Market indicators

        2. Pipeline Position:
           - Use AFTER multihop_tool for gathering specific company or market context
           - FOLLOWS multihop_tool for basic market data
           - Can be combined with other financial tools for comprehensive analysis

        **Usage:**
        technical_indicator_analyst(
            query='Technical analysis query',
            context='Optional market context or specific requirements'
        )
        \"\"\"
    """,
    """
        def macroeconomic_analyst(query: str, context: str) -> str:
        \"\"\"
        Purpose:** Expert tool for analyzing macroeconomic trends, indicators, and their market impact.

        **Priority:** 
        - HIGH PRIORITY for economic indicator analysis and market impact assessment
        - Use AFTER multihop_tool when specific economic data or context is needed
        - Essential for queries involving economic policy or market implications

        **Primary Applications:**
        1. Economic Indicator Analysis:
           - GDP, inflation, and employment data interpretation
           - Interest rate impact assessment
           - Currency market analysis

        2. Policy Impact Evaluation:
           - Monetary policy analysis
           - Fiscal policy implications
           - Global economic trend assessment

        **Integration Points:**
        1. Data Sources:
           - Economic indicators
           - Policy statements
           - Market reaction data

        2. Pipeline Position:
           - FOLLOWS multihop_tool for gathering economic context
           - Use after multihop_tool for specific indicator data
           - Can combine with other financial tools for comprehensive analysis

        **Usage:**
        macroeconomic_analyst(
            query='Economic analysis query',
            context='Optional economic context or specific requirements'
        )
        \"\"\"
    """,
    """
        def financial_news_analyst(query: str) -> str:
        \"\"\"
        Purpose: Summarize and analyze financial news for sentiment and market insights.

        **Key Use Cases**:
        - Understanding sentiment or implications of financial news.
        - Providing updates on market-moving events.

        **Pipeline Integration**:
        - Use with decomposer_tool for in-depth news-related queries.

        **Usage**:
        financial_news_analyst(query='Describe the financial news query or sentiment analysis needed')
        \"\"\"
    """,
    """
        def trading_analyst(query: str, context: str) -> str:
        \"\"\"
        Purpose:** Real-time market analysis and trading insights tool.

        **Priority:** 
        - HIGH PRIORITY for current market conditions and trading decisions
        - Use AFTER multihop_tool when specific market context is needed
        - Essential for real-time market analysis queries

        **Primary Applications:**
        1. Market Analysis:
           - Real-time price analysis
           - Volume and liquidity assessment
           - Market depth evaluation

        2. Trading Insights:
           - Market momentum analysis
           - Trading opportunity identification
           - Risk assessment

        **Integration Points:**
        1. Data Requirements:
           - Real-time market data
           - Trading volumes
           - Order book information

        2. Pipeline Position:
           - FOLLOWS multihop_tool for market context
           - Use after multihop_tool for basic market data
           - Can combine with technical analysis for comprehensive trading decisions

        **Usage:**
        trading_analyst(
            query='Trading analysis query',
            context='Optional market context or specific requirements'
        )
        \"\"\"
    """,
    """
        def search_symbol(query: str) -> str:
        \"\"\"
        Purpose: Identify stock ticker symbols for companies or financial instruments.

        **Key Use Cases**:
        - Finding tickers for specific companies or industries.
        - Supporting other financial tools by providing symbol inputs.

        **Pipeline Integration**:
        - Use as a preparatory step for financial analysis queries.

        **Usage**:
        search_symbol(query='Company or instrument name')
        \"\"\"
    """,
    """
        def wikipedia(query: str) -> str:
        \"\"\"
        Purpose: Retrieve foundational, historical, or basic background knowledge.

        **Key Use Cases**:
        - Explaining concepts or providing historical context.
        - Supplementing other tools with general information.

        **Pipeline Integration**:
        - Use as a fallback for general, non-current queries.

        **Usage**:
        wikipedia(query='Topic or concept for explanation')
        \"\"\"
    """,
    #  """
    #      def graph_tool(query: str, context: str)->None:
    #      \"\"\"
    #      Purpose: Generate visualizations of data using Plotly. The context must explicitly contain the data to be visualized.
    #      Returns nothing, just saves the plot to a file.
    #      **Usage**:
    #      graph_tool(query='The query describing what to visualize', context='Data context for creating the visualization')
    #      \"\"\"
    #  """,
    """
        def financial_news_analyst(query: str) -> str:
        \"\"\"
        Purpose: Gathering and analyzing financial news for sentiment and market insights from yahoo finance.

        **Key Use Cases**:
        - Understanding sentiment or implications of financial news.
        - Providing updates on market-moving events.

        **Pipeline Integration**:
        - Primary tool for news analysis and sentiment assessment
        - Can be used independently or with other financial tools
        
        **Usage**:
        financial_news_analyst(query='Describe the financial news query or sentiment analysis needed')
        \"\"\"
    """,
    """
        def serper_search(query: str) -> str:
        \"\"\"
        Purpose: Retrieve current, general, or trending information from web sources using Google Serper API.

        **Always use the multihop_tool before serper_search if the query or key entities, subparts or any parts of query are similar to the key entities in the additional context sources.**

        **Key Use Cases**:
        - Gathering updates on current events, trends, or public knowledge.
        - Fact-checking or validating recent developments.

        **Pipeline Integration**:
        - High priority tool for queries requiring current, general, or trending information.
        - Use tavily_search_results_json as a secondary tool for general web data along with serper_search for complex queries.
        - **Do NOT use for live financial data extraction.**

        **Usage**:  
        serper_search(query='Specific search query for general web data')
        \"\"\"
    """,
]
