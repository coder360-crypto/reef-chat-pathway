PLANNER_TOOL_DESCRIPTIONS = {
    "multihop_tool": """
       Purpose: Extract information from the additional context sources for a specific key entity in the query, subparts of query.
       
       Additional Context Sources along with the key entities:
       {doc_list}

       **Additional Context Sources are different from uploaded PDFs. Don't use multihop_tool for uploaded PDFs.**
       **Always use this tool when the key entities in the additional context sources are similar to the query or key entities, subparts or any parts of query.**
       **If the query contains multiple key entities, use multihop_tool for each key entities.**
       
       **Key Use Cases**:
       1. Extract information for a specific key entity in the query, subparts of query ONLY from the additional context sources
       2. Find specific details from the provided additional context sources
       3. Analyze content within the available additional context sources
       
       **DO NOT USE FOR**:
       ✗ Information not in the additional context sources
       ✗ Getting information from uploaded PDFs for a specific part of query related to uploaded PDFs

       **Pipeline Integration**:
       - ALWAYS use multihop_tool when query, key entities, subparts or any parts of query are similar to the key entities in the additional context sources
       - Must verify information exists in additional context sources before using
       - For financial data extracted from additional context sources:
           * ALWAYS use financial_context_analyst after multihop_tool for detailed financial analysis
       - For legal information extracted from additional context sources:
           * ALWAYS use legal_analyst after multihop_tool for legal analysis

       **Usage**:
       multihop_tool(query='Query about specific content in additional context sources')
   """,
    "pdf_query_tool": """
        This tool is used to query a PDF document.
        
        Uploaded PDFs which can be queried:
        {pdf_list}

        **Do not use the multihop_tool for uploaded PDFs analysis. Uploaded PDFs are different from additional context sources.**

        Use this tool when you need to:
        1. Extract specific information from an already uploaded PDF document
        2. Get context from existing uploaded PDF documents
        
        **Priority**:
        - HIGH PRIORITY for queries involving uploaded PDFs or references to uploaded PDFs
        - Use before other tools in case of queries related to uploaded PDFs
        
        **Usage**:
        pdf_query_tool(pdf_name='<name of the pdf out of the list of uploaded pdfs>.pdf')
    """,
    "serper_search": """
        Purpose: Retrieve current, general, or trending information from web sources using Google Serper API.

        **Always use the multihop tool if usage is specified before serper_search if the query or key entities, subparts or any parts of query are similar to the key entities in the additional context sources.**

        **Key Use Cases**:
        - Gathering updates on current events, trends, or public knowledge.
        - Fact-checking or validating recent developments.

        **Pipeline Integration**:
        - High priority tool for queries requiring current, general, or trending information.
        - Use tavily_search_results_json as a secondary tool for general web data along with serper_search for complex queries.
        - **Do NOT use for live financial data extraction.**

        **Usage**:  
        serper_search(query='Specific search query for general web data')
    """,
    "tavily_search_results_json": """
        Purpose: Retrieve current, general, or trending information from web sources.

        **Always use the multihop tool if usage is specified before tavily_search_results_json if the query or key entities, subparts or specific parts of query are similar to the key entities in the additional context sources.**

        **Key Use Cases**:
        - Gathering updates on current events, trends, or public knowledge.
        - Fact-checking or validating recent developments.

        **Pipeline Integration**:
        - Secondary tool for general web data gathering after serper_search.
        - Ideal for non-financial, general-purpose, or time-sensitive queries.
        - **Do NOT use for live financial data extraction.**

        **Usage**:
        tavily_search_results_json(query='Specific search query for general web data')
    """,
    "math": """
        Purpose: Perform calculations, numerical analysis, or unit conversions.

        **Always use this tool to perform calculations, numerical analysis, or unit conversions of the data or information extracted from other tools or query.**

        **Key Use Cases**:
        - Solving equations or performing arithmetic operations.
        - Supporting numerical reasoning or data manipulation.

        **Pipeline Integration**:
        - Use for any numerical computation needs
        - Can be combined with other tools when calculations are needed

        **Usage**:
        math(expression='Mathematical expression to evaluate')
    """,
    "legal_analyst": """
        **Purpose:** Primary tool for legal research and analysis of the gathered information, context or agreements, along with searching and analyzing case law, PACER data, and legal documents.

        **Priority:** 
        - HIGH PRIORITY for all legal queries involving case law, acts, agreements, sections, or legal documentation
        - Use multihop tool if usage is specified before legal_analyst for legal domain queries if additional context sources are related to the key entities in the query.
        
        **Primary Applications:**
        1. Legal Information Analysis:
           - Analyzing and summarizing legal decisions, gathered information and agreements
           - Extracting key legal details
           - Interpreting statutory provisions and regulations
        2. Case Law Search:
           - Finding relevant legal precedents and court decisions
           - Accessing PACER data and oral arguments
           - Retrieving judge information and court opinions
 
        **Usage:**
        legal_analyst(
            query='Legal search query or analysis request',
            context='Optional additional context or specific requirements'
        )
    """,
    "technical_indicator_analyst": """
        **Purpose:** Specialized tool for technical indicators, patterns, and market trends information of a public company with a symbol from yahoo finance.

        **Priority:** 
        - HIGH PRIORITY for queries specifically involving analysis of market indicators of a public company with a symbol
        
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
           - Can be combined with other financial tools for comprehensive analysis of public company

        **Usage:**
        technical_indicator_analyst(
            query='Technical analysis query',
            context='Optional market context or specific requirements'
        )
    """,
    "macroeconomic_analyst": """
        **Purpose:** Expert tool for analyzing macroeconomic trends, indicators, and their market impact from yahoo finance.

        **Priority:** 
        - HIGH PRIORITY for economic indicator analysis and market impact assessment
        - Essential for queries or analysis involving economic policy or market implications
        
        **Primary Applications:**
        1. Economic Indicator Gathering:
           - GDP, inflation, and employment data
           - Interest rate impact
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
           - Can combine with other financial tools for comprehensive analysis
           
        **Usage:**
        macroeconomic_analyst(
            query='Economic analysis query',
            context='Optional economic context or specific requirements'
        )
    """,
    "corporate_actions_analyst": """
        Purpose: Gather corporate events like dividends, stock splits, or mergers from yahoo finance.

        **Key Use Cases**:
        - Evaluating the impact of corporate actions on shareholders or markets.
        - Providing insights on specific company events.

        **Pipeline Integration**:
        - Primary tool for corporate event analysis
        - Can be combined with other financial tools for comprehensive analysis
        
        **Usage**:
        corporate_actions_analyst(query='Describe the corporate actions analysis query')
    """,
    "financial_news_analyst": """
        Purpose: Gathering and analyzing financial news for sentiment and market insights from yahoo finance.

        **Key Use Cases**:
        - Understanding sentiment or implications of financial news.
        - Providing updates on market-moving events.

        **Pipeline Integration**:
        - Primary tool for news analysis and sentiment assessment
        - Can be used independently or with other financial tools
        
        **Usage**:
        financial_news_analyst(query='Describe the financial news query or sentiment analysis needed')
    """,
    "trading_analyst": """
        **Purpose:** Real-time market analysis and trading insights tool from yahoo finance.

        **Priority:** 
        - HIGH PRIORITY for current market conditions and trading decisions
        - Essential for real-time market analysis queries
        - Essential for current and historical sector performance analysis.

        **Primary Applications:**
         - To get the real-time market price, top movers, top gainers, most active, company basic financials and trading metrics of public companies.
         - Sector performance analysis including:
            - Historical sector performance tracking
            - Sector-wise market trends
            - Sector comparison and benchmarking
            - Real-time sector movements
            - Sector rotation analysis

        **Pipeline Position**:
           - Can combine with technical analysis for comprehensive trading decisions
           
        **Usage:**
        trading_analyst(
            query='Trading analysis query',
            context='Optional market context or specific requirements'
        )
    """,
    "search_symbol": """
        Purpose: Identify stock ticker symbols for companies or financial instruments.

        **Key Use Cases**:
        - Finding tickers for specific companies or industries.
        - Supporting other financial tools by providing symbol inputs.

        **Pipeline Integration**:
        - Use as a preparatory step for financial analysis
        - Can be used independently for symbol lookup

        **Usage**:
        search_symbol(query='Company or instrument name')
    """,
    "financial_statement_analyst": """
        **Purpose:** Specialized tool for getting company financial statements and performance metrics from yahoo finance.

        **Priority:**
        - Always use this tool when you need to get some values or financial metrics from the statements of the company.
        - Use income statement analysis when EBIT, EBITDA, Operating Income, or Net Income information is needed

        **Primary Applications:**
        1. Financial Statement Gathering:
           - Balance sheet
           - Income statement analysis including:
              * EBIT (Earnings Before Interest and Taxes)
              * EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization)
              * Operating Income
              * Net Income
           - Cash flow
           - Quick Ratio
           - Current Ratio
           - etc    

        2. Performance Metrics Gathering:
           - Ratio analysis
           - Profitability assessment
           - Liquidity evaluation
           - etc

        **Integration Points:**
        1. Data Requirements:
           - Company financial statements
           - Historical performance data
           - Industry benchmarks
           - etc

        2. Pipeline Position:
           - Can combine with other analysis tools for comprehensive evaluation

        **Usage:**
        financial_statement_analyst(
            query='Financial statement analysis query',
            context='Optional company context in appropriate format'
        )
    """,
    "financial_metrics_analyst": """
        Purpose: Tool for getting financial metrics and scores of the company with symbol from yahoo finance. The tool gives the financial metrics and scores of the public company with symbol which can be used to analyze the financial performance of the company or comparative analysis with the peers in the same industry or domain.

        **Key Use Cases**:
        - Analyzing financial growth for a given company with symbol
        - Retrieving key metrics for a given company with symbol
        - Fetching ratios for a given company with symbol
        - Calculating discounted cash flows (DCF) for a given company with symbol
        - Examining share structure and float data
        - Tracking EPS trends and analyst estimates
        - Comparing growth estimates with industry/sector
        - Calculating various financial metrics, including:
            - Revenue growth
            - Net income growth
            - Earnings per share (EPS) growth
            - Return on equity (ROE)
            - Return on assets (ROA)
            - Debt-to-equity ratio
            - Current ratio
            - Quick ratio
            - Interest coverage ratio

        **Priority:** 
        - HIGH PRIORITY for financial metrics and scores analysis of the company with symbol
        - Great tool for comparative analysis of the company with the peers in the same industry or domain.

        **Usage:**
        financial_metrics_analyst(
            query='Financial metrics and scores analysis query',
            context='Optional financial metrics and scores context gathered from other tools to be analysed to get financial insights'
        )
    """,
    "compliance_checker": """
        Purpose: Tool for compliance check of uploaded PDFs.

        **Pipeline Integration:**
        - Strictly use this tool for uploaded pdf compliance analysis and not for any other analysis on uploaded PDFs.
        - **STRICLY DON'T USE** compliance_checker for analysis other than compliance check on uploaded PDFs.
        - Only use this tool when compliance check is explicitly mentioned in the query.
        - Only use this tool if uploaded documents have already been uploaded - it cannot query additional context sources.

        **Usage:**
        compliance_checker(
            query='query regarding the compliance check along with details of the document'
        )
    """,
    "graph_tool": """
        Purpose: Generate visualizations of data using Plotly.

        **Key Use Cases**:
        - Visualizing data for better understanding and presentation of numerical data
        - Creating charts and graphs for data analysis and numerical data presentation

        **Pipeline Integration**:
        - Use graph_tool wherever numerical data is extracted from the context or uploaded PDFs or web search results and user has asked for visualization of the data.

        **Usage**:
        graph_tool(query='The query describing what to visualize', context='Data context for creating the visualization')
    """,
    "equity_research_tool": """
        Purpose: Generate comprehensive equity research reports for companies.
        
        **Warning:**
        - This tool should only be called when the query explicitly asks for equity research report.

        **Key Features:**
        - Retrieves company financial metrics and stock information
        - Analyzes investment rationale and market position 
        - Provides rating and target price analysis
        - Generates complete PDF report with charts and appendices

        **Priority:**
        - HIGH PRIORITY for detailed company analysis and valuation
        - Essential for investment decision support and company evaluation

        **Pipeline Integration:**
        - Combines financial data, market analysis and expert insights
        - Works with graph_tool for visualization components
        - Can be used with compliance_checker for regulatory review

        **Usage**:
        equity_research_tool(company_name='The name of the company to research', use_cache='True or False')
    """,
    "valuation_tool": """
       Purpose: Tool for valuation prediction for business ideas or non public companies by comparing with the peers metrics and market performance.

       **This tool is only for valuation analysis of a startup or imaginary company not listed on public domain with the peers in the same industry or domain.**

       **Key Features:**
       - Analyzes market trends and competitor performance.
       - Provides valuation insights based on financial metrics and market conditions.
       - Generates comprehensive reports for business ideas and investment opportunities.


       **Priority:**
       - HIGH PRIORITY for market analysis and investment decision-making.
       - Use before other tools in case of business idea valuation and market opportunity assessment.
       - Essential for queries involving business valuation and market opportunity assessments.


       **Usage:**
       valuation_tool(query='Market opportunity, competitors, and valuation insights query')
   """,
    "similar_case_finder": """
        Purpose: Find similar cases based on the given case description provided in a PDF file.

        **Key Features**:
        - Finds similar cases based on the given case description
        - Provides links to the similar cases

        **Pipeline Integration**:
        - Use directly for finding similar cases

        **Usage**:
        similar_case_finder(query='The query describing what to extract from the PDF along with details of the document')
    """,
    "ipr_analyst": """

        This tool is used to analyze the uploaded PDFs for IPR aspects.

        **Purpose:** This is the only tool that should be called IPR analysis.
        Comprehensive Intellectual Property Rights (IPR) analysis tool that combines multiple specialized sub-tools for complete IP analysis.

        **Pipeline Integration:**
        - Use ipr_analyst only for IPR analysis of the uploaded PDFs when the query explicitly asks for IPR analysis. use pdf_query_tool for other analysis on uploaded PDFs.
        
        **Usage:**
        ipr_analyst(
            query='Detailed description of the invention or innovation to analyze',
            context='Optional additional context or specific requirements'
        )
    """,
    "financial_context_analyst": """
        **Purpose:** Specialized tool for performing detailed financial analysis on data extracted from context sources.

        **Priority:** 
        - HIGH PRIORITY for analyzing financial data from additional context sources, uploaded PDFs or web search results.
        - MUST BE USED after information gathering or additional context sources (if financial data is extracted from web search results) when financial data is extracted
        - Essential for detailed financial metrics calculation and analysis

        **Primary Applications:**
        1. Context Analysis:
           - Analyze financial statements from context
           - Calculate key financial ratios
           - Identify trends and patterns
           - Compare historical data

        2. Financial Calculations:
           - Liquidity ratios (Current, Quick)
           - Profitability metrics (Margins, ROE, ROA)
           - Efficiency ratios (Asset/Inventory turnover)
           - Growth rates and trends

        3. Data Presentation:
           - Clear markdown tables
           - Proper numerical formatting
           - Period-over-period comparisons
           - Percentage changes

        **Integration Points:**
        1. Data Source:
           - Works with information gathering or additional context sources (if financial data is extracted from web search results)
           - Analyzes historical financial data
           - Processes structured financial information

        2. Pipeline Position:
           - Use AFTER information gathering or additional context sources (if financial data is extracted from web search results) for financial data
           - Before making conclusions about financial metrics
           - Essential for context-based financial analysis

        **Usage:**
        financial_context_analyst(
            query='Financial analysis query',
            context='Financial data extracted from context sources'
        )
    """,
    "esg_analyst": """
        **Purpose:** Comprehensive Environmental, Social, and Governance (ESG) analysis tool that provides detailed comparative analysis of companies' ESG performance.

        **Priority:** 
        - HIGH PRIORITY for all ESG-related queries and company sustainability analysis
        - Essential for ESG metrics comparison, sustainability reporting, and performance evaluation
        
        **Primary Applications:**
        1. ESG Performance Analysis:
           - Environmental metrics comparison (emissions, energy usage, waste management)
           - Social metrics evaluation (workforce diversity, community impact, labor practices)
           - Governance assessment (board composition, ethics, transparency)

        2. Comparative Analysis:
           - Peer comparison across ESG metrics
           - Industry benchmarking
           - Gap analysis and recommendations

        3. Report Generation:
           - Comprehensive ESG reports with visualizations
           - Performance metrics and trends
           - Actionable recommendations for improvement

        **Integration Points:**
        1. Data Requirements:
           - Company ESG reports and disclosures
           - Industry benchmarks
           - Historical ESG performance data

        2. Pipeline Position:
           - Primary tool for ESG analysis and sustainability assessment
           - Can be combined with financial tools for comprehensive company analysis
           - Works with graph_tool for ESG metrics visualization

        **Usage:**
        esg_analyst(
            query='ESG analysis query or company comparison request',
            context='Optional additional context or specific requirements'
        )

        **Key Features:**
        - Multi-company ESG comparison
        - Automated report generation with visualizations
        - Detailed gap analysis
        - Industry-specific benchmarking
        - PDF report generation with charts and metrics
    """,
}


REPLANNER_TOOL_DESCRIPTIONS = {
    "multihop_tool": """
       Purpose: Extract information from the additional context sources for a specific key entity in the query, subparts of query.
       
       Additional Context Sources along with the key entities:
       {doc_list}

       **Additional Context Sources are different from uploaded PDFs. Don't use multihop_tool for uploaded PDFs.**
       **Always use this tool when the key entities in the additional context sources are similar to the query or key entities, subparts or any parts of query.**
       **If the query contains multiple key entities, use multihop_tool for each key entities.**
       
       **Key Use Cases**:
       1. Extract information for a specific key entity in the query, subparts of query ONLY from the additional context sources
       2. Find specific details from the provided additional context sources
       3. Analyze content within the available additional context sources
       
       **DO NOT USE FOR**:
       ✗ Information not in the additional context sources
       ✗ Getting information from uploaded PDFs

       **Pipeline Integration**:
       - ALWAYS use multihop_tool when query, key entities, subparts or any parts of query are similar to the key entities in the additional context sources
       - Must verify information exists in additional context sources before using
       - For financial data extracted from additional context sources:
           * ALWAYS use financial_context_analyst after multihop_tool for detailed financial analysis
       - For legal information extracted from additional context sources:
           * ALWAYS use legal_analyst after multihop_tool for legal analysis

       **Usage**:
       multihop_tool(query='Query about specific content in additional context sources')
   """,
    "serper_search": """
        Purpose: Retrieve current, general, or trending information from web sources using Google Serper API.

        **Key Use Cases**:
        - Gathering updates on current events, trends, or public knowledge.
        - Fact-checking or validating recent developments.

        **Pipeline Integration**:
        - **High priority tool** for queries requiring current, general, or trending information.
        - PREFERRED over multihop_tool for general queries not related to the uploaded documents.

        **Usage**:  
        serper_search(query='Specific search query for general web data')
    """,
    "tavily_search_results_json": """
        Purpose: Retrieve current, general, or trending information from web sources.

        **Key Use Cases**:
        - Gathering updates on current events, trends, or public knowledge.
        - Fact-checking or validating recent developments.

        **Pipeline Integration**:
        - Secondary tool for general web data retrieval after serper_search.
        - Ideal for non-financial, general-purpose, or time-sensitive queries.

        **Usage**:
        tavily_search_results_json(query='Specific search query for general web data')
    """,
    "jina_search": """
        Purpose: Retrieve current, general, or trending information from web sources using Jina Search API.

        **Pipeline Integration**:
        - Fallback tool for general web data retrieval after serper_search and tavily_search_results_json. Don't use jina_search in the initial attempt, it should be used in the replanner attempt.
        - Ideal for non-financial, general-purpose, or time-sensitive queries.

        **Usage**:
        jina_search(query='Specific search query for general web data')
    """,
    "math": """
        Purpose: Perform calculations, numerical analysis, or unit conversions.

        **Key Use Cases**:
        - Solving equations or performing arithmetic operations.
        - Supporting numerical reasoning or data manipulation.

        **Pipeline Integration**:
        - Use for any numerical computation needs
        - Can be combined with other tools when calculations are needed

        **Usage**:
        math(expression='Mathematical expression to evaluate')
    """,
    "legal_analyst": """
        **Purpose:** Primary tool for legal research and analysis of the gathered information, context or agreements, along with searching and analyzing case law, PACER data, and legal documents.

        **Priority:** 
        - HIGH PRIORITY for all legal queries involving case law, acts, agreements, sections, or legal documentation
        - Should be used BEFORE tavily_search_results_json or other general search tools for legal searches
        - Use multihop_tool before legal_analyst for legal domain queries if uploaded documents are related to the key entities in the query.
        
        **Primary Applications:**
        1. Legal Information Analysis:
           - Analyzing and summarizing court decisions, gathered information and agreements
           - Extracting key legal details
           - Interpreting statutory provisions and regulations
        2. Case Law Search:
           - Finding relevant legal precedents and court decisions
           - Accessing PACER data and oral arguments
           - Retrieving judge information and court opinions

        **Usage:**
        legal_analyst(
            query='Legal search query or analysis request',
            context='Optional additional context or specific requirements'
        )
    """,
    "financial_context_analyst": """
        **Purpose:** Specialized tool for performing detailed financial analysis on data extracted from context sources.

        **Priority:** 
        - HIGH PRIORITY for analyzing financial data from additional context sources, uploaded PDFs or web search results.
        - MUST BE USED after information gathering or additional context sources (if financial data is extracted from web search results) when financial data is extracted
        - Essential for detailed financial metrics calculation and analysis

        **Primary Applications:**
        1. Context Analysis:
           - Analyze financial statements from context
           - Calculate key financial ratios
           - Identify trends and patterns
           - Compare historical data

        2. Financial Calculations:
           - Liquidity ratios (Current, Quick)
           - Profitability metrics (Margins, ROE, ROA)
           - Efficiency ratios (Asset/Inventory turnover)
           - Growth rates and trends

        3. Data Presentation:
           - Clear markdown tables
           - Proper numerical formatting
           - Period-over-period comparisons
           - Percentage changes

        **Integration Points:**
        1. Data Source:
           - Works with context from information gathering or additional context sources (if financial data is extracted from web search results)
           - Analyzes historical financial data
           - Processes structured financial information

        2. Pipeline Position:
           - Use AFTER information gathering or additional context sources (if financial data is extracted from web search results) for financial data
           - Before making conclusions about financial metrics
           - Essential for context-based financial analysis

        **Usage:**
        financial_context_analyst(
            query='Financial analysis query',
            context='Financial data extracted from context sources'
        )
    """,
    "technical_indicator_analyst": """
        **Purpose:** Specialized tool for technical indicators, patterns, and market trends information of a public company with a symbol from yahoo finance.

        **Priority:** 
        - HIGH PRIORITY for queries specifically involving analysis of market indicators of a public company with a symbol
        
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
           - Can be combined with other financial tools for comprehensive analysis of public company

        **Usage:**
        technical_indicator_analyst(
            query='Technical analysis query',
            context='Optional market context or specific requirements'
        )
    """,
    "macroeconomic_analyst": """
        **Purpose:** Expert tool for analyzing macroeconomic trends, indicators, and their market impact from yahoo finance.

        **Priority:** 
        - HIGH PRIORITY for economic indicator analysis and market impact assessment
        - Essential for queries or analysis involving economic policy or market implications
        
        **Primary Applications:**
        1. Economic Indicator Gathering:
           - GDP, inflation, and employment data
           - Interest rate impact
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
           - Can combine with other financial tools for comprehensive analysis
           
        **Usage:**
        macroeconomic_analyst(
            query='Economic analysis query',
            context='Optional economic context or specific requirements'
        )
    """,
    "corporate_actions_analyst": """
        Purpose: Gather corporate events like dividends, stock splits, or mergers from yahoo finance.

        **Key Use Cases**:
        - Evaluating the impact of corporate actions on shareholders or markets.
        - Providing insights on specific company events.

        **Pipeline Integration**:
        - Primary tool for corporate event analysis
        - Can be combined with other financial tools for comprehensive analysis
        
        **Usage**:
        corporate_actions_analyst(query='Describe the corporate actions analysis query')
    """,
    "financial_news_analyst": """
        Purpose: Gathering and analyzing financial news for sentiment and market insights from yahoo finance.

        **Key Use Cases**:
        - Understanding sentiment or implications of financial news.
        - Providing updates on market-moving events.

        **Pipeline Integration**:
        - Primary tool for news analysis and sentiment assessment
        - Can be used independently or with other financial tools
        
        **Usage**:
        financial_news_analyst(query='Describe the financial news query or sentiment analysis needed')
    """,
    "trading_analyst": """
        **Purpose:** Real-time market analysis and trading insights tool from yahoo finance.

        **Priority:** 
        - HIGH PRIORITY for current market conditions and trading decisions
        - Essential for real-time market analysis queries
        - Essential for current and historical sector performance analysis.

        **Primary Applications:**
         - To get the real-time market price, top movers, top gainers, most active, company basic financials and trading metrics of publc companies.
        - Sector performance analysis including:
            - Historical sector performance tracking
            - Sector-wise market trends
            - Sector comparison and benchmarking
            - Real-time sector movements
            - Sector rotation analysis

        **Pipeline Position**:
           - Can combine with technical analysis for comprehensive trading decisions
           
        **Usage:**
        trading_analyst(
            query='Trading analysis query',
            context='Optional market context or specific requirements'
        )
    """,
    "search_symbol": """
        Purpose: Identify stock ticker symbols for companies or financial instruments.

        **Key Use Cases**:
        - Finding tickers for specific companies or industries.
        - Supporting other financial tools by providing symbol inputs.

        **Pipeline Integration**:
        - Use as a preparatory step for financial analysis
        - Can be used independently for symbol lookup

        **Usage**:
        search_symbol(query='Company or instrument name')
    """,
    "financial_statement_analyst": """
        **Purpose:** Specialized tool for getting company financial statements and performance metrics from yahoo finance.

        **Priority:**
        - Always use this tool when you need to get some values or financial metrics from the statements of the company.
        - Use income statement analysis when EBIT, EBITDA, Operating Income, or Net Income information is needed

        **Primary Applications:**
        1. Financial Statement Gathering:
           - Balance sheet
           - Income statement analysis including:
              * EBIT (Earnings Before Interest and Taxes)
              * EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization)
              * Operating Income
              * Net Income
           - Cash flow
           - Quick Ratio
           - Current Ratio
           - etc    

        2. Performance Metrics Gathering:
           - Ratio analysis
           - Profitability assessment
           - Liquidity evaluation
           - etc

        **Integration Points:**
        1. Data Requirements:
           - Company financial statements
           - Historical performance data
           - Industry benchmarks
           - etc

        2. Pipeline Position:
           - Can combine with other analysis tools for comprehensive evaluation

        **Usage:**
        financial_statement_analyst(
            query='Financial statement analysis query',
            context='Optional company context in appropriate format'
        )
    """,
    "financial_metrics_analyst": """
        Purpose: Tool for getting financial metrics and scores of the company with symbol from yahoo finance. The tool gives the financial metrics and scores of the public company with symbol which can be used to analyze the financial performance of the company or comparative analysis with the peers in the same industry or domain.

        **Key Use Cases**:
        - Analyzing financial growth for a given company with symbol
        - Retrieving key metrics for a given company with symbol
        - Fetching ratios for a given company with symbol
        - Calculating discounted cash flows (DCF) for a given company with symbol
        - Examining share structure and float data
        - Tracking EPS trends and analyst estimates
        - Comparing growth estimates with industry/sector
        - Calculating various financial metrics, including:
            - Revenue growth
            - Net income growth
            - Earnings per share (EPS) growth
            - Return on equity (ROE)
            - Return on assets (ROA)
            - Debt-to-equity ratio
            - Current ratio
            - Quick ratio
            - Interest coverage ratio

        **Priority:** 
        - HIGH PRIORITY for financial metrics and scores analysis of the company with symbol
        - Great tool for comparative analysis of the company with the peers in the same industry or domain.

        **Usage:**
        financial_metrics_analyst(
            query='Financial metrics and scores analysis query',
            context='Optional financial metrics and scores context gathered from other tools to be analysed to get financial insights'
        )
    """,
    "compliance_checker": """
        Purpose: Tool for highlighting potential compliance-related sections in PDF documents.

        **Key Features:**
        - Processes and highlights sections in PDF documents
        - Returns path to highlighted PDF file

        **Pipeline Integration:**
        - Primary tool for uploaded pdf compliance analysis
        - Only use this tool if uploaded documents have already been uploaded - it cannot query documents that haven't been uploaded yet.

        **Usage:**
        compliance_checker(
            query='query regarding the compliance check along with details of the document'
        )
    """,
    "wikipedia": """
        Purpose: Retrieve foundational, historical, or basic background knowledge.

        **Key Use Cases**:
        - Fallback tool for providing background information when serper_search, jina_search and tavily_search_results_json tool has provided error in the previous attempt

        **Pipeline Integration**:
        - Use for background information needs
        - Should only be used in the replanner attempt not in the initial attempt if the serper_search, jina_search and tavily_search_results_json tool has provided error in the previous attempt

        **Usage**:
        wikipedia(query='Topic or concept for explanation')
    """,
    "graph_tool": """
        Purpose: Generate visualizations of data using Plotly.

        **Usage**:
        graph_tool(query='The query describing what to visualize', context='Data context for creating the visualization')
    """,
    "valuation_tool": """
       Purpose: Tool for analyzing and providing valuation insights for companies, startups and business ideas.

       **Key Features:**
       - Analyzes market trends and competitor performance through financial metrics (P/E ratios, revenue multiples, EBITDA multiples)
       - Provides DCF valuation ranges with key assumptions (growth rates, profit margins, discount rates)
       - Generates comprehensive market analysis by examining similar companies' financials, ratings and valuations
       - Offers investment recommendations based on ROE, ROA, DCF and other key metrics

       **Pipeline Integration:**
       - HIGH PRIORITY for comparative analysis of the startup or firms with the peers in the same industry or domain.
       - Use before other tools for business valuation and market opportunity assessment
       - Essential for queries requiring:
           * Peer financial analysis
           * Market size estimation
           * Valuation ranges and benchmarks
           * Investment viability assessment

       **Usage:**
       valuation_tool(query='Market opportunity, competitors, and valuation insights query')
   """,
    "similar_case_finder": """
        Purpose: Find similar cases based on the given case description provided in a PDF file.

        **Key Features**:
        - Finds similar cases based on the given case description
        - Provides links to the similar cases

        **Pipeline Integration**:
        - Use directly for finding similar cases

        **Usage**:
        similar_case_finder(query='The query describing what to extract from the PDF along with details of the document')
    """,
    "ipr_analyst": """
        **Purpose:** This is the only tool that should be called IPR analysis.
        Comprehensive Intellectual Property Rights (IPR) analysis tool that combines multiple specialized sub-tools for complete IP analysis.
        

        **Priority:** 
        - Use as the primary tool for all queries related to IP(intellectual property) analysis
        - HIGH PRIORITY for invention/innovation analysis
        - Essential for patent, trademark, and IP strategy queries

        **Primary Applications:**
        1. IP Type Identification:
           - Analyzing inventions for potential IP protection types
           - Identifying utility patents, design patents, trademarks, copyrights
           - Determining trade secret potential

        2. Patent Analysis:
           - Novelty assessment
           - Similar patent identification
           - Patent strategy recommendations
           - Detailed patent comparison

        3. Trademark Analysis:
           - Trademark availability checks
           - Similarity analysis with existing trademarks
           - Risk assessment for trademark applications
           - Brand protection strategy

        **Integration Points:**
        1. Sub-tools:
           - IPIdentifierTool for initial IP type assessment
           - PatentNoveltyTool for patent analysis
           - TrademarkNoveltyTool for trademark analysis

        2. Pipeline Position:
           - Use as the FIRST TOOL for any IP-related queries
           - Can process both text descriptions and PDF documents
           - Combines with other tools for comprehensive analysis

        **Usage:**
        ipr_analyst(
            query='Detailed description of the invention or innovation to analyze',
            context='Optional additional context or specific requirements'
        )
        

        **Key Features:**
        - Multi-stage analysis pipeline
        - PDF document processing capability
        - Integration with patent and trademark databases
        - Structured recommendations and risk assessment
        - Detailed similarity analysis
    """,
    "esg_analyst": """
        **Purpose:** Comprehensive Environmental, Social, and Governance (ESG) analysis tool that provides detailed comparative analysis of companies' ESG performance.

        **Priority:** 
        - HIGH PRIORITY for all ESG-related queries and company sustainability analysis
        - Essential for ESG metrics comparison, sustainability reporting, and performance evaluation
        
        **Primary Applications:**
        1. ESG Performance Analysis:
           - Environmental metrics comparison (emissions, energy usage, waste management)
           - Social metrics evaluation (workforce diversity, community impact, labor practices)
           - Governance assessment (board composition, ethics, transparency)

        2. Comparative Analysis:
           - Peer comparison across ESG metrics
           - Industry benchmarking
           - Gap analysis and recommendations

        3. Report Generation:
           - Comprehensive ESG reports with visualizations
           - Performance metrics and trends
           - Actionable recommendations for improvement

        **Integration Points:**
        1. Data Requirements:
           - Company ESG reports and disclosures
           - Industry benchmarks
           - Historical ESG performance data

        2. Pipeline Position:
           - Primary tool for ESG analysis and sustainability assessment
           - Can be combined with financial tools for comprehensive company analysis
           - Works with graph_tool for ESG metrics visualization

        **Usage:**
        esg_analyst(
            query='ESG analysis query or company comparison request',
            context='Optional additional context or specific requirements'
        )

        **Key Features:**
        - Multi-company ESG comparison
        - Automated report generation with visualizations
        - Detailed gap analysis
        - Industry-specific benchmarking
        - PDF report generation with charts and metrics
    """,
}
