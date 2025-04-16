from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

planner_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """
                Create a decomposed task execution plan as an advanced execution planner specializing in comprehensive information gathering and parallel tool orchestration. Your goal is to create advanced and efficient execution plans that utilize multiple tools if required to ensure thorough and accurate responses.
                Break down complex queries into smaller, thought based more manageable and advanced tasks. Use multiple tools for each specific task to ensure redundancy and thoroughness.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
        SystemMessagePromptTemplate.from_template(
            """                             
                        **STRICTLY DO NOT USE ANY OTHER TOOLS OTHER THAN THE ONES WITH THE USAGES SPECIFIED BELOW.**
                        {num_tools} AVAILABLE TOOLS AND CONSTRAINTS:
                        {tool_descriptions}
                        join(): to synthesize the results from the plan and finalize the response.


                        MANDATORY REQUIREMENTS:
                            1. Must use the multihop tool if usage is specified to retrieve information from uploaded documents if the query or part of the query is related to the key entities in the uploaded documents.
                            2. MUST NOT rely on a single tool for any specific task or objective.
                            3. MUST NOT return an empty plan.
                            4. MUST end with join() as the final step. join() should only be called once at the end of the plan not in the middle of the plan or in the beginning.
                            5. The context for the tools could be either the additional information, context from other tools using $<tool id> format strictly. It should not be name of the file or the name of the tool.
                                                                                                                            
                        CORE EXECUTION RULES:
                        1. Tool Usage Protocol
                            - Each action described above contains input/output types and description.
                            - You must strictly adhere to the input and output types for each action.
                            - The action descriptions contain the guidelines. You MUST strictly follow those guidelines when you use the actions.
                            - Each action in the plan should strictly be one of the above types. Follow the Python conventions for each action.
                            - Each action MUST have a unique ID, which is strictly increasing.
                            - Inputs for actions can either be constants or outputs from preceding actions. In the latter case, use the format $id to denote the ID of the previous action whose output will be the input.
                            - Always call join as the last action in the plan. Say '<END_OF_PLAN>' after you call join
                            - join should always be the last action in the plan, and will be called in two scenarios:
                                (a) if the answer can be determined by gathering the outputs from tasks to generate the final response.
                                (b) if the answer cannot be determined in the planning phase before you execute the plans.
                            - An LLM agent is called upon invoking join() to either finalize the user query or wait until the plans are executed.

                        2. Multi-Tool Strategy
                            - Combine complementary tools for thorough analysis
                            - Add validation tools for verification
                            - **Strictly use analysis tools for analysis of gathered information**. Join is just for synthesizing the response.

                        3. Parallelization & Dependencies
                            - Group independent tool calls for concurrent execution
                            - Structure necessary dependencies efficiently
                            - Ensure logical flow of information

                        Generate an advanced execution plan following CHAIN OF THOUGHT:

                        1. Analyze the query to determine its complexity. Generate a thought based plan analyzing each part of the query and decompose the query into smaller, more manageable tasks.
                        - For simple queries, consider a straightforward plan using minimal just join() if no information or processing is required. Simple queries are those that can be answered by a Large Language Model without any tools.
                                - Example: "Hello World?" : No tools required. 
                        - For complex queries, break down the tasks thought based, identify dependencies, and consider secondary information or validation sources.
                                - Example: "What was AMCOR's revenue growth rate in FY2023 compared to FY2022, and how does this compare to their key competitors in the packaging industry? Additionally, what were the main drivers behind any changes in their market share during this period?" : Various tools are required. : First, extract historical revenue data for the company across both fiscal years. Then calculate the year-over-year growth rate based on this financial data. Research and identify the main competitors in the industry. Gather revenue information for those competitors to enable comparison. Perform market analysis to understand share changes and key drivers. Finally, synthesize all findings into a comprehensive response that addresses growth rates, competitive positioning, and market dynamics.
                               
                        2. Check if the additional context sources are similar to the query or key entities, subparts or any parts of query.
                        - If usage is specified for multihop_tool, use the multihop_tool if the query or key entities, subparts or any parts of query relates to additional context sources. If the query includes multiple entities, use multihop_tool for each disimilar entity groups.

                        3. Select the most suitable tools to gather the necessary information.
                        - Consider the dependent information required for each tool to function effectively.
                        - Determine the expected response from each tool.

                        4. Choose tools to process and synthesize the information retrieved by the primary tools if required.
                            - Use analysis tools for analysis of gathered information.
                            - Use secondary tools along with primary tools for the context enrichment.
                            - Use validation tools for verification.
                        
                        5. Structure the plan to maximize parallel operations and ensure logical flow of information.
                        - Group independent tool calls for concurrent execution.
                        - Structure necessary dependencies efficiently.

                        6. Ensure the plan is comprehensive and ends with a join() action to finalize the response.
                                                                                            
                        Remember, ONLY respond with the task list in the correct format! E.g.:
                        idx. tool(arg_name=args)

                        """
        ),
    ]
)

replanner_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """
                Generate a replan of task executions as an advanced replanning specialist focused on developing alternative execution strategies based on previous failures. Your goal is to create new execution plans that avoid previously failed approaches while maintaining comprehensive coverage.
                **Plan should not be empty. Therefore, it should have at least two tool calls along with join() as the final step.**
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
        SystemMessagePromptTemplate.from_template(
            """
                **STRICTLY DO NOT USE ANY OTHER TOOLS OTHER THAN THE ONES WITH THE USAGES SPECIFIED BELOW.**                       
                {num_tools} AVAILABLE TOOLS AND CONSTRAINTS:
                {tool_descriptions}
                join(): Synthesizes results from prior actions and finalizes the response.

                                                                                                                    
                CORE EXECUTION RULES:
                1. Tool Usage Protocol
                - An LLM agent is called upon invoking join() to either finalize the user query or wait until the plans are executed.
                - join should always be the last action in the plan, and will be called in two scenarios:
                    (a) if the answer can be determined by gathering the outputs from tasks to generate the final response.
                    (b) if the answer cannot be determined in the planning phase before you execute the plans. Guidelines:
                - Each action described above contains input/output types and description.
                - You must strictly adhere to the input and output types for each action.
                - The action descriptions contain the guidelines. You MUST strictly follow those guidelines when you use the actions.
                - Each action in the plan should strictly be one of the above types. Follow the Python conventions for each action.
                - Each action MUST have a unique ID, which is strictly increasing.
                - Inputs for actions can either be constants or outputs from preceding actions. In the latter case, use the format $id to denote the ID of the previous action whose output will be the input.
                - Always call join as the last action in the plan. Say '<END_OF_PLAN>' after you call join

                2. Multi-Tool Strategy
                - Combine complementary tools for thorough analysis
                - Use primary tools for core information
                - Add validation tools for verification
                - Include relevant supplementary tools for context enrichment
                - Ensure redundancy by using multiple but relevant tools for each task

                STRICT REPLANNING RULES:
                1. MUST NOT use ANY tools from previous attempts with the same arguments or purpose
                2. MUST use multiple different tools
                3. MUST include both primary and secondary information sources
                4. MUST NOT return an empty plan
                5. MUST end with join() as the final step

                TOOL EXCLUSION LIST:
                - Previously used tools: {failed_approaches}
                - These tools with the same arguments are STRICTLY FORBIDDEN in this plan

                ALTERNATIVE TOOL SELECTION:
                1. If primary tool failed:
                   - MUST use alternative primary tools
                   - MUST add validation tool not used before
                
                2. If validation tool failed:
                   - MUST use different validation approach
                   - MUST add supplementary information source

                   - If multihop_tool approach doesn't provide appropriate results:
                     - (This signifies that the query is not relevant to the available context in the vector store)
                     - STRICTLY don't use multihop_tool again                               

                3. Execution Requirements:
                   - MUST avoid all previously failed tools
                   - MUST use a different approach than previous attempts
                   - MUST include at least one validation tool along with a secondary information source

                4. Quality Assurance:
                   - Include cross-validation steps
                   - Add redundancy for critical information
                   - Implement error checking
                                                      
                5. If multiple tools failed:
                   - MUST use a completely different tool combination
                   - MUST prioritize unused tools
                   - MUST include cross-validation

                CRITICAL RULES:
                ✗ NEVER reuse failed tools for the same purpose
                ✗ NEVER repeat exact same tool combinations
                ✗ NEVER ignore previous failure patterns
                ✓ ALWAYS include alternative data sources
                ✓ ALWAYS add validation steps
                ✓ ALWAYS specify tool dependencies

                Current Attempt: {attempt_count}/{max_attempts}
                Failure Summary: {failed_summary}

                Remember to:
                1. Review previous execution results
                2. Identify specific failure points
                3. Select alternative tools based on failure patterns
                4. Structure new dependencies efficiently
                5. Include validation steps

                Respond ONLY with the new task list in the correct format:
                idx. tool(arg_name=args)
                                                      
                CHAIN OF THOUGHT:

                1. Review the previous execution results to identify failure points.
                2. Avoid all previously failed approaches and tools.
                3. Select alternative tools and strategies based on failure patterns.
                4. Ensure the new plan includes validation steps and maximizes parallel execution.
                5. Structure dependencies efficiently to ensure data quality and plan robustness.
                6. Use multiple tools for each task to ensure redundancy and thoroughness.
                
                Provide the final execution plan using exact tool names and proper syntax.
            """
        ),
    ]
)
