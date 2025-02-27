# ---------------------------
# Query Analyzer and Planner
# ---------------------------
import json
import re
from typing import Dict, List, Tuple, Optional
from langchain_core.messages import HumanMessage, SystemMessage

# from deepGoT_main import GraphOfThoughts
from utils import log_phase, retrieve_relevant_documents


class QueryIntent:
    """Represents the detected intent of a user query."""
    def __init__(self, intent_type: str, confidence: float,
                 parameters: Optional[Dict[str, any]] = None):
        """
        Initialize a QueryIntent object.

        :param intent_type: The type of intent detected (e.g., "pipeline_design", "explanation", etc.)
        :param confidence: Confidence score (0-1) of the intent detection
        :param parameters: Additional parameters extracted from the query
        """
        self.intent_type = intent_type
        self.confidence = confidence
        self.parameters = parameters or {}

    def __repr__(self):
        return f"<QueryIntent: {self.intent_type} ({self.confidence:.2f})>"

class ActionStep:
    """Represents a single step in an execution plan."""
    def __init__(self, step_id: int, action: str,
                 description: str, dependencies: List[int] = None):
        """
        Initialize an ActionStep object.

        :param step_id: Unique identifier for this step
        :param action: The action to take (e.g., "retrieve_docs", "generate_code", etc.)
        :param description: Human-readable description of the step
        :param dependencies: List of step_ids that must be completed before this step
        """
        self.step_id = step_id
        self.action = action
        self.description = description
        self.dependencies = dependencies or []
        self.result = None  # Will store the result of executing this step
        self.is_completed = False

    def __repr__(self):
        status = "✓" if self.is_completed else "○"
        return f"<Step {self.step_id} {status}: {self.action}>"

class ExecutionPlan:
    """Represents a complete execution plan composed of ordered steps."""
    def __init__(self, query: str, steps: List[ActionStep] = None):
        """
        Initialize an ExecutionPlan object.

        :param query: The original user query
        :param steps: List of ActionStep objects
        """
        self.query = query
        self.steps = steps or []
        self.current_step_idx = 0

    def add_step(self, action: str, description: str, dependencies: List[int] = None) -> int:
        """Add a new step to the plan and return its ID."""
        step_id = len(self.steps)
        self.steps.append(ActionStep(step_id, action, description, dependencies))
        return step_id

    def get_next_executable_steps(self) -> List[ActionStep]:
        """Get all steps that are ready to be executed based on dependencies."""
        executable_steps = []
        for step in self.steps:
            if step.is_completed:
                continue

            dependencies_met = all(
                self.steps[dep_id].is_completed
                for dep_id in step.dependencies
            )

            if dependencies_met:
                executable_steps.append(step)

        return executable_steps

    def has_remaining_steps(self) -> bool:
        """Check if there are any incomplete steps."""
        return any(not step.is_completed for step in self.steps)

    def mark_step_completed(self, step_id: int, result: any = None):
        """Mark a step as completed and store its result."""
        if 0 <= step_id < len(self.steps):
            self.steps[step_id].is_completed = True
            self.steps[step_id].result = result

    def visualize(self) -> str:
        """Generate a text visualization of the execution plan."""
        visualization = [f"Execution Plan for: '{self.query}'"]

        for step in self.steps:
            status = "✓" if step.is_completed else "○"
            deps = f" (depends on: {', '.join(map(str, step.dependencies))})" if step.dependencies else ""
            visualization.append(f"{status} Step {step.step_id}: {step.action}{deps}")
            visualization.append(f"    {step.description}")
            if step.is_completed and step.result:
                result_preview = str(step.result)
                if len(result_preview) > 100:
                    result_preview = result_preview[:97] + "..."
                visualization.append(f"    Result: {result_preview}")

        return "\n".join(visualization)

class QueryAnalyzer:
    """
    Analyzes user queries and creates execution plans.
    Uses LLMs to understand user intent and plan a structured approach to answering queries.
    """
    def __init__(self, planning_model):
        """
        Initialize a QueryAnalyzer object.

        :param planning_model: An LLM to use for planning (from get_llm_model)
        """
        self.planning_model = planning_model
        self.intent_patterns = self._initialize_intent_patterns()

    def _initialize_intent_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Initialize regex patterns for basic intent detection."""
        return {
            "pipeline_design": [
                re.compile(r"design\s+(?:a|an)\s+pipeline", re.IGNORECASE),
                re.compile(r"create\s+(?:a|an)\s+(?:streaming|stream)\s+pipeline", re.IGNORECASE),
                re.compile(r"build\s+(?:a|an)\s+(?:data|streaming|stream|event)", re.IGNORECASE),
                re.compile(r"implement\s+(?:a|an)\s+(?:streaming|stream|real-time|realtime)", re.IGNORECASE)
            ],
            "explanation": [
                re.compile(r"explain\s+how", re.IGNORECASE),
                re.compile(r"how\s+does\s+(?:it|this|that)\s+work", re.IGNORECASE),
                re.compile(r"what\s+is\s+(?:a|an)\s+(?:streaming|stream|flink|kafka)", re.IGNORECASE),
                re.compile(r"tell\s+me\s+about", re.IGNORECASE)
            ],
            "comparison": [
                re.compile(r"compare\s+(?:\w+)\s+(?:and|vs|versus)\s+(?:\w+)", re.IGNORECASE),
                re.compile(r"difference\s+between", re.IGNORECASE),
                re.compile(r"pros\s+and\s+cons", re.IGNORECASE),
                re.compile(r"advantages\s+(?:and|or)\s+disadvantages", re.IGNORECASE)
            ],
            "troubleshooting": [
                re.compile(r"debug\s+(?:a|an|the|my)", re.IGNORECASE),
                re.compile(r"fix\s+(?:a|an|the|my)", re.IGNORECASE),
                re.compile(r"solve\s+(?:a|an|the|my)", re.IGNORECASE),
                re.compile(r"error\s+(?:in|with|when)", re.IGNORECASE),
                re.compile(r"not\s+working", re.IGNORECASE)
            ],
            "optimization": [
                re.compile(r"optimize\s+(?:a|an|the|my)", re.IGNORECASE),
                re.compile(r"improve\s+performance", re.IGNORECASE),
                re.compile(r"make\s+(?:it|this)\s+faster", re.IGNORECASE),
                re.compile(r"reduce\s+latency", re.IGNORECASE)
            ]
        }

    def detect_intent(self, query: str) -> QueryIntent:
        """
        Detect the intent of a user query using regex patterns.
        This is a simple first-pass approach before using LLM for deeper analysis.

        :param query: The user query
        :return: A QueryIntent object with the detected intent
        """
        max_confidence = 0.0
        detected_intent = "general_question"  # Default intent

        # First try with regex patterns
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    confidence = 0.7  # Base confidence for regex matches
                    if confidence > max_confidence:
                        max_confidence = confidence
                        detected_intent = intent_type

        # If confidence is still low, use LLM for more sophisticated intent detection
        if max_confidence < 0.7:
            return self._detect_intent_with_llm(query)

        return QueryIntent(detected_intent, max_confidence)

    def _detect_intent_with_llm(self, query: str) -> QueryIntent:
        """
        Use an LLM to detect query intent.

        :param query: The user query
        :return: A QueryIntent object with the detected intent
        """
        prompt = (
            "Analyze the following user query and determine the most likely intent. "
            "Respond with a JSON object containing 'intent_type' and 'confidence' (0-1).\n\n"
            f"User query: {query}\n\n"
            "Possible intent types: pipeline_design, explanation, comparison, troubleshooting, "
            "optimization, general_question\n\n"
            "JSON response:"
        )

        response = self.planning_model.invoke([HumanMessage(content=prompt)])

        # Extract JSON from response
        import json
        try:
            # Try to find a JSON block in the response
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.content

            json_str = json_str.strip()
            result = json.loads(json_str)

            return QueryIntent(
                intent_type=result.get("intent_type", "general_question"),
                confidence=result.get("confidence", 0.5),
                parameters=result.get("parameters", {})
            )
        except (json.JSONDecodeError, KeyError) as e:
            # If JSON parsing fails, return a default intent
            return QueryIntent("general_question", 0.5)

    def extract_parameters(self, query: str, intent: QueryIntent) -> Dict[str, any]:
        """
        Extract parameters from the query based on the detected intent.
        Includes error handling for rate limits and API failures.

        :param query: The user query
        :param intent: The detected QueryIntent
        :return: Dictionary of extracted parameters
        """
        prompt = (
            f"Extract relevant parameters from this {intent.intent_type} query: '{query}'\n\n"
            "Return a JSON object with parameters appropriate for this type of request. "
            "For example, for a pipeline_design intent, extract parameters like data_source, "
            "throughput_requirements, latency_requirements, etc.\n\n"
            "JSON response:"
        )

        # Default parameters based on intent type
        default_params = {
            "pipeline_design": {
                "data_source": "generic",
                "throughput": "medium",
                "latency_requirements": "standard",
                "output_destination": "generic"
            },
            "explanation": {
                "topic": query[:50] + "..." if len(query) > 50 else query,
                "depth": "medium"
            },
            "comparison": {
                "systems": ["system1", "system2"],
                "criteria": ["performance", "usability", "scalability"]
            },
            "troubleshooting": {
                "issue_type": "general",
                "severity": "medium"
            },
            "optimization": {
                "target": "performance",
                "constraint": "none"
            },
            "general_question": {
                "topic": query[:50] + "..." if len(query) > 50 else query
            }
        }

        # Try using the model, with fallback to regex extraction and defaults
        try:
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.planning_model.invoke([HumanMessage(content=prompt)])

                    # Try to parse JSON from response
                    import json
                    import re

                    try:
                        # Try to find a JSON block in the response
                        json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            json_str = response.content

                        json_str = json_str.strip()
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, fall back to regex extraction
                        return self._extract_params_with_regex(query, intent)

                except Exception as e:
                    # Log the error
                    import logging
                    logging.warning(f"Parameter extraction attempt {attempt+1} failed: {str(e)}")

                    if "rate limit" in str(e).lower() or "429" in str(e):
                        # Wait before retrying if we hit a rate limit
                        time.sleep(2 * (attempt + 1))  # Exponential backoff
                    elif attempt == max_retries - 1:
                        # If this was our last attempt, fall back to regex extraction
                        return self._extract_params_with_regex(query, intent)
                    else:
                        # For other errors, wait a bit and retry
                        time.sleep(1)
        except:
            # Final fallback to default parameters if all else fails
            return default_params.get(intent.intent_type, {})

    def _extract_params_with_regex(self, query: str, intent: QueryIntent) -> Dict[str, any]:
        """
        Extract parameters using regex patterns as a fallback when the LLM fails.

        :param query: The user query
        :param intent: The detected intent
        :return: Dictionary of extracted parameters
        """
        params = {}

        # Extract potential data sources
        import re

        # Data sources (e.g., Kafka, file, database)
        data_sources = re.findall(r'from\s+(\w+)', query, re.IGNORECASE)
        if data_sources:
            params["data_source"] = data_sources[0]

        # Throughput indicators
        if any(word in query.lower() for word in ["high volume", "large", "many", "millions"]):
            params["throughput"] = "high"
        elif any(word in query.lower() for word in ["small", "few", "tiny"]):
            params["throughput"] = "low"

        # Latency requirements
        if any(word in query.lower() for word in ["real-time", "realtime", "immediate", "instant"]):
            params["latency_requirements"] = "real-time"
        elif any(word in query.lower() for word in ["batch", "periodic", "daily"]):
            params["latency_requirements"] = "batch"

        # Output destinations
        output_matches = re.findall(r'to\s+(\w+)', query, re.IGNORECASE)
        if output_matches:
            params["output_destination"] = output_matches[0]

        return params

    def create_execution_plan(self, query: str, intent: QueryIntent,
                              streaming_system: str, use_rag: bool = True) -> ExecutionPlan:
        """
        Create an execution plan based on the query and detected intent.

        :param query: The user query
        :param intent: The detected QueryIntent
        :param streaming_system: The selected streaming system
        :param use_rag: Whether to use RAG in the plan
        :return: An ExecutionPlan object
        """
        plan = ExecutionPlan(query)

        # Common first step is to analyze query complexity
        complexity_step = plan.add_step(
            "analyze_complexity",
            "Analyze the complexity of the user query",
            []
        )

        # Different planning paths based on intent type
        if intent.intent_type == "pipeline_design":
            # For pipeline design, we need several steps

            # First gather requirements
            requirements_step = plan.add_step(
                "gather_requirements",
                "Extract and clarify pipeline requirements",
                [complexity_step]
            )

            # If using RAG, add a document retrieval step
            if use_rag:
                rag_step = plan.add_step(
                    "retrieve_documents",
                    f"Retrieve relevant documents for {streaming_system}",
                    [requirements_step]
                )
                prev_step = rag_step
            else:
                prev_step = requirements_step

            # Design the pipeline architecture
            architecture_step = plan.add_step(
                "design_architecture",
                f"Design the overall architecture for {streaming_system} pipeline",
                [prev_step]
            )

            # Generate the pipeline code
            code_gen_step = plan.add_step(
                "generate_code",
                f"Generate pipeline code for {streaming_system}",
                [architecture_step]
            )

            # Generate deployment instructions
            deployment_step = plan.add_step(
                "generate_deployment",
                "Generate deployment and operation instructions",
                [code_gen_step]
            )

            # Final response synthesis
            plan.add_step(
                "synthesize_response",
                "Create the final comprehensive response",
                [deployment_step]
            )

        elif intent.intent_type == "explanation":
            # For explanations, we focus on retrieving and presenting information

            if use_rag:
                rag_step = plan.add_step(
                    "retrieve_documents",
                    f"Retrieve explanatory documents about {streaming_system}",
                    [complexity_step]
                )
                prev_step = rag_step
            else:
                prev_step = complexity_step

            # Generate the explanation
            explanation_step = plan.add_step(
                "generate_explanation",
                f"Generate a clear explanation about {streaming_system}",
                [prev_step]
            )

            # Add examples if needed
            examples_step = plan.add_step(
                "provide_examples",
                "Provide illustrative examples",
                [explanation_step]
            )

            # Final response synthesis
            plan.add_step(
                "synthesize_response",
                "Create the final explanatory response",
                [examples_step]
            )

        elif intent.intent_type == "comparison":
            # For comparisons between streaming systems

            if use_rag:
                rag_step = plan.add_step(
                    "retrieve_comparison_docs",
                    "Retrieve documents comparing stream processing systems",
                    [complexity_step]
                )
                prev_step = rag_step
            else:
                prev_step = complexity_step

            # Extract comparison criteria
            criteria_step = plan.add_step(
                "identify_criteria",
                "Identify key comparison criteria",
                [prev_step]
            )

            # Perform the comparison
            comparison_step = plan.add_step(
                "perform_comparison",
                "Compare systems across identified criteria",
                [criteria_step]
            )

            # Final response synthesis
            plan.add_step(
                "synthesize_response",
                "Create the final comparison response",
                [comparison_step]
            )

        elif intent.intent_type == "troubleshooting":
            # For troubleshooting pipeline issues

            # Identify the problem first
            problem_step = plan.add_step(
                "identify_problem",
                "Identify the specific problem to troubleshoot",
                [complexity_step]
            )

            if use_rag:
                rag_step = plan.add_step(
                    "retrieve_solution_docs",
                    f"Retrieve troubleshooting documents for {streaming_system}",
                    [problem_step]
                )
                prev_step = rag_step
            else:
                prev_step = problem_step

            # Generate possible solutions
            solutions_step = plan.add_step(
                "generate_solutions",
                "Generate potential solutions",
                [prev_step]
            )

            # Final response synthesis
            plan.add_step(
                "synthesize_response",
                "Create the final troubleshooting response",
                [solutions_step]
            )

        elif intent.intent_type == "optimization":
            # For optimization requests

            # Identify optimization targets
            targets_step = plan.add_step(
                "identify_targets",
                "Identify specific optimization targets",
                [complexity_step]
            )

            if use_rag:
                rag_step = plan.add_step(
                    "retrieve_optimization_docs",
                    f"Retrieve optimization best practices for {streaming_system}",
                    [targets_step]
                )
                prev_step = rag_step
            else:
                prev_step = targets_step

            # Generate optimization strategies
            strategies_step = plan.add_step(
                "generate_strategies",
                "Generate optimization strategies",
                [prev_step]
            )

            # Final response synthesis
            plan.add_step(
                "synthesize_response",
                "Create the final optimization response",
                [strategies_step]
            )

        else:  # general_question or any other intent
            # For general questions, keep it simple

            if use_rag:
                rag_step = plan.add_step(
                    "retrieve_documents",
                    f"Retrieve relevant documents about {streaming_system}",
                    [complexity_step]
                )
                prev_step = rag_step
            else:
                prev_step = complexity_step

            # Generate the answer
            plan.add_step(
                "synthesize_response",
                "Generate a direct answer to the query",
                [prev_step]
            )

        return plan

class PlanExecutor:
    """
    Executes the steps in an execution plan using the appropriate models and tools.
    """
    def __init__(self, models: List, graph: 'GraphOfThoughts', streaming_system: str):
        """
        Initialize a PlanExecutor object.

        :param models: List of LLM models to use for execution
        :param graph: The current reasoning graph
        :param streaming_system: The selected streaming system
        """
        self.models = models
        self.graph = graph
        self.streaming_system = streaming_system
        self.context = {}  # Stores results from previous steps

        # Initialize the results saver
        from step_results_saver import create_results_saver
        self.results_saver = create_results_saver()

    # def execute_plan(self, plan: ExecutionPlan) -> str:
    #     """
    #     Execute all steps in the execution plan and return the final result.
    #
    #     :param plan: The ExecutionPlan to execute
    #     :return: The final response string
    #     """
    #     log_phase("=== Plan Execution Phase ===")
    #     log_phase(plan.visualize())
    #
    #     while plan.has_remaining_steps():
    #         executable_steps = plan.get_next_executable_steps()
    #
    #         for step in executable_steps:
    #             log_phase(f"Executing step {step.step_id}: {step.action}")
    #             result = self._execute_step(step, plan)
    #             plan.mark_step_completed(step.step_id, result)
    #
    #             # Store the result in context for future steps
    #             self.context[f"step_{step.step_id}_result"] = result
    #
    #             # Add result to the graph
    #             node_id = self.graph.add_node(
    #                 f"Step {step.step_id} ({step.action}) result: {str(result)[:200]}...",
    #                 node_type="plan_step"
    #             )
    #             log_phase(f"Added plan step node {node_id}")
    #
    #     # The final step should be a synthesize_response step
    #     final_steps = [s for s in plan.steps if s.action == "synthesize_response"]
    #     if final_steps:
    #         return final_steps[0].result
    #
    #     # Fallback if no synthesize_response step exists
    #     return "Plan execution complete, but no final response was generated."
    def execute_plan(self, plan: ExecutionPlan) -> str:
        """
        Execute all steps in the execution plan and return the final result.

        :param plan: The ExecutionPlan to execute
        :return: The final response string
        """
        log_phase("=== Plan Execution Phase ===")
        log_phase(plan.visualize())

        # Save the execution plan visualization
        self.results_saver.save_execution_plan(plan.visualize())

        while plan.has_remaining_steps():
            executable_steps = plan.get_next_executable_steps()

            for step in executable_steps:
                log_phase(f"Executing step {step.step_id}: {step.action}")

                # Execute the step
                result = self._execute_step(step, plan)

                # Save the step result
                self.results_saver.save_step_result(
                    step_id=step.step_id,
                    step_name=step.action,
                    result=result
                )

                # Check if this step might contain Java code and save it separately
                if isinstance(result, str) and step.action in [
                    "generate_code",
                    "design_architecture",
                    "generate_deployment",
                    "synthesize_response"
                ]:
                    # Try to extract and save Java code
                    self.results_saver.extract_and_save_java_code(
                        text=result,
                        filename_prefix=f"{step.action}_{step.step_id}"
                    )

                # Mark the step as completed and store its result
                plan.mark_step_completed(step.step_id, result)

                # Store the result in context for future steps
                self.context[f"step_{step.step_id}_result"] = result

                # Add result to the graph
                node_id = self.graph.add_node(
                    f"Step {step.step_id} ({step.action}) result: {str(result)[:200]}...",
                    node_type="plan_step"
                )
                log_phase(f"Added plan step node {node_id}")

        # The final step should be a synthesize_response step
        final_steps = [s for s in plan.steps if s.action == "synthesize_response"]
        if final_steps and final_steps[0].is_completed:
            final_response = final_steps[0].result

            # Save the final response
            self.results_saver.save_final_response(
                response=final_response,
                query=plan.query
            )

            return final_response

        # Fallback if no synthesize_response step exists
        return "Plan execution complete, but no final response was generated."

    def save_session_summary(self, query: str, intent: str):
        """
        Save a summary of the completed session.

        :param query: The original query
        :param intent: The detected intent
        """
        return self.results_saver.save_session_summary(
            query=query,
            intent=intent,
            streaming_system=self.streaming_system
        )

    # def _execute_step(self, step: ActionStep, plan: ExecutionPlan) -> any:
    #     """
    #     Execute a single step in the plan.
    #
    #     :param step: The ActionStep to execute
    #     :param plan: The current ExecutionPlan (for context)
    #     :return: The result of executing the step
    #     """
    #     # Get dependency results
    #     dependency_results = {
    #         f"step_{dep_id}_result": plan.steps[dep_id].result
    #         for dep_id in step.dependencies
    #         if plan.steps[dep_id].is_completed
    #     }
    #
    #     # Execute different actions based on the step type
    #     if step.action == "analyze_complexity":
    #         return self._analyze_complexity(plan.query)
    #
    #     elif step.action == "gather_requirements":
    #         return self._gather_requirements(plan.query, dependency_results)
    #
    #     elif step.action == "retrieve_documents":
    #         return self._retrieve_documents(plan.query, self.streaming_system)
    #
    #     elif step.action == "design_architecture":
    #         return self._design_architecture(plan.query, dependency_results, self.streaming_system)
    #
    #     elif step.action == "generate_code":
    #         return self._generate_code(plan.query, dependency_results, self.streaming_system)
    #
    #     # elif step.action == "generate_deployment":
    #     #     return self._generate_deployment(dependency_results, self.streaming_system)
    #
    #     elif step.action == "generate_explanation":
    #         return self._generate_explanation(plan.query, dependency_results, self.streaming_system)
    #
    #     elif step.action == "provide_examples":
    #         return self._provide_examples(dependency_results, self.streaming_system)
    #
    #     elif step.action == "identify_problem":
    #         return self._identify_problem(plan.query)
    #     elif step.action == "generate_solutions":
    #         return self._generate_solutions(dependency_results, self.streaming_system)
    #
    #     elif step.action == "identify_criteria":
    #         return self._identify_criteria(plan.query, dependency_results)
    #
    #     elif step.action == "perform_comparison":
    #         return self._perform_comparison(plan.query, dependency_results)
    #
    #     elif step.action == "identify_targets":
    #         return self._identify_targets(plan.query)
    #
    #     elif step.action == "generate_strategies":
    #         return self._generate_strategies(dependency_results, self.streaming_system)
    #
    #     elif step.action == "synthesize_response":
    #         return self._synthesize_response(plan.query, dependency_results, self.streaming_system)
    #
    #     else:
    #         # Default handler for unknown action types
    #         return f"Unknown action: {step.action}"

    def _execute_step(self, step: ActionStep, plan: ExecutionPlan) -> any:
        """
        Execute a single step in the plan.

        :param step: The ActionStep to execute
        :param plan: The current ExecutionPlan (for context)
        :return: The result of executing the step
        """
        # Get dependency results
        dependency_results = {
            f"step_{dep_id}_result": plan.steps[dep_id].result
            for dep_id in step.dependencies
            if plan.steps[dep_id].is_completed
        }

        # Execute different actions based on the step type
        if step.action == "analyze_complexity":
            return self._analyze_complexity(plan.query)

        elif step.action == "gather_requirements":
            return self._gather_requirements(plan.query, dependency_results)

        elif step.action == "retrieve_documents":
            return self._retrieve_documents(plan.query, self.streaming_system)

        elif step.action == "design_architecture":
            return self._design_architecture(plan.query, dependency_results, self.streaming_system)

        elif step.action == "generate_code":
            return self._generate_code(plan.query, dependency_results, self.streaming_system)

        elif step.action == "generate_deployment":
            return self._generate_deployment(dependency_results, self.streaming_system)

        elif step.action == "generate_explanation":
            return self._generate_explanation(plan.query, dependency_results, self.streaming_system)

        elif step.action == "provide_examples":
            return self._provide_examples(dependency_results, self.streaming_system)

        elif step.action == "identify_problem":
            return self._identify_problem(plan.query)

        elif step.action == "generate_solutions":
            return self._generate_solutions(dependency_results, self.streaming_system)

        elif step.action == "identify_criteria":
            return self._identify_criteria(plan.query, dependency_results)

        elif step.action == "perform_comparison":
            return self._perform_comparison(plan.query, dependency_results)

        elif step.action == "identify_targets":
            return self._identify_targets(plan.query)

        elif step.action == "generate_strategies":
            return self._generate_strategies(dependency_results, self.streaming_system)

        elif step.action == "synthesize_response":
            return self._synthesize_response(plan.query, dependency_results, self.streaming_system)

        else:
            # Default handler for unknown action types
            return f"Unknown action: {step.action}"

    def _analyze_complexity(self, query: str) -> Dict:
        """Analyze the complexity of a query"""
        prompt = (
            f"Analyze the complexity of this query: '{query}'\n\n"
            "Provide a complexity score (1-10) and identify key components that need to be addressed.\n"
            "Return the result as a JSON object with 'complexity_score' and 'key_components' fields."
        )

        response = self.models[0].invoke([HumanMessage(content=prompt)])

        # Try to parse JSON response
        import json
        import re

        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.content

            return json.loads(json_str.strip())
        except:
            # Fallback if JSON parsing fails
            return {
                "complexity_score": 5,
                "key_components": ["General query processing"]
            }

    def _gather_requirements(self, query: str, context: Dict) -> Dict:
        """Extract requirements from the query"""
        prompt = (
            f"Extract detailed requirements from this query: '{query}'\n\n"
            "Consider: data sources, data rate, latency requirements, processing needs, output destinations.\n"
            "Return a JSON object with the extracted requirements."
        )

        response = self.models[0].invoke([HumanMessage(content=prompt)])

        # Try to parse JSON response
        import json
        import re

        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.content

            return json.loads(json_str.strip())
        except:
            # Fallback if JSON parsing fails
            return {
                "data_sources": ["Unknown"],
                "processing_needs": ["General processing"],
                "output_destinations": ["Default output"]
            }

    def _retrieve_documents(self, query: str, streaming_system: str) -> List[str]:
        """Retrieve relevant documents"""
        documents = retrieve_relevant_documents(query, top_k=3, folder_path=f"Data/output/{streaming_system.lower()}")
        return documents

    def _design_architecture(self, query: str, context: Dict, streaming_system: str) -> str:
        """Design the pipeline architecture"""
        requirements = context.get("step_1_result", {})
        documents = context.get("step_2_result", [])

        # Prepare the document context
        doc_context = "\n\n".join(documents) if documents else ""

        prompt = (
            f"Design a stream processing pipeline architecture for {streaming_system} based on these requirements:\n"
            f"{json.dumps(requirements, indent=2)}\n\n"
            f"Additional context from documentation:\n{doc_context}\n\n"
            "Provide a detailed architecture with components and their interactions."
        )

        response = self.models[0].invoke([HumanMessage(content=prompt)])
        return response.content

    def _generate_code(self, query: str, context: Dict, streaming_system: str) -> str:
        """Generate pipeline code with emphasis on Java for stream processing"""
        architecture = context.get("step_3_result", "")

        prompt = (
            f"Generate production-ready Java code for a {streaming_system} pipeline based on this architecture:\n"
            f"{architecture}\n\n"
            "Include all necessary imports, proper package structure, classes, and detailed comments.\n"
            "The code should be complete and ready to compile with minimal modifications.\n"
            "Use Java best practices and design patterns appropriate for stream processing.\n"
            "Format the response with ```java code blocks for easy extraction."
        )

        response = self.models[0].invoke([HumanMessage(content=prompt)])
        return response.content

    def _generate_deployment(self, context: Dict, streaming_system: str) -> str:
        """Generate deployment instructions"""
        code = context.get("step_4_result", "")

        prompt = (
            f"Generate step-by-step deployment instructions for this {streaming_system} pipeline code:\n"
            f"{code[:1000]}...\n\n"  # Limit code length
            "Include environment setup, dependencies, configuration, and monitoring setup."
        )

        response = self.models[0].invoke([HumanMessage(content=prompt)])
        return response.content

    def _generate_explanation(self, query: str, context: Dict, streaming_system: str) -> str:
        """Generate an explanation"""
        documents = context.get("step_1_result", [])

        # Prepare the document context
        doc_context = "\n\n".join(documents) if documents else ""

        prompt = (
            f"Explain {streaming_system} concepts related to this query: '{query}'\n\n"
            f"Additional context from documentation:\n{doc_context}\n\n"
            "Provide a clear and comprehensive explanation."
        )

        response = self.models[0].invoke([HumanMessage(content=prompt)])
        return response.content

    def _provide_examples(self, context: Dict, streaming_system: str) -> str:
        """Provide examples"""
        explanation = context.get("step_2_result", "")

        prompt = (
            f"Based on this explanation about {streaming_system}:\n"
            f"{explanation[:1000]}...\n\n"  # Limit explanation length
            "Provide 2-3 concrete examples that illustrate these concepts."
        )

        response = self.models[0].invoke([HumanMessage(content=prompt)])
        return response.content

    def _identify_problem(self, query: str) -> Dict:
        """Identify the problem to troubleshoot"""
        prompt = (
            f"Identify the specific problem in this troubleshooting query: '{query}'\n\n"
            "Return a JSON object with 'problem_type', 'symptoms', and 'possible_causes' fields."
        )

        response = self.models[0].invoke([HumanMessage(content=prompt)])

        # Try to parse JSON response
        import json
        import re

        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.content

            return json.loads(json_str.strip())
        except:
            # Fallback if JSON parsing fails
            return {
                "problem_type": "Unknown",
                "symptoms": ["Issue with stream processing"],
                "possible_causes": ["Configuration", "Code error"]
            }

    def _generate_solutions(self, context: Dict, streaming_system: str) -> List[Dict]:
        """Generate solutions to a problem"""

        import json
        problem = context.get("step_1_result", {})
        documents = context.get("step_2_result", [])

        # Prepare the document context
        doc_context = "\n\n".join(documents) if documents else ""

        prompt = (
            f"Generate solutions for this {streaming_system} problem:\n"
            f"{json.dumps(problem, indent=2)}\n\n"
            f"Additional context from documentation:\n{doc_context}\n\n"
            "Return a JSON array of solution objects, each with 'solution_title', 'steps', and 'rationale' fields."
        )

        response = self.models[0].invoke([HumanMessage(content=prompt)])

        # Try to parse JSON response
        import json
        import re

        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.content

            return json.loads(json_str.strip())
        except:
            # Fallback if JSON parsing fails
            return [{
                "solution_title": "General troubleshooting",
                "steps": ["Check configuration", "Restart service", "Check logs"],
                "rationale": "These are standard troubleshooting steps."
            }]

    def _identify_criteria(self, query: str, context: Dict) -> List[str]:
        """Identify comparison criteria"""
        prompt = (
            f"Identify key criteria for comparing stream processing systems in this query: '{query}'\n\n"
            "Return a JSON array of criteria strings."
        )

        response = self.models[0].invoke([HumanMessage(content=prompt)])

        # Try to parse JSON response
        import json
        import re

        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.content

            return json.loads(json_str.strip())
        except:
            # Fallback if JSON parsing fails
            return ["Performance", "Ease of use", "Community support", "Features"]

    def _perform_comparison(self, query: str, context: Dict) -> Dict:
        """Perform a comparison between systems"""
        criteria = context.get("step_2_result", [])
        documents = context.get("step_1_result", [])

        # Extract system names from query
        import re
        systems = re.findall(r'(?:compare|versus|vs\.?|and)\s+([A-Za-z0-9]+)', query)

        # Prepare the document context
        doc_context = "\n\n".join(documents) if documents else ""

        prompt = (
            f"Compare these stream processing systems: {', '.join(systems)}\n"
            f"Use these criteria: {', '.join(criteria)}\n\n"
            f"Additional context from documentation:\n{doc_context}\n\n"
            "Return a JSON object with each system as a key, and an object of criteria ratings as values."
        )

        response = self.models[0].invoke([HumanMessage(content=prompt)])

        # Try to parse JSON response
        import json
        import re

        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.content

            return json.loads(json_str.strip())
        except:
            # Fallback if JSON parsing fails
            return {
                system: {
                    criterion: "Average" for criterion in criteria
                } for system in systems
            }

    def _identify_targets(self, query: str) -> List[str]:
        """Identify optimization targets"""
        prompt = (
            f"Identify optimization targets in this query: '{query}'\n\n"
            "Return a JSON array of target strings (e.g., 'latency', 'throughput', etc.)."
        )

        response = self.models[0].invoke([HumanMessage(content=prompt)])

        # Try to parse JSON response
        import json
        import re

        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.content

            return json.loads(json_str.strip())
        except:
            # Fallback if JSON parsing fails
            return ["Performance", "Resource usage", "Reliability"]

    def _generate_strategies(self, context: Dict, streaming_system: str) -> List[Dict]:
        """Generate optimization strategies"""
        targets = context.get("step_1_result", [])
        documents = context.get("step_2_result", [])

        # Prepare the document context
        doc_context = "\n\n".join(documents) if documents else ""

        prompt = (
            f"Generate optimization strategies for {streaming_system} targeting: {', '.join(targets)}\n\n"
            f"Additional context from documentation:\n{doc_context}\n\n"
            "Return a JSON array of strategy objects, each with 'name', 'description', and 'implementation' fields."
        )

        response = self.models[0].invoke([HumanMessage(content=prompt)])

        # Try to parse JSON response
        import json
        import re

        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.content

            return json.loads(json_str.strip())
        except:
            # Fallback if JSON parsing fails
            return [{
                "name": "General optimization",
                "description": "Improve overall performance",
                "implementation": "Optimize resource allocation and configurations."
            }]

    def _synthesize_response(self, query: str, context: Dict, streaming_system: str) -> str:
        """Synthesize the final response"""
        # Gather all results from previous steps
        all_results = context.copy()

        # Create a unified prompt based on the intent and available results
        prompt_parts = [
            f"Based on the query: '{query}'",
            f"About the streaming system: {streaming_system}",
        ]

        # Add specific result sections based on what's available
        if "step_5_result" in all_results:  # Pipeline design flow
            prompt_parts.extend([
                f"With the architecture: {all_results.get('step_3_result', '')[:500]}...",
                f"And the code implementation: {all_results.get('step_4_result', '')[:500]}...",
                f"And deployment instructions: {all_results.get('step_5_result', '')[:500]}...",
                "Generate a comprehensive and well-structured final response that includes architecture overview, code snippets, and deployment steps."
            ])
        elif "step_3_result" in all_results:  # Explanation flow
            prompt_parts.extend([
                f"With the explanation: {all_results.get('step_2_result', '')[:500]}...",
                f"And examples: {all_results.get('step_3_result', '')[:500]}...",
                "Generate a comprehensive and well-structured final response that explains the concepts clearly and includes helpful examples."
            ])
        elif "step_3_result" in all_results and "comparison" in query.lower():  # Comparison flow
            prompt_parts.extend([
                f"With the comparison results: {json.dumps(all_results.get('step_3_result', {}), indent=2)}",
                "Generate a comprehensive and well-structured final response that compares the systems clearly across all criteria."
            ])
        elif "step_3_result" in all_results and any(x in query.lower() for x in ["problem", "error", "issue", "fix"]):  # Troubleshooting flow
            prompt_parts.extend([
                f"With the identified problem: {json.dumps(all_results.get('step_1_result', {}), indent=2)}",
                f"And the solutions: {json.dumps(all_results.get('step_3_result', []), indent=2)[:500]}...",
                "Generate a comprehensive and well-structured final response that addresses the problem and provides clear solutions."
            ])
        elif "step_3_result" in all_results and any(x in query.lower() for x in ["optimize", "performance", "improve"]):  # Optimization flow
            prompt_parts.extend([
                f"With the optimization targets: {', '.join(all_results.get('step_1_result', []))}",
                f"And the strategies: {json.dumps(all_results.get('step_3_result', []), indent=2)[:500]}...",
                "Generate a comprehensive and well-structured final response that outlines the optimization strategies clearly."
            ])
        else:  # General flow
            docs = all_results.get("step_1_result", [])
            doc_summary = "\n".join(docs)[:1000] if docs else ""
            prompt_parts.extend([
                f"With the retrieved information: {doc_summary}",
                "Generate a comprehensive and well-structured final response that directly answers the query."
            ])

        prompt = "\n\n".join(prompt_parts)

        response = self.models[0].invoke([HumanMessage(content=prompt)])
        return response.content