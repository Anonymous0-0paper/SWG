# ---------------------------
# Resilient Execution Module
# ---------------------------
import time
import logging
import random
import json
from typing import List, Dict, Any, Optional, Callable
from langchain_core.messages import HumanMessage

from utils import log_phase


class ResilientModelHandler:
    """
    Handles LLM API calls with retry logic, rate limit handling, and model fallback.
    """
    def __init__(self, models: List, max_retries: int = 5, base_delay: float = 1.0):
        """
        Initialize a ResilientModelHandler.

        :param models: List of LLM models to use
        :param max_retries: Maximum number of retry attempts
        :param base_delay: Base delay between retries (will be increased with backoff)
        """
        self.models = models
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.logger = logging.getLogger(__name__)
        self.model_usage = {i: 0 for i in range(len(models))}  # Track usage of each model

        # Initialize model provider tracking (for nice logging)
        self.model_providers = []
        for model in models:
            if hasattr(model, 'model'):
                if 'mistral' in model.model.lower():
                    self.model_providers.append('mistral')
                elif 'gpt' in model.model.lower() or 'openai' in model.model.lower():
                    self.model_providers.append('openai')
                elif 'claude' in model.model.lower() or 'anthropic' in model.model.lower():
                    self.model_providers.append('anthropic')
                elif 'command' in model.model.lower() or 'cohere' in model.model.lower():
                    self.model_providers.append('cohere')
                else:
                    self.model_providers.append('unknown')
            else:
                self.model_providers.append('unknown')

    def invoke_with_retry(self, prompt: str, default_result: Any = None) -> Any:
        """
        Invoke an LLM with retry logic and model fallback.

        :param prompt: The prompt to send to the LLM
        :param default_result: Default result to return if all attempts fail
        :return: The LLM response or default_result
        """
        # Create a jittered delay function to avoid thundering herd problem
        def get_delay(attempt: int) -> float:
            """Get delay with exponential backoff and jitter."""
            return self.base_delay * (2 ** attempt) * (0.5 + random.random())

        # Try each model in turn, with retries
        for primary_model_idx in self._get_least_used_models():
            model = self.models[primary_model_idx]
            provider = self.model_providers[primary_model_idx]

            # Attempt with exponential backoff
            for attempt in range(self.max_retries):
                try:
                    self.logger.info(f"Invoking {provider} model (attempt {attempt+1})")
                    response = model.invoke([HumanMessage(content=prompt)])

                    # Update usage counter for successful model
                    self.model_usage[primary_model_idx] += 1

                    return response.content
                except Exception as e:
                    error_msg = str(e).lower()

                    # Handle rate limit errors
                    if "rate limit" in error_msg or "429" in error_msg or "too many requests" in error_msg:
                        delay = get_delay(attempt)
                        self.logger.warning(
                            f"{provider} rate limit hit (attempt {attempt+1}/{self.max_retries}). "
                            f"Waiting {delay:.2f}s before retry."
                        )
                        time.sleep(delay)
                    # Handle quota errors
                    elif "quota" in error_msg or "billing" in error_msg:
                        self.logger.warning(f"{provider} quota exceeded. Trying next model.")
                        break  # Skip to next model
                    # Handle other errors
                    else:
                        self.logger.warning(
                            f"Error with {provider} model (attempt {attempt+1}/{self.max_retries}): {str(e)}"
                        )
                        # For other errors, wait less time
                        time.sleep(get_delay(attempt) * 0.5)

        # If we've exhausted all models and retries, return the default result
        self.logger.error("All models failed. Returning default result.")
        return default_result

    def _get_least_used_models(self) -> List[int]:
        """
        Get model indices ordered from least to most used.
        This helps balance usage across available models.

        :return: List of model indices
        """
        return sorted(range(len(self.models)), key=lambda i: self.model_usage[i])


class ResilientPlanExecutor:
    """
    Enhanced plan executor with resilient LLM calls.
    """
    def __init__(self, models: List, graph: 'GraphOfThoughts', streaming_system: str):
        """
        Initialize a ResilientPlanExecutor.

        :param models: List of LLM models to use
        :param graph: The current reasoning graph
        :param streaming_system: The selected streaming system
        """
        self.model_handler = ResilientModelHandler(models)
        self.graph = graph
        self.streaming_system = streaming_system
        self.context = {}  # Stores results from previous steps

        # Initialize the results saver
        from step_results_saver import create_results_saver
        self.results_saver = create_results_saver()

        # Logger setup
        self.logger = logging.getLogger(__name__)

    def execute_plan(self, plan: 'ExecutionPlan') -> str:
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

                try:
                    # Execute the step with resilient handling
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
                        java_path = self.results_saver.extract_and_save_java_code(
                            text=result,
                            filename_prefix=f"{step.action}_{step.step_id}"
                        )
                        if java_path:
                            log_phase(f"Extracted and saved Java code to: {java_path}")

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

                except Exception as e:
                    self.logger.error(f"Error executing step {step.step_id}: {str(e)}")
                    log_phase(f"Error executing step {step.step_id}: {str(e)}")

                    # Add error to graph
                    error_node_id = self.graph.add_node(
                        f"Error in step {step.step_id} ({step.action}): {str(e)}",
                        node_type="error"
                    )
                    log_phase(f"Added error node {error_node_id}")

                    # Create a simplified result for the failed step
                    fallback_result = f"Step execution failed: {str(e)}"
                    plan.mark_step_completed(step.step_id, fallback_result)
                    self.context[f"step_{step.step_id}_result"] = fallback_result

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

        # If synthesize_response failed or doesn't exist, create a fallback response
        log_phase("No valid synthesize_response step found, creating fallback response")

        # Generate a fallback response using the completed steps
        fallback_prompt = (
            f"Create a comprehensive response for this query: '{plan.query}'\n"
            f"About the streaming system: {self.streaming_system}\n\n"
            "Here are the results from our analysis steps (some might be incomplete):\n"
        )

        # Add results from completed steps
        for step in plan.steps:
            if step.is_completed and step.result:
                fallback_prompt += f"\n{step.action}: "

                # Truncate long results for the prompt
                result_text = str(step.result)
                if len(result_text) > 500:
                    result_text = result_text[:500] + "..."

                fallback_prompt += result_text

        fallback_prompt += "\n\nCreate a well-structured response that uses the available information."

        fallback_response = self.model_handler.invoke_with_retry(
            prompt=fallback_prompt,
            default_result="Unable to generate a response due to multiple failures. Please try again later."
        )

        # Save the fallback response
        self.results_saver.save_final_response(
            response=fallback_response,
            query=plan.query
        )

        return fallback_response

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

    def _execute_step(self, step: 'ActionStep', plan: 'ExecutionPlan') -> any:
        """
        Execute a single step in the plan with resilient handling.

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

    # Each of these methods now uses the resilient model handler

    def _analyze_complexity(self, query: str) -> Dict:
        """Analyze the complexity of a query with resilient handling"""

        # Try to parse JSON response
        import json
        import re

        prompt = (
            f"Analyze the complexity of this query: '{query}'\n\n"
            "Provide a complexity score (1-10) and identify key components that need to be addressed.\n"
            "Return the result as a JSON object with 'complexity_score' and 'key_components' fields."
        )

        response = self.model_handler.invoke_with_retry(
            prompt=prompt,
            default_result=json.dumps({
                "complexity_score": 5,
                "key_components": ["Input data", "Processing", "Output format"]
            })
        )

        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response

            return json.loads(json_str.strip())
        except:
            # Fallback if JSON parsing fails
            return {
                "complexity_score": 5,
                "key_components": ["Input data", "Processing", "Output format"]
            }

    def _gather_requirements(self, query: str, context: Dict) -> Dict:
        """Extract requirements from the query with resilient handling"""

        import json

        prompt = (
            f"Extract detailed requirements from this query: '{query}'\n\n"
            "Consider: data sources, data rate, latency requirements, processing needs, output destinations.\n"
            "Return a JSON object with the extracted requirements."
        )

        response = self.model_handler.invoke_with_retry(
            prompt=prompt,
            default_result=json.dumps({
                "data_sources": ["Kafka"],
                "processing_needs": ["Data transformation"],
                "output_destinations": ["File system"]
            })
        )

        # Try to parse JSON response
        import json
        import re

        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response

            return json.loads(json_str.strip())
        except:
            # Fallback if JSON parsing fails
            return {
                "data_sources": ["Kafka"],
                "processing_needs": ["Data transformation"],
                "output_destinations": ["File system"]
            }

    def _retrieve_documents(self, query: str, streaming_system: str) -> List[str]:
        """Retrieve relevant documents"""
        from utils import retrieve_relevant_documents
        documents = retrieve_relevant_documents(query, top_k=3, folder_path=f"Data/output/{streaming_system.lower()}")
        return documents

    def _design_architecture(self, query: str, context: Dict, streaming_system: str) -> str:
        """Design the pipeline architecture with resilient handling"""
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

        return self.model_handler.invoke_with_retry(
            prompt=prompt,
            default_result=f"Basic {streaming_system} architecture with Kafka source, processing operators, and file sink."
        )

    def _generate_code(self, query: str, context: Dict, streaming_system: str) -> str:
        """Generate pipeline code with resilient handling"""
        architecture = context.get("step_3_result", "")

        prompt = (
            f"Generate production-ready Java code for a {streaming_system} pipeline based on this architecture:\n"
            f"{architecture}\n\n"
            "Include all necessary imports, proper package structure, classes, and detailed comments.\n"
            "The code should be complete and ready to compile with minimal modifications.\n"
            "Use Java best practices and design patterns appropriate for stream processing.\n"
            "Format the response with ```java code blocks for easy extraction."
        )

        return self.model_handler.invoke_with_retry(
            prompt=prompt,
            default_result=f"// Basic {streaming_system} pipeline template\npublic class Pipeline"
        )

    def _generate_deployment(self, context: Dict, streaming_system: str) -> str:
        """Generate deployment instructions with resilient handling"""
        code = context.get("step_4_result", "")

        prompt = (
            f"Generate step-by-step deployment instructions for this {streaming_system} pipeline code:\n"
            f"{code[:1000]}...\n\n"  # Limit code length
            "Include environment setup, dependencies, configuration, and monitoring setup."
        )

        return self.model_handler.invoke_with_retry(
            prompt=prompt,
            default_result=f"Basic deployment steps for {streaming_system}:\n1. Set up environment\n2. Configure resources\n3. Deploy application"
        )

    def _synthesize_response(self, query: str, context: Dict, streaming_system: str) -> str:
        """Synthesize the final response with resilient handling"""
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

        return self.model_handler.invoke_with_retry(
            prompt=prompt,
            default_result=f"Here is a {streaming_system} solution for your requirements. [Basic solution description with placeholders for missing content]"
        )

# Import for convenience
import json

# Helper function to easily replace the standard PlanExecutor with this resilient version
def create_resilient_executor(models, graph, streaming_system):
    """
    Create a ResilientPlanExecutor instance.

    :param models: List of LLM models to use
    :param graph: The current reasoning graph
    :param streaming_system: The selected streaming system
    :return: A configured ResilientPlanExecutor
    """
    return ResilientPlanExecutor(models, graph, streaming_system)