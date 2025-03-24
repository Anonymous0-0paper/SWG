import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import required libraries from existing code
from langchain_core.messages import HumanMessage, SystemMessage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
VALIDATION_DIR = "validation_history"
PROMPT_TEMPLATES_DIR = "prompt_templates"
os.makedirs(VALIDATION_DIR, exist_ok=True)
os.makedirs(PROMPT_TEMPLATES_DIR, exist_ok=True)

# ---------------------------
# Validator Base Class
# ---------------------------
class StreamProcessingValidator:
    """Base class for stream processing code validators."""

    def __init__(self, framework: str):
        """
        Initialize validator for a specific streaming framework.

        Args:
            framework: Name of the stream processing framework (e.g., "Apache Flink")
        """
        self.framework = framework
        self.rules = self._load_framework_rules()

    def _load_framework_rules(self) -> Dict[str, Any]:
        """Load validation rules specific to the framework."""
        rules_path = f"validation_rules/{self.framework.lower().replace(' ', '_')}.json"
        try:
            if os.path.exists(rules_path):
                with open(rules_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"No specific rules found for {self.framework}. Using default rules.")
                return self._get_default_rules()
        except Exception as e:
            logger.error(f"Error loading rules: {str(e)}")
            return self._get_default_rules()

    def _get_default_rules(self) -> Dict[str, Any]:
        """Return default validation rules."""
        return {
            "patterns": {
                "stateful_operations": r"(keyBy|window|process|aggregate|fold|reduce)",
                "error_handling": r"(try|catch|exception|error|retry)",
                "checkpointing": r"(enableCheckpointing|CheckpointConfig|StateBackend)"
            },
            "anti_patterns": {
                "unbounded_state": r"(Map|FlatMap).+without.+cleaning",
                "blocking_operations": r"(Thread.sleep|wait\(|join\()",
                "inefficient_joins": r"(CartesianProduct|crossWithHuge)"
            }
        }

    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Validate stream processing code.

        Args:
            code: The code to validate

        Returns:
            Dict containing validation results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "framework": self.framework,
            "passed": True,
            "issues": [],
            "warnings": [],
            "suggestions": []
        }

        # Check for required patterns
        for pattern_name, pattern in self.rules["patterns"].items():
            if not re.search(pattern, code):
                results["warnings"].append(f"Missing {pattern_name} pattern")

        # Check for anti-patterns
        for anti_pattern_name, pattern in self.rules["anti_patterns"].items():
            if re.search(pattern, code):
                results["issues"].append(f"Found {anti_pattern_name} anti-pattern")
                results["passed"] = False

        # Add framework-specific validations
        self._framework_specific_validations(code, results)

        return results

    def _framework_specific_validations(self, code: str, results: Dict[str, Any]):
        """Framework-specific validations to be implemented by subclasses."""
        pass


# ---------------------------
# Framework-Specific Validators
# ---------------------------
class FlinkValidator(StreamProcessingValidator):
    """Validator for Apache Flink stream processing code."""

    def __init__(self):
        super().__init__("Apache Flink")

    def _framework_specific_validations(self, code: str, results: Dict[str, Any]):
        # Check for checkpointing configuration
        if "enableCheckpointing" in code and not re.search(r"enableCheckpointing\([^)]+\)", code):
            results["issues"].append("Checkpointing enabled without specifying interval")
            results["passed"] = False

        # Check for proper parallelism configuration
        if "setParallelism" in code and re.search(r"setParallelism\(1\)", code):
            results["warnings"].append("Pipeline uses parallelism of 1, which may limit scalability")

        # Check for watermark strategy
        if "timeWindow" in code and not "WatermarkStrategy" in code:
            results["warnings"].append("Time windows used without explicit watermark strategy")


class SparkValidator(StreamProcessingValidator):
    """Validator for Apache Spark Streaming code."""

    def __init__(self):
        super().__init__("Apache Spark")

    def _framework_specific_validations(self, code: str, results: Dict[str, Any]):
        # Check for checkpoint directory
        if "streamingContext" in code and not re.search(r"checkpointDir", code):
            results["warnings"].append("StreamingContext used without checkpoint directory")

        # Check for proper batch interval
        if re.search(r"StreamingContext\([^,]+,\s*Seconds\((\d+)\)", code):
            batch_interval = re.search(r"StreamingContext\([^,]+,\s*Seconds\((\d+)\)", code).group(1)
            if int(batch_interval) < 1:
                results["warnings"].append(f"Very small batch interval ({batch_interval}s) may cause performance issues")


class KafkaStreamsValidator(StreamProcessingValidator):
    """Validator for Kafka Streams code."""

    def __init__(self):
        super().__init__("Kafka Streams")

    def _framework_specific_validations(self, code: str, results: Dict[str, Any]):
        # Check for proper exception handler
        if not re.search(r"setUncaughtExceptionHandler", code):
            results["warnings"].append("No uncaught exception handler defined")

        # Check for proper state store configuration
        if "Materialized" in code and not re.search(r"withCachingEnabled|withLoggingEnabled", code):
            results["warnings"].append("State store used without explicit caching/logging configuration")


# ---------------------------
# Validation Factory
# ---------------------------
def get_validator(framework: str) -> StreamProcessingValidator:
    """
    Factory method to get the appropriate validator for a framework.

    Args:
        framework: The streaming framework name

    Returns:
        A validator instance for the specified framework
    """
    validators = {
        "Apache Flink": FlinkValidator,
        "Apache Spark": SparkValidator,
        "Kafka Streams": KafkaStreamsValidator
    }

    validator_class = validators.get(framework)
    if validator_class:
        return validator_class()
    else:
        logger.info(f"No specific validator for {framework}. Using generic validator.")
        return StreamProcessingValidator(framework)


# ---------------------------
# Feedback Loop System
# ---------------------------
class FeedbackLoopSystem:
    """
    System that provides validation feedback to the LLM and enables code revision.
    """

    def __init__(self, llm_model, streaming_system: str):
        """
        Initialize the feedback loop system.

        Args:
            llm_model: The LLM model to use for code generation and improvement
            streaming_system: The streaming system being used
        """
        self.llm_model = llm_model
        self.streaming_system = streaming_system
        self.validator = get_validator(streaming_system)
        self.history_path = os.path.join(VALIDATION_DIR, f"{streaming_system.lower().replace(' ', '_')}_history.json")
        self.improvement_history = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load improvement history if it exists."""
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def _save_history(self):
        """Save improvement history."""
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        with open(self.history_path, 'w') as f:
            json.dump(self.improvement_history, f, indent=2)

    def validate_and_improve(self, code: str, task_description: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        Validate code and improve it based on feedback.

        Args:
            code: The code to validate and improve
            task_description: Description of the task the code should accomplish
            max_iterations: Maximum number of improvement iterations

        Returns:
            Dict containing improved code and validation history
        """
        iteration_history = []
        current_code = code

        for iteration in range(max_iterations):
            logger.info(f"Validation iteration {iteration + 1}/{max_iterations}")

            # Validate current code
            validation_results = self.validator.validate_code(current_code)

            # Record this iteration
            iteration_record = {
                "iteration": iteration + 1,
                "code": current_code,
                "validation_results": validation_results,
                "timestamp": datetime.now().isoformat()
            }
            iteration_history.append(iteration_record)

            # If no issues, we're done
            if validation_results["passed"] and not validation_results["warnings"]:
                logger.info("Code passed validation with no issues or warnings")
                break

            # Generate feedback for the LLM
            feedback = self._generate_feedback(validation_results)
            logger.info(f"Generated feedback: {feedback}")

            # Request improved code from LLM
            improved_code = self._request_improved_code(current_code, feedback, task_description)

            # Update current code for next iteration
            current_code = improved_code

        # Record overall improvement journey
        improvement_record = {
            "task_description": task_description,
            "initial_code": code,
            "final_code": current_code,
            "iterations": iteration_history,
            "timestamp": datetime.now().isoformat()
        }

        # Update and save history
        self.improvement_history.append(improvement_record)
        self._save_history()

        # Compute improvement metrics
        improvement_metrics = self._calculate_improvement_metrics(improvement_record)

        return {
            "improved_code": current_code,
            "history": iteration_history,
            "metrics": improvement_metrics
        }

    def _generate_feedback(self, validation_results: Dict[str, Any]) -> str:
        """Generate human-readable feedback from validation results."""
        feedback_parts = []

        if not validation_results["passed"]:
            feedback_parts.append("Critical issues that must be fixed:")
            for issue in validation_results["issues"]:
                feedback_parts.append(f"- {issue}")

        if validation_results["warnings"]:
            feedback_parts.append("Suggestions for improvement:")
            for warning in validation_results["warnings"]:
                feedback_parts.append(f"- {warning}")

        if validation_results["suggestions"]:
            feedback_parts.append("Additional recommendations:")
            for suggestion in validation_results["suggestions"]:
                feedback_parts.append(f"- {suggestion}")

        return "\n".join(feedback_parts)

    def _request_improved_code(self, code: str, feedback: str, task_description: str) -> str:
        """
        Request improved code from the LLM based on feedback.

        Args:
            code: Current code
            feedback: Validation feedback
            task_description: Original task description

        Returns:
            Improved code
        """
        prompt = (
            f"You need to improve the following {self.streaming_system} code based on validation feedback.\n\n"
            f"TASK DESCRIPTION:\n{task_description}\n\n"
            f"CURRENT CODE:\n```\n{code}\n```\n\n"
            f"VALIDATION FEEDBACK:\n{feedback}\n\n"
            f"Please provide an improved version of the code that addresses the feedback."
            f"Return only the improved code without explanations."
        )

        try:
            response = self.llm_model.invoke([HumanMessage(content=prompt)])
            improved_code = self._extract_code_from_response(response.content)
            return improved_code
        except Exception as e:
            logger.error(f"Error requesting improved code: {str(e)}")
            return code  # Return original code if request fails

    def _extract_code_from_response(self, response: str) -> str:
        """Extract code blocks from LLM response."""
        # Try to extract code between triple backticks
        code_blocks = re.findall(r"```(?:\w+)?\n(.+?)```", response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()

        # If no code blocks, return the whole response
        return response.strip()

    def _calculate_improvement_metrics(self, improvement_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metrics on how the code improved across iterations.

        Args:
            improvement_record: Record of improvement iterations

        Returns:
            Dict containing improvement metrics
        """
        iterations = improvement_record["iterations"]

        # Base metrics
        metrics = {
            "iteration_count": len(iterations),
            "issues_by_iteration": [],
            "warnings_by_iteration": [],
            "code_length_by_iteration": [],
            "improved": False
        }

        # Collect metrics by iteration
        for iteration in iterations:
            validation = iteration["validation_results"]
            metrics["issues_by_iteration"].append(len(validation["issues"]))
            metrics["warnings_by_iteration"].append(len(validation["warnings"]))
            metrics["code_length_by_iteration"].append(len(iteration["code"]))

        # Determine if code improved
        if len(iterations) > 1:
            first_validation = iterations[0]["validation_results"]
            last_validation = iterations[-1]["validation_results"]

            first_issue_count = len(first_validation["issues"])
            last_issue_count = len(last_validation["issues"])

            first_warning_count = len(first_validation["warnings"])
            last_warning_count = len(last_validation["warnings"])

            metrics["improved"] = (
                    (last_issue_count < first_issue_count) or
                    (last_issue_count == 0 and last_warning_count < first_warning_count)
            )

            metrics["improvement_rate"] = 1.0 if first_issue_count == 0 else (
                    (first_issue_count - last_issue_count) / first_issue_count
            )

        return metrics


# ---------------------------
# Prompt Engineering System
# ---------------------------
class PromptEngineeringSystem:
    """
    System that automatically enhances prompts with specific correctness requirements.
    """

    def __init__(self, streaming_system: str):
        """
        Initialize the prompt engineering system.

        Args:
            streaming_system: The streaming system being used
        """
        self.streaming_system = streaming_system
        self.template_path = os.path.join(
            PROMPT_TEMPLATES_DIR,
            f"{streaming_system.lower().replace(' ', '_')}_template.json"
        )
        self.templates = self._load_templates()
        self.pitfall_history = []

    def _load_templates(self) -> Dict[str, Any]:
        """Load prompt templates if they exist."""
        if os.path.exists(self.template_path):
            try:
                with open(self.template_path, 'r') as f:
                    return json.load(f)
            except:
                return self._get_default_templates()
        return self._get_default_templates()

    def _get_default_templates(self) -> Dict[str, Any]:
        """Return default prompt templates."""
        return {
            "base_prompt": (
                f"Generate production-grade {self.streaming_system} code that follows best practices, "
                f"including proper error handling, checkpointing, and efficient state management."
            ),
            "correctness_requirements": [
                "Implement proper error handling for all operations",
                "Configure checkpointing for reliable processing",
                "Use appropriate windowing strategies for time-based operations",
                "Ensure efficient state management"
            ],
            "common_pitfalls": [
                "Using unbounded state without cleanup mechanisms",
                "Missing error handling for source/sink failures",
                "Implementing blocking operations that affect performance",
                "Incorrect watermark configuration leading to late event issues"
            ],
            "examples": {}
        }

    def _save_templates(self):
        """Save prompt templates."""
        os.makedirs(os.path.dirname(self.template_path), exist_ok=True)
        with open(self.template_path, 'w') as f:
            json.dump(self.templates, f, indent=2)

    def enhance_prompt(self, base_prompt: str, task_type: str = "general",
                       include_examples: bool = True, include_pitfalls: bool = True) -> str:
        """
        Enhance a base prompt with correctness requirements and examples.

        Args:
            base_prompt: The original prompt
            task_type: Type of task (general, wordcount, join, etc.)
            include_examples: Whether to include examples
            include_pitfalls: Whether to include common pitfalls

        Returns:
            Enhanced prompt
        """
        enhanced_parts = [base_prompt]

        # Add correctness requirements
        enhanced_parts.append("\nIMPORTANT CORRECTNESS REQUIREMENTS:")
        for i, req in enumerate(self.templates["correctness_requirements"], 1):
            enhanced_parts.append(f"{i}. {req}")

        # Add task-specific requirements if available
        task_specific = self.templates.get("task_specific", {}).get(task_type, [])
        if task_specific:
            enhanced_parts.append(f"\nSPECIFIC REQUIREMENTS FOR {task_type.upper()} TASK:")
            for i, req in enumerate(task_specific, 1):
                enhanced_parts.append(f"{i}. {req}")

        # Add pitfalls to avoid
        if include_pitfalls:
            enhanced_parts.append("\nCOMMON PITFALLS TO AVOID:")
            for i, pitfall in enumerate(self.templates["common_pitfalls"], 1):
                enhanced_parts.append(f"{i}. {pitfall}")

        # Add examples if available
        if include_examples and task_type in self.templates.get("examples", {}):
            enhanced_parts.append(f"\nHERE IS AN EXAMPLE OF CORRECT {task_type.upper()} IMPLEMENTATION:")
            enhanced_parts.append(f"```\n{self.templates['examples'][task_type]}\n```")

        return "\n".join(enhanced_parts)

    def generate_detailed_spec(self, user_query: str, llm_model) -> Dict[str, Any]:
        """
        Generate detailed specifications for expected behavior.

        Args:
            user_query: The user's original query
            llm_model: LLM model to use for spec generation

        Returns:
            Dict containing detailed specifications
        """
        spec_prompt = (
            f"Based on this user query: '{user_query}', generate detailed specifications "
            f"for a {self.streaming_system} stream processing pipeline. "
            f"Include input data characteristics, processing requirements, expected output, "
            f"performance constraints, and error handling expectations. "
            f"Format the response as JSON with the following keys: "
            f"'data_sources', 'processing_steps', 'output_requirements', 'performance_constraints', "
            f"'error_handling', and 'additional_notes'."
        )

        try:
            response = llm_model.invoke([HumanMessage(content=spec_prompt)])
            spec_text = response.content

            # Try to extract JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', spec_text, re.DOTALL)
            if json_match:
                spec_json = json.loads(json_match.group(1))
            else:
                # Try to directly parse the response
                try:
                    spec_json = json.loads(spec_text)
                except:
                    # Create structured JSON from text if parsing fails
                    spec_json = self._structure_spec_from_text(spec_text)

            return spec_json
        except Exception as e:
            logger.error(f"Error generating specifications: {str(e)}")
            return {
                "data_sources": "Could not determine",
                "processing_steps": ["Parse user requirements"],
                "output_requirements": "Based on user query",
                "performance_constraints": "Standard performance expectations",
                "error_handling": "Basic error handling",
                "additional_notes": "Specification generation failed, using defaults"
            }

    def _structure_spec_from_text(self, text: str) -> Dict[str, Any]:
        """Create structured specification from unstructured text."""
        spec = {
            "data_sources": "",
            "processing_steps": [],
            "output_requirements": "",
            "performance_constraints": "",
            "error_handling": "",
            "additional_notes": ""
        }

        # Extract sections using common patterns
        sections = re.split(r'#+\s+|(?:Input|Output|Processing|Performance|Error|Note)s?:\s*', text)
        sections = [s.strip() for s in sections if s.strip()]

        # Try to match sections to spec keys
        for i, section in enumerate(sections):
            if "source" in section.lower() or "input" in section.lower():
                spec["data_sources"] = section
            elif "process" in section.lower() or "step" in section.lower():
                spec["processing_steps"] = [step.strip() for step in section.split('\n') if step.strip()]
            elif "output" in section.lower() or "result" in section.lower():
                spec["output_requirements"] = section
            elif "performance" in section.lower() or "constraint" in section.lower():
                spec["performance_constraints"] = section
            elif "error" in section.lower() or "exception" in section.lower():
                spec["error_handling"] = section
            elif i == len(sections) - 1:  # Last section as notes
                spec["additional_notes"] = section

        return spec

    def update_from_validation(self, validation_results: Dict[str, Any], code: str, task_type: str = "general"):
        """
        Update templates based on validation results to improve future prompts.

        Args:
            validation_results: Results from code validation
            code: The corresponding code
            task_type: Type of task the code implements
        """
        # Extract new pitfalls from issues
        for issue in validation_results.get("issues", []):
            if issue not in self.templates["common_pitfalls"]:
                self.templates["common_pitfalls"].append(issue)
                self.pitfall_history.append({
                    "pitfall": issue,
                    "discovered": datetime.now().isoformat(),
                    "frequency": 1
                })
            else:
                # Update frequency of known pitfall
                for pitfall in self.pitfall_history:
                    if pitfall["pitfall"] == issue:
                        pitfall["frequency"] += 1
                        break

        # Update examples if code passed validation
        if validation_results.get("passed", False) and not validation_results.get("warnings", []):
            if task_type not in self.templates.get("examples", {}):
                self.templates.setdefault("examples", {})[task_type] = code

        # Save updated templates
        self._save_templates()


# ---------------------------
# Integration into Main System
# ---------------------------
def integrate_into_deepgot(
        user_query: str,
        streaming_system: str,
        llm_model,
        task_type: str = "general",
        validate_code: bool = True,
        enhance_prompt: bool = True
) -> Dict[str, Any]:
    """
    Integrated function for generating and validating stream processing code.

    Args:
        user_query: User's original query
        streaming_system: The streaming system to use
        llm_model: LLM model for code generation
        task_type: Type of streaming task
        validate_code: Whether to validate and improve the code
        enhance_prompt: Whether to use prompt engineering

    Returns:
        Dict containing generated code, validation results, and metadata
    """
    # Initialize systems
    prompt_engineer = PromptEngineeringSystem(streaming_system)
    feedback_system = FeedbackLoopSystem(llm_model, streaming_system)

    # Generate detailed specifications
    specifications = prompt_engineer.generate_detailed_spec(user_query, llm_model)
    logger.info(f"Generated specifications: {json.dumps(specifications, indent=2)}")

    # Create base prompt
    base_prompt = (
        f"Generate {streaming_system} code for the following user requirements:\n"
        f"{user_query}\n\n"
        f"The code should implement a streaming pipeline addressing these specifications:\n"
        f"- Data Sources: {specifications['data_sources']}\n"
        f"- Processing Steps: {', '.join(specifications['processing_steps'])}\n"
        f"- Output Requirements: {specifications['output_requirements']}\n"
        f"- Performance Constraints: {specifications['performance_constraints']}"
    )

    # Enhance prompt if requested
    final_prompt = (
        prompt_engineer.enhance_prompt(base_prompt, task_type)
        if enhance_prompt else base_prompt
    )

    # Generate initial code
    response = llm_model.invoke([HumanMessage(content=final_prompt)])
    initial_code = extract_code_from_response(response.content)

    results = {
        "specifications": specifications,
        "enhanced_prompt": final_prompt,
        "initial_code": initial_code,
        "validation_results": None,
        "improved_code": None,
        "improvement_metrics": None,
        "timestamp": datetime.now().isoformat()
    }

    # Validate and improve code if requested
    if validate_code:
        improvement_results = feedback_system.validate_and_improve(
            initial_code,
            f"{user_query}\n\n{json.dumps(specifications, indent=2)}"
        )

        results["validation_results"] = improvement_results["history"][-1]["validation_results"]
        results["improved_code"] = improvement_results["improved_code"]
        results["improvement_metrics"] = improvement_results["metrics"]

        # Update prompt templates based on validation results
        prompt_engineer.update_from_validation(
            results["validation_results"],
            results["improved_code"],
            task_type
        )

    return results


def extract_code_from_response(response: str) -> str:
    """Extract code blocks from LLM response."""
    code_blocks = re.findall(r"```(?:\w+)?\n(.+?)```", response, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    return response.strip()


# ---------------------------
# Command Line Interface
# ---------------------------
def validate_pipeline_cli():
    """Command line interface for pipeline validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Stream Processing Pipeline Validator")
    parser.add_argument("--file", type=str, help="Path to the code file to validate")
    parser.add_argument("--system", type=str, default="Apache Flink",
                        help="Stream processing system (Apache Flink, Apache Spark, Kafka Streams)")
    parser.add_argument("--improve", action="store_true", help="Attempt to improve the code")
    parser.add_argument("--iterations", type=int, default=3, help="Maximum improvement iterations")

    args = parser.parse_args()

    if not args.file:
        print("Please specify a file to validate with --file")
        return

    try:
        with open(args.file, 'r') as f:
            code = f.read()

        validator = get_validator(args.system)
        results = validator.validate_code(code)

        print(f"\n=== Validation Results for {args.system} ===")
        print(f"Passed: {'Yes' if results['passed'] else 'No'}")

        if results['issues']:
            print("\nIssues:")
            for issue in results['issues']:
                print(f"- {issue}")

        if results['warnings']:
            print("\nWarnings:")
            for warning in results['warnings']:
                print(f"- {warning}")

        if args.improve and (results['issues'] or results['warnings']):
            print("\n=== Attempting to improve code ===")
            # Note: This would require an LLM model instance
            print("Code improvement requires an LLM model instance.")
            print("Please use the integrated API instead.")

    except FileNotFoundError:
        print(f"File not found: {args.file}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    validate_pipeline_cli()