import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

class StepResultsSaver:
    """
    Class for saving the results of each step in the query analyzer process
    and specifically handling Java code extraction and saving.
    """
    def __init__(self, base_output_dir: str = "query_analyzer_results"):
        """
        Initialize a StepResultsSaver object.

        :param base_output_dir: Base directory for saving results
        """
        self.base_output_dir = base_output_dir
        # Create a session-specific directory using timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_output_dir, f"session_{self.timestamp}")

        # Create directories
        self.steps_dir = os.path.join(self.session_dir, "steps")
        self.code_dir = os.path.join(self.session_dir, "generated_code")
        self.java_dir = os.path.join(self.code_dir, "java")

        # Ensure directories exist
        os.makedirs(self.steps_dir, exist_ok=True)
        os.makedirs(self.code_dir, exist_ok=True)
        os.makedirs(self.java_dir, exist_ok=True)

        # Logger setup
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Results will be saved to: {self.session_dir}")

        # Track the steps saved
        self.steps_saved = []

    def save_step_result(self, step_id: int, step_name: str, result: Any) -> str:
        """
        Save the result of a step to a file.

        :param step_id: ID of the step
        :param step_name: Name of the step (action)
        :param result: The result data to save
        :return: Path to the saved file
        """
        # Create a filename with step ID and name
        safe_name = self._sanitize_filename(step_name)
        filename = f"step_{step_id:02d}_{safe_name}.json"
        filepath = os.path.join(self.steps_dir, filename)

        # Prepare data for saving
        data_to_save = {
            "step_id": step_id,
            "step_name": step_name,
            "timestamp": datetime.now().isoformat(),
            "result": result
        }

        # Save to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, default=str)

            self.logger.info(f"Saved step result to: {filepath}")
            self.steps_saved.append({"step_id": step_id, "path": filepath})
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving step result: {str(e)}")
            return ""

    def extract_and_save_java_code(self, text: str, filename_prefix: str = "pipeline") -> Optional[str]:
        """
        Extract Java code from text and save it to a separate file.

        :param text: Text that might contain Java code
        :param filename_prefix: Prefix for the filename
        :return: Path to the saved Java file, or None if no code found
        """
        # Extract Java code blocks from the text
        java_code = self._extract_java_code(text)

        if not java_code:
            self.logger.info("No Java code found in the provided text")
            return None

        # Create a filename with timestamp to avoid overwriting
        safe_prefix = self._sanitize_filename(filename_prefix)
        filename = f"{safe_prefix}_{datetime.now().strftime('%H%M%S')}.java"
        filepath = os.path.join(self.java_dir, filename)

        # Save the Java code to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(java_code)

            self.logger.info(f"Saved Java code to: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving Java code: {str(e)}")
            return None

    def save_final_response(self, response: str, query: str) -> str:
        """
        Save the final response to a file.

        :param response: The final response text
        :param query: The original query
        :return: Path to the saved file
        """
        # Create a filename for the final response
        filename = f"final_response_{self.timestamp}.txt"
        filepath = os.path.join(self.session_dir, filename)

        # Prepare content
        content = f"QUERY:\n{query}\n\nRESPONSE:\n{response}"

        # Save to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            self.logger.info(f"Saved final response to: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving final response: {str(e)}")
            return ""

    def save_execution_plan(self, plan_visualization: str) -> str:
        """
        Save the execution plan visualization to a file.

        :param plan_visualization: The plan visualization text
        :return: Path to the saved file
        """
        # Create a filename for the plan
        filename = "execution_plan.txt"
        filepath = os.path.join(self.session_dir, filename)

        # Save to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(plan_visualization)

            self.logger.info(f"Saved execution plan to: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving execution plan: {str(e)}")
            return ""

    def save_session_summary(self, query: str, intent: str, streaming_system: str) -> str:
        """
        Save a summary of the session including links to all files.

        :param query: The original query
        :param intent: The detected intent
        :param streaming_system: The streaming system used
        :return: Path to the saved file
        """
        # Create a filename for the summary
        filename = "session_summary.txt"
        filepath = os.path.join(self.session_dir, filename)

        # Prepare content
        lines = [
            f"SESSION SUMMARY: {self.timestamp}",
            f"=================================",
            f"",
            f"QUERY: {query}",
            f"INTENT: {intent}",
            f"STREAMING SYSTEM: {streaming_system}",
            f"",
            f"FILES GENERATED:",
            f"-----------------"
        ]

        # Add steps saved
        if self.steps_saved:
            lines.append("\nSTEP RESULTS:")
            for step in self.steps_saved:
                lines.append(f"  - Step {step['step_id']}: {step['path']}")

        # Add Java files
        java_files = [f for f in os.listdir(self.java_dir) if f.endswith('.java')]
        if java_files:
            lines.append("\nJAVA CODE FILES:")
            for java_file in java_files:
                lines.append(f"  - {os.path.join(self.java_dir, java_file)}")

        # Save to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

            self.logger.info(f"Saved session summary to: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving session summary: {str(e)}")
            return ""

    def _sanitize_filename(self, name: str) -> str:
        """
        Sanitize a string to be used as a filename.

        :param name: The string to sanitize
        :return: Sanitized string
        """
        # Replace spaces and special characters
        import re
        sanitized = re.sub(r'[^\w\s-]', '', name)
        sanitized = re.sub(r'[\s-]+', '_', sanitized).strip('_')
        return sanitized.lower()

    def _extract_java_code(self, text: str) -> str:
        """
        Extract Java code from text using regex.

        :param text: The text that may contain Java code
        :return: Extracted Java code
        """
        import re

        # Try to find code blocks marked as Java
        java_blocks = re.findall(r'```java\s*(.*?)\s*```', text, re.DOTALL)

        if java_blocks:
            # Join multiple blocks with newlines and comments
            return '\n\n// Next code block\n\n'.join(java_blocks)

        # If no explicit Java blocks, try to detect Java code by looking for common patterns
        if 'public class' in text or 'import java.' in text:
            # Find the start of what appears to be Java code
            possible_starts = [
                'import java.',
                'package ',
                'public class',
                'class ',
                'interface ',
                'enum '
            ]

            start_positions = []
            for pattern in possible_starts:
                pos = text.find(pattern)
                if pos != -1:
                    start_positions.append(pos)

            if start_positions:
                # Start from the earliest Java code indicator
                start_pos = min(start_positions)

                # Try to find the end of the code block
                end_markers = [
                    '\n```',
                    '\n---',
                    '\n###'
                ]

                end_positions = []
                for marker in end_markers:
                    pos = text.find(marker, start_pos)
                    if pos != -1:
                        end_positions.append(pos)

                if end_positions:
                    end_pos = min(end_positions)
                else:
                    # If no clear end, take the rest of the text
                    end_pos = len(text)

                return text[start_pos:end_pos].strip()

        return ""

# Function to integrate with the rest of the code
def create_results_saver() -> StepResultsSaver:
    """
    Create and configure a StepResultsSaver instance.

    :return: Configured StepResultsSaver
    """
    return StepResultsSaver()