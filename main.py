import argparse
import sys
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq

from Config.env_config import configure_llms_environment, configure_langchain_environment
from code_evaluation import comprehensive_code_evaluation, print_evaluation_report

# Initialize environment for all LLMs
configure_llms_environment()

def get_llm_model(model_choice: str):
    """
    Returns an LLM model instance based on the given model choice string.
    If the choice is invalid or not recognized, returns a default model.
    """
    # Dictionary mapping model choice to the actual LLM object
    model_mapping = {
        "mistral": ChatMistralAI(model="mistral-large-latest"),
        "openai": ChatOpenAI(model="gpt-4o"),
        "anthropic": ChatAnthropic(model="claude-3-sonnet"),
        # "groq": ChatGroq(model="llama3-8b-8192"),
        "cohere": ChatCohere(model="command-r-plus"),
    }

    if model_choice.lower() in model_mapping:
        return model_mapping[model_choice.lower()]
    else:
        print(f"WARNING: Invalid model choice '{model_choice}'. Defaulting to Mistral.")
        return ChatMistralAI(model="mistral-large-latest")

def prompt_for_streaming_system() -> str:
    """
    Prompts the user to select a streaming system from a menu.
    Returns the chosen system as a string.
    """
    systems = {
        "1": "Apache Flink",
        "2": "Apache Storm",
        "3": "Apache Spark",
        "4": "Kafka Stream",
    }

    for key, value in systems.items():
        print(f"{key}. {value}")

    choice = input("Enter your choice (1/2/3/4): ").strip()
    if choice in systems:
        return systems[choice]
    else:
        print("Invalid choice. Defaulting to 'Apache Flink'.")
        return "Apache Flink"

def build_messages(streaming_system: str, user_message: str):
    """
    Builds the list of system and human messages to be passed to the model.
    """
    return [
        SystemMessage(
            content=(
                f"You are a highly skilled stream-processing expert."
                f"Your task is to generate a complete application pipeline for {streaming_system} (latest version) "
                f"Provide well-commented, production-grade code samples, along with any setup instructions."
                f"Use the user's request below as a key reference."
            )
        ),
        HumanMessage(content=user_message),
    ]

def invoke_model(model, messages):
    """
    Invokes the chosen LLM model and returns its response content.
    Includes optional exception handling.
    """
    try:
        response = model.invoke(messages)
        return response.content
    except Exception as e:
        print(f"ERROR: Failed to invoke the model. Details: {e}")
        return "I'm sorry, something went wrong while generating the response."

def evaluate_response(response_content: str, streaming_system: str, user_message: str) -> dict:
    """
    Evaluates the quality of generated code response.

    Args:
        response_content: Generated response from LLM
        streaming_system: Name of streaming system
        user_message: Original user query

    Returns:
        Dict containing evaluation metrics
    """
    metrics = {
        'code_presence': False,
        'code_blocks': 0,
        'system_specific': False,
        'compilation_ready': False,
        'has_comments': False,
        'has_setup': False,
        'score': 0
    }

    # Check for code blocks
    code_blocks = response_content.count('```')
    metrics['code_blocks'] = code_blocks // 2
    metrics['code_presence'] = code_blocks > 0

    # Check for streaming system references
    metrics['system_specific'] = streaming_system.lower() in response_content.lower()

    # Check for common code patterns
    if metrics['code_presence']:
        metrics['has_comments'] = '//' in response_content or '#' in response_content
        metrics['has_setup'] = 'import' in response_content or 'dependencies' in response_content.lower()
        metrics['compilation_ready'] = all([
            'class' in response_content or 'def' in response_content,
            'main' in response_content.lower(),
            metrics['has_setup']
        ])

    # Calculate overall score
    metrics['score'] = sum([
        20 if metrics['code_presence'] else 0,
        20 if metrics['system_specific'] else 0,
        20 if metrics['compilation_ready'] else 0,
        20 if metrics['has_comments'] else 0,
        20 if metrics['has_setup'] else 0
    ]) / 100.0

    return metrics

def print_evaluation(metrics: dict):
    """Prints evaluation results in a formatted way."""
    print("\nResponse Evaluation:")
    print(f"✓ Code Present: {'Yes' if metrics['code_presence'] else 'No'}")
    print(f"✓ Code Blocks: {metrics['code_blocks']}")
    print(f"✓ System Specific: {'Yes' if metrics['system_specific'] else 'No'}")
    print(f"✓ Compilation Ready: {'Yes' if metrics['compilation_ready'] else 'No'}")
    print(f"✓ Has Comments: {'Yes' if metrics['has_comments'] else 'No'}")
    print(f"✓ Has Setup: {'Yes' if metrics['has_setup'] else 'No'}")
    print(f"Overall Score: {metrics['score']*100:.1f}%")


def main(args):
    """
    Main function that orchestrates the user prompt for the streaming system,
    collects the message, invokes the model, and prints the result.
    """
    # Pick the LLM based on command-line argument or default to 'mistral'
    model = get_llm_model(args.model)

    # Get user input for streaming system and the message
    streaming_system = prompt_for_streaming_system()
    user_message = input("Please enter your message: ")
    print("Please be patient, we are preparing your code...")

    # Build messages to send
    messages = build_messages(streaming_system, user_message)

    # Invoke the model and print the response
    response_content = invoke_model(model, messages)
    print(response_content)

    metrics = evaluate_response(response_content, streaming_system, user_message)
    print_evaluation(metrics)

    # Comprehensive evaluation
    eval_result = comprehensive_code_evaluation(response_content, streaming_system)
    print_evaluation_report(eval_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate code snippets for a chosen streaming system using an LLM model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral",
        help="Model choice: mistral, openai, anthropic, groq, or cohere. Defaults to 'mistral'."
    )
    args = parser.parse_args()
    configure_langchain_environment()
    configure_llms_environment()
    # If you need advanced logging, replace prints with Python's logging library setup:
    # import logging
    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger(__name__)

    main(args)
