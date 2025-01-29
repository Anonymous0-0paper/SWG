import argparse
import logging
import json
import os
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq

from Config.env_config import configure_llms_environment, configure_langchain_environment
from code_evaluation import comprehensive_code_evaluation, print_evaluation_report

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize environment for all LLMs
configure_langchain_environment()
configure_llms_environment()

# Memory file directory
MEMORY_DIR = "memory_files"
os.makedirs(MEMORY_DIR, exist_ok=True)

# Load a memory file
def load_memory(file_name: str):
    memory_path = os.path.join(MEMORY_DIR, file_name)
    if os.path.exists(memory_path):
        with open(memory_path, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                logger.warning(f"Memory file '{file_name}' is corrupted. Starting with an empty memory.")
                return []
    return []

# Save memory to a timestamped file
def save_memory(memory):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    memory_path = os.path.join(MEMORY_DIR, f"memory_{timestamp}.json")
    with open(memory_path, 'w') as file:
        json.dump(memory, file, indent=2)
    logger.info(f"Memory saved to {memory_path}")

# List available memory files
def list_memory_files():
    files = [f for f in os.listdir(MEMORY_DIR) if f.endswith(".json")]
    if not files:
        print("No memory files found.")
        return []
    print("\nAvailable Memory Files:")
    for i, file_name in enumerate(files, 1):
        print(f"  {i}. {file_name}")
    return files

# Prompt user to choose a memory file
def choose_memory_file():
    files = list_memory_files()
    if not files:
        print("Starting with an empty memory.")
        return []

    choice = input("\nEnter the number of the memory file to load (or press Enter to skip): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(files):
        selected_file = files[int(choice) - 1]
        print(f"Using memory file: {selected_file}")
        return load_memory(selected_file)
    else:
        print("No memory file selected. Starting with an empty memory.")
        return []

# In-memory conversation history
conversation_memory = []

def get_llm_model(model_choice: str, temperature: float, top_p: float, max_tokens: int):
    """
    Returns an LLM model instance based on the given model choice string.
    Configures the model with temperature, top-p, and max_tokens.
    """
    model_mapping = {
        "mistral": ChatMistralAI(model="mistral-large-latest", temperature=temperature, top_p=top_p, max_tokens=max_tokens),
        "openai": ChatOpenAI(model="gpt-4o", temperature=temperature, top_p=top_p, max_tokens=max_tokens),
        "anthropic": ChatAnthropic(model="claude-3-sonnet", temperature=temperature, top_p=top_p, max_tokens=max_tokens),
        "cohere": ChatCohere(model="command-r-plus", temperature=temperature, top_p=top_p, max_tokens=max_tokens),
    }

    if model_choice.lower() in model_mapping:
        return model_mapping[model_choice.lower()]
    else:
        logger.warning(f"Invalid model choice '{model_choice}'. Defaulting to Mistral.")
        return ChatMistralAI(model="mistral-large-latest", temperature=temperature, top_p=top_p, max_tokens=max_tokens)

# def prompt_for_streaming_system() -> str:
#     """
#     Prompts the user to select a streaming system from a menu or via auto-completion.
#     Returns the chosen system as a string.
#     """
#     systems_completer = WordCompleter(
#         [
#             "Apache Flink", "Apache Storm", "Apache Spark", "Kafka Stream",
#             "Apache Samza", "Apache Heron", "Materialize", "Apache Pulsar",
#             "Redpanda", "Google Dataflow", "Amazon Kinesis"
#         ],
#         ignore_case=True,
#     )
#
#     return prompt("Choose a streaming system: ", completer=systems_completer)

# List available streaming systems
def list_streaming_systems():
    systems = [
        "Apache Flink", "Apache Storm", "Apache Spark", "Kafka Stream",
        "Apache Samza", "Apache Heron", "Materialize", "Apache Pulsar",
        "Redpanda", "Google Dataflow", "Amazon Kinesis"
    ]
    print("\nAvailable Streaming Systems:")
    for i, system in enumerate(systems, 1):
        print(f"  {i}. {system}")
    return systems

# Prompt user to choose a streaming system
def prompt_for_streaming_system():
    systems = list_streaming_systems()
    choice = input("\nEnter the number of the streaming system to use: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(systems):
        return systems[int(choice) - 1]
    else:
        print("Invalid choice. Defaulting to 'Apache Flink'.")
        return "Apache Flink"

def build_messages(streaming_system: str, user_message: str, use_memory: bool, custom_template: str = None):
    """
    Builds the list of system and human messages to be passed to the model.
    Optionally includes conversation history in the messages.
    """
    system_content = custom_template or (
        f"You are a highly skilled stream-processing expert. "
        f"Your task is to generate a complete application pipeline for {streaming_system} (latest version). "
        f"Provide well-commented, production-grade code samples, along with any setup instructions. "
        f"Use the user's request below as a key reference."
    )

    messages = [SystemMessage(content=system_content)]

    # Include memory if the user chooses to use it
    if use_memory:
        for past_message in conversation_memory:
            messages.append(HumanMessage(content=past_message["user_message"]))
            messages.append(SystemMessage(content=past_message["response_content"]))

    messages.append(HumanMessage(content=user_message))
    return messages

def invoke_model(models, messages):
    """
    Invokes the chosen LLM models in sequence and returns the final response content.
    """
    response_content = ""
    for model in models:
        try:
            response = model.invoke(messages)
            response_content += response.content + "\n"
        except Exception as e:
            logger.error(f"Failed to invoke model {model}. Details: {e}")
            response_content += "An error occurred while generating the response.\n"
    return response_content

def interactive_mode(args):
    """
    Runs the script in interactive mode, allowing multiple interactions in a single session.
    """
    global conversation_memory
    conversation_memory = choose_memory_file()

    models = [
        get_llm_model(model, args.temperature, args.top_p, args.max_tokens)
        for model in args.models.split(',')
    ]

    while True:
        streaming_system = prompt_for_streaming_system()
        user_message = input("\nPlease enter your message: ")
        use_memory = input("Do you want to use the loaded memory in this interaction? (y/n): ").strip().lower() == "y"

        if args.prompt_file:
            with open(args.prompt_file, 'r') as file:
                custom_template = file.read()
        else:
            custom_template = input("Enter custom system message template (optional): ").strip() or None

        messages = build_messages(streaming_system, user_message, use_memory, custom_template)
        response_content = invoke_model(models, messages)

        print("\nGenerated Response:")
        print(response_content)

        # Save interaction to memory
        conversation_memory.append({
            "streaming_system": streaming_system,
            "user_message": user_message,
            "response_content": response_content
        })

        save_memory(conversation_memory)

        # Continue or exit
        if input("\nDo you want to continue? (y/n): ").strip().lower() != "y":
            break

def main(args):
    """
    Main function that handles single interaction or starts interactive mode.
    """
    if args.interactive:
        interactive_mode(args)
    else:
        models = [
            get_llm_model(model, args.temperature, args.top_p, args.max_tokens)
            for model in args.models.split(',')
        ]
        streaming_system = prompt_for_streaming_system()
        user_message = input("Please enter your message: ")
        use_memory = input("Do you want to use the loaded memory in this interaction? (y/n): ").strip().lower() == "y"

        if args.prompt_file:
            with open(args.prompt_file, 'r') as file:
                custom_template = file.read()
        else:
            custom_template = input("Enter custom system message template (optional): ").strip() or None

        messages = build_messages(streaming_system, user_message, use_memory, custom_template)
        response_content = invoke_model(models, messages)

        print(response_content)

        # Save interaction to memory
        conversation_memory.append({
            "streaming_system": streaming_system,
            "user_message": user_message,
            "response_content": response_content
        })

        save_memory(conversation_memory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate code snippets for a chosen streaming system using LLM models."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="mistral",
        help="Comma-separated model choices: mistral, openai, anthropic, groq, cohere. Defaults to 'mistral'."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Set the temperature for the model response. Defaults to 0.7."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Set the top-p value for nucleus sampling. Defaults to 0.9."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=15000,
        help="Set the maximum number of tokens in the model's response. Defaults to 500."
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        help="Path to a file containing the custom system message template."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode for continuous usage."
    )
    args = parser.parse_args()

    if True:
        while True:
            interactive_mode(args)
            if input("Do you want to continue? (y/n): ").strip().lower() != 'y':
                break
    else:
        main(args)
