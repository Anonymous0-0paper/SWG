import argparse
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq

from Config.env_config import configure_llms_environment, configure_langchain_environment

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize environment for all LLMs
configure_langchain_environment()
configure_llms_environment()

# Memory file directory
MEMORY_DIR = "memory_files"
os.makedirs(MEMORY_DIR, exist_ok=True)

class ThoughtNode:
    """Represents a node in the Graph of Thoughts"""
    def __init__(self, node_id: int, content: str,
                 parent_ids: Optional[List[int]] = None,
                 node_type: str = "model"):
        self.id = node_id
        self.content = content
        self.parent_ids = parent_ids if parent_ids else []
        self.type = node_type  # Types: 'system', 'user', 'model', 'error'

    def __repr__(self):
        return f"<ThoughtNode {self.id} {self.type}>"

class GraphOfThoughts:
    """Manages the graph structure and node relationships"""
    def __init__(self):
        self.nodes: Dict[int, ThoughtNode] = {}
        self.current_id = 0
        self.final_output: Optional[str] = None

    def add_node(self, content: str,
                 parent_ids: Optional[List[int]] = None,
                 node_type: str = "model") -> int:
        """Add a new node to the graph"""
        node_id = self.current_id
        self.nodes[node_id] = ThoughtNode(node_id, content, parent_ids, node_type)
        self.current_id += 1
        return node_id

    def get_node(self, node_id: int) -> Optional[ThoughtNode]:
        return self.nodes.get(node_id)

    def add_relationship(self, from_id: int, to_id: int):
        """Create a relationship between two nodes"""
        if to_id in self.nodes and from_id in self.nodes:
            self.nodes[to_id].parent_ids.append(from_id)

    def get_relevant_nodes(self, node_types: Optional[List[str]] = None) -> List[ThoughtNode]:
        """Retrieve nodes filtered by type"""
        if not node_types:
            return list(self.nodes.values())
        return [n for n in self.nodes.values() if n.type in node_types]

    def set_final_output(self, content: str):
        """Set the final output node"""
        self.final_output = content

    def visualize(self) -> str:
        """Simple text visualization of the graph structure"""
        visualization = []
        for node in self.nodes.values():
            parents = ", ".join(map(str, node.parent_ids)) if node.parent_ids else "None"
            visualization.append(
                f"Node {node.id} ({node.type}):\n"
                f"  Content: {node.content[:]}...\n"
                f"  Parents: {parents}\n"
            )
        return "\n".join(visualization)

def load_memory(file_name: str) -> List[dict]:
    memory_path = os.path.join(MEMORY_DIR, file_name)
    if os.path.exists(memory_path):
        try:
            with open(memory_path, 'r') as file:
                return json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"Memory file '{file_name}' is corrupted or inaccessible. Starting fresh.")
    return []

def save_memory(memory: List[dict]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    memory_path = os.path.join(MEMORY_DIR, f"memory_{timestamp}.json")
    with open(memory_path, 'w') as file:
        json.dump(memory, file, indent=2)
    logger.info(f"Memory saved to {memory_path}")

def list_memory_files() -> List[str]:
    files = [f for f in os.listdir(MEMORY_DIR) if f.endswith(".json")]
    if not files:
        print("No memory files found.")
    return files

def choose_memory_file() -> List[dict]:
    files = list_memory_files()
    if not files:
        return []

    print("\nAvailable Memory Files:")
    for i, file_name in enumerate(files, 1):
        print(f"  {i}. {file_name}")

    try:
        choice = input("\nEnter the number to load (Enter to skip): ").strip()
        if choice and 1 <= int(choice) <= len(files):
            return load_memory(files[int(choice)-1])
    except ValueError:
        pass
    return []

def get_llm_model(model_choice: str, temperature: float,
                  top_p: float, max_tokens: int):
    models = {
        "mistral": ChatMistralAI(model="mistral-large-latest"),
        "openai": ChatOpenAI(model="gpt-4o"),
        "anthropic": ChatAnthropic(model="claude-3-sonnet"),
        "cohere": ChatCohere(model="command-r-plus"),
        # "groq": ChatGroq(model="mixtral-8x7b-32768")
    }
    model = models.get(model_choice.lower(), models["mistral"])
    return model.bind(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

def list_streaming_systems() -> List[str]:
    return [
        "Apache Flink", "Apache Storm", "Apache Spark", "Kafka Stream",
        "Apache Samza", "Apache Heron", "Materialize", "Apache Pulsar",
        "Redpanda", "Google Dataflow", "Amazon Kinesis"
    ]

def prompt_for_streaming_system() -> str:
    systems = list_streaming_systems()
    print("\nAvailable Streaming Systems:")
    for i, system in enumerate(systems, 1):
        print(f"  {i}. {system}")

    while True:
        try:
            choice = int(input("\nEnter system number: "))
            if 1 <= choice <= len(systems):
                return systems[choice-1]
        except ValueError:
            pass
        print("Invalid choice. Using Apache Flink.")
        return "Apache Flink"

def build_initial_graph(streaming_system: str, user_message: str,
                        use_memory: bool, custom_template: str = None) -> GraphOfThoughts:
    graph = GraphOfThoughts()

    # System message node
    system_content = custom_template or (
        f"Expert stream-processing assistant for {streaming_system}. "
        f"Generate complete, production-grade pipelines with setup instructions."
    )
    system_node = graph.add_node(system_content, node_type="system")

    # Historical memory nodes
    if use_memory:
        for memory_entry in conversation_memory:
            user_node = graph.add_node(memory_entry["user_message"], node_type="user")
            response_node = graph.add_node(
                memory_entry["response_content"],
                parent_ids=[user_node],
                node_type="model"
            )
            graph.add_relationship(system_node, response_node)

    # Current user message node
    user_node = graph.add_node(user_message, node_type="user")
    graph.add_relationship(system_node, user_node)

    return graph

def invoke_got(models: list, graph: GraphOfThoughts) -> str:
    try:
        for model in models:
            context_nodes = graph.get_relevant_nodes(["system", "user", "model"])
            messages = []
            for node in context_nodes:
                if node.type == "system":
                    messages.append(SystemMessage(content=node.content))
                else:
                    messages.append(HumanMessage(content=node.content))

            response = model.invoke(messages)
            new_node_id = graph.add_node(
                response.content,
                parent_ids=[n.id for n in context_nodes],
                node_type="model"
            )

            logger.debug(f"Added node {new_node_id} from {model}")

        final_nodes = graph.get_relevant_nodes(["model"])
        if final_nodes:
            graph.set_final_output(final_nodes[-1].content)
            return final_nodes[-1].content
        return "No valid response generated."

    except Exception as e:
        logger.error(f"GoT processing error: {str(e)}")
        graph.add_node(f"ERROR: {str(e)}", node_type="error")
        return f"Processing error: {str(e)}"

def interactive_mode(args):
    global conversation_memory
    conversation_memory = choose_memory_file()

    models = [
        get_llm_model(model, args.temperature, args.top_p, args.max_tokens)
        for model in args.models.split(',')
    ]

    while True:
        # Clear screen between interactions
        os.system('cls' if os.name == 'nt' else 'clear')

        print("╔════════════════════════════════════════════╗")
        print("║       STREAM PROCESSING ASSISTANT         ║")
        print("╚════════════════════════════════════════════╝")

        system = prompt_for_streaming_system()
        user_msg = input("\n📝 Your query: ")
        use_memory = input("💾 Use memory? (y/n): ").lower() == "y"

        custom_template = None
        if args.prompt_file:
            with open(args.prompt_file, 'r') as f:
                custom_template = f.read()
        else:
            custom_input = input("⚙️  Custom template (Enter to skip): ")
            if custom_input:
                custom_template = custom_input

        thought_graph = build_initial_graph(system, user_msg, use_memory, custom_template)
        response = invoke_got(models, thought_graph)

        # Print formatted response
        print("\n🔍 Processing complete!")
        print("╔════════════════════════════════════════════╗")
        print("║                RESPONSE                   ║")
        print("╠════════════════════════════════════════════╣")
        print(response)
        print("╚════════════════════════════════════════════╝")

        # Optional debug view
        if input("\n🔧 Show thought graph? (y/n): ").lower() == "y":
            print("\n🧠 Thought Process Visualization:")
            print(thought_graph.visualize())

        conversation_memory.append({
            "streaming_system": system,
            "user_message": user_msg,
            "response_content": response,
            "graph_visualization": thought_graph.visualize()
        })
        save_memory(conversation_memory)

        if input("\n🔄 Continue? (y/n): ").lower() != "y":
            print("\n👋 Session ended. Goodbye!")
            break

def main():
    parser = argparse.ArgumentParser(description="GoT-enhanced LLM Pipeline Generator")
    parser.add_argument("--models", default="mistral",
                        help="Comma-separated models: mistral,openai,anthropic,cohere,groq")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=25000)
    parser.add_argument("--prompt_file", help="Path to custom system prompt template")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode(args)
    else:
        print("Single-run mode not fully implemented. Use --interactive for GoT features.")

if __name__ == "__main__":
    main()