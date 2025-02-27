import argparse
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import the message types
from langchain_core.messages import HumanMessage, SystemMessage
# Import the LLM providers
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq

import utils
from Config.env_config import configure_llms_environment, configure_langchain_environment
from query_analyzer import PlanExecutor, QueryAnalyzer

from utils import log_phase, retrieve_relevant_documents


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize environment for all LLMs
configure_langchain_environment()
configure_llms_environment()

# Memory file directory
MEMORY_DIR = "memory_files"
os.makedirs(MEMORY_DIR, exist_ok=True)

# ---------------------------
# Core Data Structures
# ---------------------------
class ThoughtNode:
    """Represents a node in the reasoning graph (or hypergraph)."""
    def __init__(self, node_id: int, content: str,
                 parent_ids: Optional[List[int]] = None,
                 node_type: str = "model", score: float = 0.0):
        self.id = node_id
        self.content = content
        self.parent_ids = parent_ids if parent_ids else []
        self.type = node_type  # e.g., 'system', 'user', 'model', 'agent_x', 'error'
        self.score = score     # New attribute for scoring

    def __repr__(self):
        return f"<ThoughtNode {self.id} ({self.type}) - Score: {self.score}>"

class GraphOfThoughts:
    """Manages a graph structure and node relationships (pairwise edges)."""
    def __init__(self):
        self.nodes: Dict[int, ThoughtNode] = {}
        self.current_id = 0
        self.final_output: Optional[str] = None

    def add_node(self, content: str,
                 parent_ids: Optional[List[int]] = None,
                 node_type: str = "model", score: float = 0.0) -> int:
        """Add a new node to the graph."""
        node_id = self.current_id
        self.nodes[node_id] = ThoughtNode(node_id, content, parent_ids, node_type, score)
        self.current_id += 1
        return node_id

    def get_node(self, node_id: int) -> Optional[ThoughtNode]:
        return self.nodes.get(node_id)

    def add_relationship(self, from_id: int, to_id: int):
        """Create a relationship (edge) between two nodes."""
        if to_id in self.nodes and from_id in self.nodes:
            self.nodes[to_id].parent_ids.append(from_id)

    def get_relevant_nodes(self, node_types: Optional[List[str]] = None) -> List[ThoughtNode]:
        """Retrieve nodes filtered by type."""
        if not node_types:
            return list(self.nodes.values())
        return [n for n in self.nodes.values() if n.type in node_types]

    def set_final_output(self, content: str):
        """Set the final output node (for convenience)."""
        self.final_output = content

    def visualize(self) -> str:
        """Simple text visualization of the graph structure."""
        visualization = []
        for node in self.nodes.values():
            parents = ", ".join(map(str, node.parent_ids)) if node.parent_ids else "None"
            visualization.append(
                f"Node {node.id} ({node.type}):\n"
                f"  Content: {node.content[:100]}...\n"
                f"  Parents: {parents}\n"
            )
        return "\n".join(visualization)

    def prune_graph(self, min_score: float = 0.5):
        """Remove nodes with scores below a threshold."""
        self.nodes = {
            node_id: node
            for node_id, node in self.nodes.items()
            if node.score >= min_score
        }

# ---------------------------
# HYPERGRAPH OF THOUGHTS (HGoT)
# ---------------------------
class Hyperedge:
    """
    Represents a hyperedge that connects multiple nodes at once.
    Allows for higher-order relationships (more than two nodes).
    """
    def __init__(self, edge_id: int, connected_nodes: List[int], description: str = ""):
        """
        :param edge_id: Unique identifier for this hyperedge
        :param connected_nodes: List of node IDs connected by this hyperedge
        :param description: Optional text describing the hyperedge
        """
        self.id = edge_id
        self.connected_nodes = connected_nodes
        self.description = description

    def __repr__(self):
        return f"<Hyperedge {self.id} connects nodes {self.connected_nodes}>"

class HypergraphOfThoughts(GraphOfThoughts):
    """
    Extends GraphOfThoughts with hyperedges that can connect multiple nodes at once.
    Useful for capturing higher-order relationships in the reasoning process.
    """
    def __init__(self):
        super().__init__()
        self.hyperedges: Dict[int, Hyperedge] = {}
        self.current_edge_id = 0

    def add_hyperedge(self, node_ids: List[int], description: str = "") -> int:
        """
        Create a new hyperedge connecting multiple nodes simultaneously.
        """
        edge_id = self.current_edge_id
        self.hyperedges[edge_id] = Hyperedge(edge_id, node_ids, description)
        self.current_edge_id += 1
        return edge_id

    def get_hyperedge(self, edge_id: int) -> Optional[Hyperedge]:
        return self.hyperedges.get(edge_id)

    def visualize(self) -> str:
        """
        Extended visualization to include hyperedges as well as node relationships.
        """
        node_visual = super().visualize()
        edge_visual = []
        for he in self.hyperedges.values():
            edge_visual.append(
                f"Hyperedge {he.id}:\n"
                f"  Connects nodes: {he.connected_nodes}\n"
                f"  Description: {he.description}\n"
            )
        return node_visual + "\n" + "\n".join(edge_visual)

    def prune_hyperedges(self, valid_node_ids: List[int]):
        """
        Remove hyperedges if none or only one of their connected nodes survive pruning.
        """
        new_hyperedges = {}
        for he_id, he in self.hyperedges.items():
            surviving = [nid for nid in he.connected_nodes if nid in valid_node_ids]
            if len(surviving) > 1:
                he.connected_nodes = surviving
                new_hyperedges[he_id] = he
        self.hyperedges = new_hyperedges

    def prune_graph(self, min_score: float = 0.5):
        """
        Override prune_graph to also remove hyperedges that no longer connect
        two or more valid nodes.
        """
        super().prune_graph(min_score)
        valid_node_ids = list(self.nodes.keys())
        self.prune_hyperedges(valid_node_ids)

# ---------------------------
# Memory Functions
# ---------------------------
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

def summarize_memory(memory: List[dict]) -> str:
    summary_prompt = "Summarize the following conversation:\n" + "\n".join(
        [f"User: {entry['user_message']}\nResponse: {entry['response_content']}" for entry in memory]
    )
    summarizer_model = get_llm_model("openai", temperature=0.2, top_p=0.9, max_tokens=1500)
    summary_response = summarizer_model.invoke([HumanMessage(content=summary_prompt)])
    return summary_response.content

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

# ---------------------------
# Multi-Agent Functions
# ---------------------------
def multi_agent_invoke(models: list, graph: GraphOfThoughts, max_iterations: int = 1) -> str:
    """
    Loops over each model (agent), sends the current context, and adds its output as a node.
    Includes a feedback loop to refine the response iteratively.
    """
    log_phase("=== Multi-Agent Invocation Phase ===")
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        log_phase(f"=== Iteration {iteration} ===")

        # Get context nodes (include RAG nodes if applicable)
        context_nodes = graph.get_relevant_nodes(["system", "user", "model", "rag"])
        messages = [
            SystemMessage(content=node.content) if node.type == "system" else HumanMessage(content=node.content)
            for node in context_nodes
        ]

        # Invoke each model (agent)
        for i, model in enumerate(models):
            response = model.invoke(messages)
            new_node_id = graph.add_node(
                response.content,
                parent_ids=[n.id for n in context_nodes],
                node_type=f"agent_{i+1}"
            )
            log_phase(f"Agent {i+1} added node {new_node_id}: {response.content}")

        # Get the latest agent-generated node
        agent_nodes = [node for node in graph.nodes.values() if node.type.startswith("agent_")]
        if not agent_nodes:
            log_phase("No valid agent nodes found.")
            return "No valid response generated."

        latest_response = agent_nodes[-1].content

        # Evaluate the response
        evaluation = evaluate_response(latest_response, graph.get_node(graph.current_id - 1).content, graph)
        log_phase(f"Evaluation Score: {evaluation['score']}, Feedback: {evaluation['feedback']}")

        # Check if the response is satisfactory
        if evaluation["score"] >= 0.8:  # Threshold for acceptable quality
            graph.set_final_output(latest_response)
            log_phase(f"Final output set after {iteration} iterations: {latest_response}")
            return latest_response

        # Otherwise, update the context with feedback and re-invoke
        feedback_node_id = graph.add_node(
            evaluation["feedback"],
            parent_ids=[agent_nodes[-1].id],
            node_type="feedback"
        )
        log_phase(f"Added feedback node {feedback_node_id}: {evaluation['feedback']}")

    # If max iterations are reached, return the best response so far
    final_response = agent_nodes[-1].content if agent_nodes else "No valid response generated."
    graph.set_final_output(final_response)
    log_phase(f"Max iterations reached. Final output: {final_response}")
    return final_response

def invoke_with_retry(models: list, graph: GraphOfThoughts, retries: int = 3) -> str:
    """Wrapper to try multi-agent invocation multiple times in case of errors."""
    for attempt in range(retries):
        try:
            return multi_agent_invoke(models, graph)
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            log_phase(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == retries - 1:
                raise

def evaluate_response(response: str, query: str, graph: GraphOfThoughts) -> Dict[str, Any]:
    """
    Evaluate the quality of a response using an LLM.

    :param response: The generated response to evaluate.
    :param query: The original user query.
    :param graph: The reasoning graph containing the context.
    :return: A dictionary containing the evaluation score, feedback, and improvement suggestions.
    """
    evaluator_model = get_llm_model("mistral", temperature=0.2, top_p=0.9, max_tokens=25000)
    evaluation_prompt = (
        f"Evaluate the following response to the query '{query}':\n"
        f"Response: {response}\n\n"
        f"Provide a score between 0 and 1, and suggest improvements if necessary.\n"
        f"Criteria: Relevance, Accuracy, Completeness, Coherence.\n"
    )
    evaluation_response = evaluator_model.invoke([HumanMessage(content=evaluation_prompt)])

    try:
        # Parse the evaluation result
        lines = evaluation_response.content.strip().split("\n")
        score = float(lines[0].split(":")[1].strip())
        feedback = "\n".join(lines[1:])
        return {"score": score, "feedback": feedback}
    except Exception as e:
        logger.warning(f"Failed to parse evaluation response: {str(e)}")
        return {"score": 0.5, "feedback": "Evaluation failed."}
# ---------------------------
# LLM Model Initialization
# ---------------------------
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

# ---------------------------
# Streaming System Functions
# ---------------------------
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

import os
from typing import List

def retrieve_relevant_documents(query: str, top_k: int = 3, folder_path: str = "Data/output/flink") -> List[str]:
    """
    Retrieve relevant documents from a folder.

    :param query: User input or question
    :param top_k: Number of documents to retrieve
    :param folder_path: Path to the folder containing documents
    :return: List of retrieved documents
    """
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")

    # Get all text files in the folder
    document_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    # Placeholder for document content
    documents = []

    # Read the content of each file
    for file_name in document_files[:top_k]:  # Limit to top_k files
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(content)
        except Exception as e:
            logger.warning(f"Failed to read file {file_name}: {str(e)}")

    return documents

# ---------------------------
# Graph Building Functions
# ---------------------------
# def build_initial_graph(streaming_system: str, user_message: str,
#                         use_memory: bool, custom_template: str = None,
#                         hypergraph: bool = False) -> GraphOfThoughts:
#     """
#     Builds either a standard GraphOfThoughts or a HypergraphOfThoughts,
#     depending on the 'hypergraph' flag.
#     """
#     log_phase("=== Building Initial Graph Phase ===")
#     if hypergraph:
#         graph = HypergraphOfThoughts()
#         log_phase("Initialized a HypergraphOfThoughts.")
#     else:
#         graph = GraphOfThoughts()
#         log_phase("Initialized a GraphOfThoughts.")
#
#     # system_content = custom_template or (
#     #     f"Expert stream-processing assistant for {streaming_system}. "
#     #     f"Generate complete, production-grade pipelines with setup instructions."
#     # )
#
#     system_content = custom_template or (
#         f"You are a highly experienced architect in stream processing systems. Your task is to design and generate a complete, production-grade pipeline for {streaming_system}. "
#         "Follow these detailed steps:\n"
#         "1. **Requirements Gathering:** Identify the data sources, ingestion rates, and performance requirements.\n"
#         "2. **Data Ingestion:** Specify the mechanisms (e.g., Kafka, Pulsar, file (.txt)) and configurations for capturing real-time data.\n"
#         "3. **Data Transformation:** Outline the necessary processing steps, (e.g., filtering, aggregation, enrichment, and error handling).\n"
#         "4. **Pipeline Orchestration:** Design the workflow that ties together ingestion, transformation, and delivery stages, ensuring scalability and fault tolerance.\n"
#         "5. **Output Configuration:** Define the data sinks (e.g., databases, dashboards, message queues, file (.txt)) along with relevant connection settings.\n"
#         "Generate a comprehensive solution that includes all necessary configurations, code snippets, and setup instructions to deploy this pipeline in a production environment."
#     )
#
#     system_node = graph.add_node(system_content, node_type="system")
#     log_phase(f"Added system node {system_node}: {system_content}")
#
#     if use_memory:
#         for memory_entry in conversation_memory:
#             user_node = graph.add_node(memory_entry["user_message"], node_type="user")
#             response_node = graph.add_node(
#                 memory_entry["response_content"],
#                 parent_ids=[user_node],
#                 node_type="model"
#             )
#             graph.add_relationship(system_node, response_node)
#             log_phase(f"Added memory nodes: User node {user_node} and response node {response_node}")
#
#     user_node = graph.add_node(user_message, node_type="user")
#     graph.add_relationship(system_node, user_node)
#     log_phase(f"Added current user node {user_node}: {user_message}")
#
#     if hypergraph and isinstance(graph, HypergraphOfThoughts):
#         edge_id = graph.add_hyperedge(
#             node_ids=[system_node, user_node],
#             description="Initial synergy between system instructions and user request"
#         )
#         log_phase(f"Created hyperedge {edge_id} connecting nodes {system_node} and {user_node}")
#
#     return graph

# ---------------------------
# Interactive Mode
# ---------------------------

def build_initial_graph(
        streaming_system: str,
        user_message: str,
        use_memory: bool = False,
        custom_template: Optional[str] = None,
        hypergraph: bool = False,
        use_rag: bool = True  # New flag for RAG
) -> GraphOfThoughts:
    """
    Builds either a standard GraphOfThoughts or a HypergraphOfThoughts.
    """
    log_phase("=== Building Initial Graph Phase ===")
    if hypergraph:
        graph = HypergraphOfThoughts()
        log_phase("Initialized a HypergraphOfThoughts.")
    else:
        graph = GraphOfThoughts()
        log_phase("Initialized a GraphOfThoughts.")

    # Add system instructions node
    system_content = custom_template or (
        f"You are a highly experienced architect in stream processing systems. Your task is to design and generate a complete, production-grade pipeline for {streaming_system}. "
        "Follow these detailed steps:\n"
        "1. Requirements Gathering: Identify the data sources, ingestion rates, and performance requirements.\n"
        "2. Data Ingestion: Specify the mechanisms (e.g., Kafka, Pulsar) and configurations for capturing real-time data.\n"
        "3. Data Transformation: Outline the necessary processing steps, including filtering, aggregation, enrichment, and error handling.\n"
        "4. Pipeline Orchestration: Design the workflow that ties together ingestion, transformation, and delivery stages, ensuring scalability and fault tolerance.\n"

    )
    system_node = graph.add_node(system_content, node_type="system")
    log_phase(f"Added system node {system_node}: {system_content}")

    # Add memory nodes if enabled
    if use_memory:
        for memory_entry in conversation_memory:
            user_node = graph.add_node(memory_entry["user_message"], node_type="user")
            response_node = graph.add_node(
                memory_entry["response_content"],
                parent_ids=[user_node],
                node_type="model"
            )
            graph.add_relationship(system_node, response_node)
            log_phase(f"Added memory nodes: User node {user_node} and response node {response_node}")

    # Add user message node
    user_node = graph.add_node(user_message, node_type="user")
    graph.add_relationship(system_node, user_node)
    log_phase(f"Added current user node {user_node}: {user_message}")

    # Add RAG-retrieved documents as nodes
    if use_rag:
        rag_docs = retrieve_relevant_documents(user_message)
        for doc in rag_docs:
            doc_node = graph.add_node(doc, node_type="rag")
            graph.add_relationship(user_node, doc_node)
            log_phase(f"Added RAG document node {doc_node}: {doc}")

    # Add hyperedges if using HypergraphOfThoughts
    if hypergraph and isinstance(graph, HypergraphOfThoughts):
        edge_id = graph.add_hyperedge(
            node_ids=[system_node, user_node],
            description="Initial synergy between system instructions and user request"
        )
        log_phase(f"Created hyperedge {edge_id} connecting nodes {system_node} and {user_node}")

    return graph

# ---------------------------
# Main Function with Query Analyzer Integration
# ---------------------------

# Add these changes to your interactive_mode_with_planner function in deepGoT_main.py:
def interactive_mode_with_planner(args):
    global conversation_memory
    conversation_memory = choose_memory_file()

    # Initialize models
    models = [
        get_llm_model(model, args.temperature, args.top_p, args.max_tokens)
        for model in args.models.split(',')
    ]

    # Create a dedicated planning model with lower temperature
    planning_model = get_llm_model(args.models.split(',')[0], 0.2, 0.9, 2000)

    # Try to initialize backup models if available
    backup_models = []
    if args.backup_models:
        backup_model_names = args.backup_models.split(',')
        for model_name in backup_model_names:
            try:
                model = get_llm_model(model_name, args.temperature, args.top_p, args.max_tokens)
                backup_models.append(model)
                log_phase(f"Initialized backup model: {model_name}")
            except Exception as e:
                log_phase(f"Failed to initialize backup model {model_name}: {str(e)}")

    # If backup models are available, add them to the models list
    if backup_models:
        models.extend(backup_models)
        log_phase(f"Using {len(models)} total models for resilient execution")

    # Initialize the query analyzer
    query_analyzer = QueryAnalyzer(planning_model)

    # Initialize retry handler
    from resilient_execution import ResilientModelHandler
    retry_handler = ResilientModelHandler(models)

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        log_phase("╔════════════════════════════════════════════╗")
        log_phase("║  STREAM PROCESSING ASSISTANT (RESILIENT)   ║")
        log_phase("╚════════════════════════════════════════════╝")

        system = prompt_for_streaming_system()
        user_msg = input("\n📝 Your query: ")
        use_memory = input("💾 Use memory? (y/n): ").lower() == "y"
        hypergraph_choice = input("🔀 Use Hypergraph of Thoughts? (y/n): ").lower() == "y"
        use_rag = input("📚 Use RAG (Retrieval-Augmented Generation)? (y/n): ").lower() == "y"

        custom_template = None
        if args.prompt_file:
            with open(args.prompt_file, 'r') as f:
                custom_template = f.read()
        else:
            custom_input = input("⚙️  Custom template (Enter to skip): ")
            if custom_input:
                custom_template = custom_input

        # 1. Analyze the user query to detect intent with resilient handling
        log_phase("=== Query Analysis Phase ===")

        # Use the retry handler for intent detection
        intent_prompt = (
            f"Analyze the following user query and determine the most likely intent. "
            f"Respond with a JSON object containing 'intent_type' and 'confidence' (0-1).\n\n"
            f"User query: {user_msg}\n\n"
            f"Possible intent types: pipeline_design, explanation, comparison, troubleshooting, "
            f"optimization, general_question\n\n"
            f"JSON response:"
        )

        intent_response = retry_handler.invoke_with_retry(
            intent_prompt,
            # Default fallback if all models fail
            default_result='{"intent_type": "pipeline_design", "confidence": 0.8}'
        )

        # Parse the intent response
        import json
        import re
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', intent_response, re.DOTALL)
            if json_match:
                intent_json = json.loads(json_match.group(1))
            else:
                intent_json = json.loads(intent_response)

            from query_analyzer import QueryIntent
            intent = QueryIntent(
                intent_type=intent_json.get("intent_type", "pipeline_design"),
                confidence=intent_json.get("confidence", 0.8)
            )
        except Exception as e:
            log_phase(f"Error parsing intent: {str(e)}")
            # Fallback to default intent
            from query_analyzer import QueryIntent
            intent = QueryIntent(intent_type="pipeline_design", confidence=0.8)

        log_phase(f"Detected intent: {intent.intent_type} (confidence: {intent.confidence:.2f})")

        # Extract parameters if confidence is high enough
        if intent.confidence >= 0.6:
            try:
                parameters_prompt = (
                    f"Extract relevant parameters from this {intent.intent_type} query: '{user_msg}'\n\n"
                    "Return a JSON object with parameters appropriate for this type of request. "
                    "For example, for a pipeline_design intent, extract parameters like data_source, "
                    "throughput_requirements, latency_requirements, etc.\n\n"
                    "JSON response:"
                )

                parameters_response = retry_handler.invoke_with_retry(
                    parameters_prompt,
                    # Default fallback if all models fail
                    default_result='{}'
                )

                # Parse parameters
                try:
                    json_match = re.search(r'```json\s*(.*?)\s*```', parameters_response, re.DOTALL)
                    if json_match:
                        parameters = json.loads(json_match.group(1))
                    else:
                        parameters = json.loads(parameters_response)

                    intent.parameters = parameters
                    log_phase(f"Extracted parameters: {parameters}")
                except Exception as e:
                    log_phase(f"Error parsing parameters: {str(e)}")
                    log_phase("Continuing with basic parameters.")
                    intent.parameters = {}
            except Exception as e:
                log_phase(f"Error extracting parameters: {str(e)}")
                log_phase("Continuing with basic parameters.")
                intent.parameters = {}

        # 2. Create an execution plan based on the intent
        execution_plan = query_analyzer.create_execution_plan(
            query=user_msg,
            intent=intent,
            streaming_system=system,
            use_rag=use_rag
        )
        log_phase("Created execution plan:")
        log_phase(execution_plan.visualize())

        # 3. Build the initial graph with system instructions and user query
        thought_graph = build_initial_graph(
            streaming_system=system,
            user_message=user_msg,
            use_memory=use_memory,
            custom_template=custom_template,
            hypergraph=hypergraph_choice,
            use_rag=use_rag
        )
        log_phase("Initial graph built successfully:")
        log_phase(thought_graph.visualize())

        # 4. Add query analysis results to the graph
        analysis_node = thought_graph.add_node(
            f"Query analysis: Intent={intent.intent_type}, Confidence={intent.confidence:.2f}",
            node_type="analysis"
        )
        plan_node = thought_graph.add_node(
            f"Execution plan created with {len(execution_plan.steps)} steps",
            parent_ids=[analysis_node],
            node_type="plan"
        )
        log_phase(f"Added analysis node {analysis_node} and plan node {plan_node}")

        # 5. Execute the plan with resilient execution
        from resilient_execution import create_resilient_executor
        plan_executor = create_resilient_executor(models, thought_graph, system)
        response = plan_executor.execute_plan(execution_plan)

        # 6. Save session summary
        summary_path = plan_executor.save_session_summary(user_msg, intent.intent_type)
        log_phase(f"Saved session summary to: {summary_path}")

        # 7. Set final output
        thought_graph.set_final_output(response)
        log_phase("=== Final Response Phase ===")
        log_phase(response)

        print("\n🔍 Processing complete!")
        print("╔════════════════════════════════════════════╗")
        print("║                RESPONSE                   ║")
        print("╠════════════════════════════════════════════╣")
        print(response)
        print("╚════════════════════════════════════════════╝")

        # Display file saving information
        print("\n💾 Files saved:")
        print(f"  Session summary: {summary_path}")
        java_dir = os.path.join("query_analyzer_results", f"session_{plan_executor.results_saver.timestamp}", "generated_code", "java")
        if os.path.exists(java_dir):
            java_files = [f for f in os.listdir(java_dir) if f.endswith('.java')]
            if java_files:
                print("  Java files:")
                for java_file in java_files:
                    print(f"    - {os.path.join(java_dir, java_file)}")

        if input("\n🔧 Show thought graph? (y/n): ").lower() == "y":
            graph_viz = thought_graph.visualize()
            print("\n🧠 Thought Process Visualization:")
            print(graph_viz)
            log_phase("Thought Process Visualization:")
            log_phase(graph_viz)

        conversation_memory.append({
            "streaming_system": system,
            "user_message": user_msg,
            "response_content": response,
            "query_intent": intent.intent_type,
            "graph_visualization": thought_graph.visualize(),
            "results_path": summary_path
        })
        save_memory(conversation_memory)

        if input("\n🔄 Continue? (y/n): ").lower() != "y":
            log_phase("\n👋 Session ended. Goodbye!")
            break

# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="GoT/HGoT-enhanced LLM Pipeline Generator")
    parser.add_argument("--models", default="mistral",
                        help="Comma-separated models: mistral,openai,anthropic,cohere,groq")
    parser.add_argument("--backup_models", default="",
                        help="Comma-separated backup models for resilient execution")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=25000)
    parser.add_argument("--prompt_file", help="Path to custom system prompt template")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode")
    parser.add_argument("--use_planner", action="store_true", help="Enable the query analyzer with planning")
    parser.add_argument("--resilient", action="store_true", help="Enable resilient execution with retry and fallback")
    parser.add_argument("--results_dir", default="query_analyzer_results",
                        help="Directory for saving step results and code files")
    parser.add_argument("--max_retries", type=int, default=5,
                        help="Maximum number of retries for API calls")
    args = parser.parse_args()

    # Clear the phase log file at the start of a new run
    with open(utils.PHASE_LOG_FILE, "w") as f:
        f.write("Phase Log Started at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

    if args.interactive:
        if args.use_planner:
            if args.resilient:
                interactive_mode_with_planner(args)  # Uses resilient execution by default now
            else:
                interactive_mode_with_planner(args)  # Uses standard execution
        else:
            print("Use --use_planner to execute the code properly.")
    else:
        print("Single-run mode not fully implemented. Use --interactive for GoT/HGoT features.")


if __name__ == "__main__":
    main()