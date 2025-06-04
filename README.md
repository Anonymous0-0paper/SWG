# AutoPipe: LLM Assisted Automatic Stream Processing Pipeline Generation

AutoPipe (a.k.a. SWG) is an advanced framework for automatic generation, validation, and improvement of stream processing pipelines using Large Language Models (LLMs). It supports multiple LLM providers (OpenAI, Anthropic, Mistral, Cohere, Groq) and a variety of streaming frameworks (Apache Flink, Spark, Kafka Streams, and more).

## Features
- **Automatic Pipeline Generation:** Generate production-grade code for stream processing systems from natural language queries.
- **Multi-Model Support:** Use and combine multiple LLMs for robust code generation and fallback.
- **Query Analyzer & Planner:** Decompose complex queries into executable plans with stepwise reasoning.
- **Validation & Feedback Loop:** Validate generated code against framework-specific rules and iteratively improve it.
- **Prompt Engineering:** Enhance prompts automatically based on validation feedback.
- **Memory & RAG:** Retain conversation history and retrieve relevant documents for context-aware generation.
- **Interactive & Batch Modes:** Use interactively or via CLI for single queries.

## Repository Structure
```
SWG/
├── main.py                  # Entry point for basic interactive mode
├── deepGoT_main.py          # Advanced GoT/HGoT mode with planning/validation
├── query_analyzer.py        # Query analysis, planning, and execution logic
├── validation_system.py     # Code validation and feedback loop
├── resilient_execution.py   # Resilient LLM invocation and plan execution
├── code_evaluation.py       # Code evaluation metrics and reporting
├── step_results_saver.py    # Utilities for saving step results
├── utils.py                 # Logging and document retrieval utilities
├── Query_docs.txt           # Example queries (see below)
├── requirement.txt          # Python dependencies
├── Config/
│   └── env_config.py        # LLM and LangChain environment configuration
├── Data/
│   ├── output/              # Example code/data for RAG
│   └── Dataflow/            # Pipeline diagrams and related files
├── memory_files/            # Saved conversation memory files
├── query_analyzer_results/  # Output: generated code, plans, logs
├── prompt_templates/        # Custom prompt templates (optional)
├── validation_history/      # Validation feedback history
├── validation_results/      # Validation output files
├── template/                # Additional templates
├── Output/                  # (Optional) Output directory
└── ... (other support files)
```

## Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd SWG
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirement.txt
   ```
3. **Set up API keys:**
   - Edit `Config/env_config.py` or set the following environment variables for LLM and LangChain access:
     - `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY`, `COHERE_API_KEY`
     - `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`, etc.
   - (For security, do **not** commit your API keys.)

## Usage
### 1. Basic Interactive Mode
```bash
python main.py --interactive
```
- Choose your streaming system (e.g., Apache Flink, Spark, Kafka Streams, etc.).
- Enter your pipeline requirements in natural language.
- Optionally load or save memory files for context.

#### CLI Options (main.py)
- `--models mistral,openai,...`  (comma-separated list)
- `--temperature 0.7`            (model creativity)
- `--top_p 0.9`                  (nucleus sampling)
- `--max_tokens 15000`           (response length)
- `--prompt_file <file>`         (custom system prompt)
- `--interactive`                (interactive session)

### 2. Advanced GoT/HGoT Mode (with Query Analyzer & Validation)
```bash
python deepGoT_main.py --interactive --use_planner --validate_code --enhance_prompts
```
- Enables stepwise query planning, code validation, and prompt engineering.
- Additional options:
  - `--resilient` (resilient execution with retry/fallback)
  - `--validation_iterations 3` (feedback loop iterations)
  - `--results_dir <dir>` (where to save outputs)

#### CLI Options (deepGoT_main.py)
See `python deepGoT_main.py --help` for all options.

## Example Queries
See `Query_docs.txt` for a variety of example queries, including:
- **Simple:** Word count, CSV transformation, log aggregation
- **Medium:** Event filtering, temperature monitoring, predictive maintenance
- **Complex:** Real-time chat moderation, image compression pipelines

### About `Query_docs.txt`
The `Query_docs.txt` file provides a curated set of example queries and pipeline requirements for use with AutoPipe. It is structured into three sections:
- **[Simple]**: Basic pipelines (e.g., word count, CSV transformation)
- **[Medium]**: Intermediate pipelines (e.g., event filtering, monitoring, predictive maintenance)
- **[Complex]**: Advanced pipelines (e.g., real-time chat moderation, image compression)

Each example includes a description of the pipeline, requirements, and sometimes detailed instructions for what the generated code should include. You can use these queries as input to the system, or extend the file with your own examples for testing and benchmarking.

## Output & Results
- Generated code, plans, and validation results are saved in `query_analyzer_results/` and `memory_files/`.
- Example output files:
  - `query_analyzer_results/session_*/final_response_*.txt`
  - `query_analyzer_results/session_*/execution_plan.txt`
  - `memory_files/memory_*.json`

## Extending & Customizing
- **Validation Rules:** Add or edit rules in `validation_system.py` or add JSON files under `validation_rules/`.
- **Prompt Templates:** Place custom templates in `prompt_templates/` and use `--prompt_file`.
- **Data & RAG:** Place relevant documents in `Data/output/<framework>/` for retrieval-augmented generation.

## Acknowledgments
- Built on [LangChain](https://www.langchain.com/), [OpenAI](https://openai.com/), [Anthropic](https://www.anthropic.com/), [Mistral](https://mistral.ai/), [Cohere](https://cohere.com/), [Groq](https://groq.com/).
