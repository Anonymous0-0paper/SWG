# utils.py
import os
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a phase log file to save detailed outputs
PHASE_LOG_FILE = "phase_log.txt"
def log_phase(message: str):
    """Print the message to console and append it to the phase log file."""
    print(message)
    with open(PHASE_LOG_FILE, "a") as f:
        f.write(message + "\n")

def retrieve_relevant_documents(query: str, top_k: int = 3, folder_path: str = "Data/output/flink") -> list:
    """
    Retrieve relevant documents from a folder.
    """
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        logger.warning(f"Folder '{folder_path}' does not exist.")
        return []

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