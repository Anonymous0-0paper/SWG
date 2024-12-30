import os
import subprocess
import shutil
import argparse
import json
import time
import gzip
import hashlib
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Config:
    """
    Configuration class for repository processing.

    Attributes:
        default_repositories (List[str]): List of default repository URLs to process
        max_retries (int): Maximum number of retry attempts for git operations
        chunk_size (int): Size of chunks for reading/writing files
        timeout (int): Timeout in seconds for git operations
        max_file_size (int): Maximum allowed file size in bytes
        file_extensions (List[str]): List of file extensions to process
        compress_output (bool): Whether to compress output files
        auth_token (Optional[str]): GitHub authentication token
    """
    default_repositories: List[str]
    max_retries: int
    chunk_size: int
    timeout: int
    max_file_size: int
    file_extensions: List[str]
    compress_output: bool
    auth_token: Optional[str]

def load_config(config_file: str = 'config.json') -> Config:
    """
    Load configuration from a JSON file with fallback to default values.

    Args:
        config_file (str): Path to the configuration JSON file

    Returns:
        Config: Configuration object with merged default and user settings

    Notes:
        - If config file is not found, uses default values
        - User config values override default values when present
    """
    default_config = {
        'default_repositories': [
            "https://github.com/apache/flink.git",
            "https://github.com/apache/storm.git",
            "https://github.com/apache/spark.git"
        ],
        'max_retries': 3,
        'chunk_size': 8192,
        'timeout': 300,
        'max_file_size': 10_000_000,  # 10MB
        'file_extensions': ['.java'],
        'compress_output': False,
        'auth_token': None
    }

    try:
        with open(config_file, 'r') as f:
            user_config = json.load(f)
            default_config.update(user_config)
    except FileNotFoundError:
        logging.warning(f"Config file {config_file} not found, using defaults")

    return Config(**default_config)

def calculate_checksum(file_path: str) -> str:
    """
    Calculate SHA-256 checksum of a file.

    Args:
        file_path (str): Path to the file

    Returns:
        str: Hexadecimal string representation of the SHA-256 hash

    Notes:
        - Reads file in 4KB chunks to handle large files efficiently
        - Uses SHA-256 for strong cryptographic hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def clone_or_update_repository(repo_url: str, clone_path: str, config: Config) -> bool:
    """
    Clone a new repository or update an existing one.

    Args:
        repo_url (str): URL of the GitHub repository
        clone_path (str): Local path for cloning
        config (Config): Configuration object

    Returns:
        bool: True if operation successful, False otherwise

    Notes:
        - Implements retry mechanism with exponential backoff
        - Supports authentication via GitHub token
        - Sets timeout for git operations
        - Handles both clone and pull operations
    """
    env = os.environ.copy()
    if config.auth_token:
        repo_url = repo_url.replace('https://', f'https://{config.auth_token}@')

    for attempt in range(config.max_retries):
        try:
            if os.path.exists(clone_path):
                subprocess.run(
                    ["git", "-C", clone_path, "pull"],
                    check=True,
                    timeout=config.timeout,
                    env=env
                )
            else:
                subprocess.run(
                    ["git", "clone", repo_url, clone_path],
                    check=True,
                    timeout=config.timeout,
                    env=env
                )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            if attempt < config.max_retries - 1:
                wait_time = 2 ** attempt
                logging.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed after {config.max_retries} attempts: {e}")
                return False
    return False

def find_example_folders(path: str) -> List[str]:
    """
    Find all folders that contain 'example' in their name.

    Args:
        path (str): Root path to start searching from

    Returns:
        List[str]: List of paths to example folders
    """
    example_folders = []
    for root, dirs, _ in os.walk(path):
        for dir_name in dirs:
            if 'example' in dir_name.lower():
                example_folders.append(os.path.join(root, dir_name))
    return example_folders

def save_java_files_to_txt(repo_path: str, target_folder: str, output_file: str, config: Config) -> Dict:
    """
    Save specified files from repository to a text file.

    Args:
        repo_path (str): Path to the repository
        target_folder (str): Folder to search for files
        output_file (str): Output file path
        config (Config): Configuration object

    Returns:
        Dict: Statistics about processed files

    Notes:
        - Supports file compression
        - Implements file size limits
        - Calculates checksums for verification
        - Handles files in chunks for memory efficiency
        - Tracks processing statistics
    """
    stats = {'processed': 0, 'skipped': 0, 'errors': 0}
    target_path = os.path.join(repo_path, target_folder)

    if not os.path.exists(target_path):
        logging.warning(f"Target folder '{target_folder}' does not exist")
        return stats

    with open(output_file + '.txt', 'w', encoding='utf-8') as out_file:
        for root, _, files in os.walk(target_path):
            for file in files:
                if any(file.endswith(ext) for ext in config.file_extensions):
                    file_path = os.path.join(root, file)

                    if os.path.getsize(file_path) > config.max_file_size:
                        logging.warning(f"Skipping {file_path}: exceeds size limit")
                        stats['skipped'] += 1
                        continue

                    try:
                        with open(file_path, 'r', encoding='utf-8') as src_file:
                            checksum = calculate_checksum(file_path)
                            out_file.write(f"// File: {file_path}\n")
                            out_file.write(f"// Checksum: {checksum}\n")

                            while True:
                                chunk = src_file.read(config.chunk_size)
                                if not chunk:
                                    break
                                out_file.write(chunk)
                            out_file.write("\n\n")
                            stats['processed'] += 1
                    except Exception as e:
                        logging.error(f"Error processing {file_path}: {e}")
                        stats['errors'] += 1

    return stats

def process_repository(repo_url: str, target_folder: str, output_dir: str,
                       clone_base_dir: str, config: Config) -> Dict:
    repo_name = repo_url.rstrip('.git').split('/')[-1]
    repo_clone_path = os.path.join(clone_base_dir, repo_name)
    output_file = os.path.join(output_dir, f"{repo_name}_java_files")

    if clone_or_update_repository(repo_url, repo_clone_path, config):
        stats = {'processed': 0, 'skipped': 0, 'errors': 0}

        # Process specified target folder
        if target_folder != "":
            target_stats = save_java_files_to_txt(repo_clone_path, target_folder, output_file, config)
            for key in stats:
                stats[key] += target_stats[key]

        # Find and process all example folders
        example_folders = find_example_folders(repo_clone_path)
        for example_folder in example_folders:
            relative_path = os.path.relpath(example_folder, repo_clone_path)
            example_output = f"{output_file}_{os.path.basename(example_folder)}"
            folder_stats = save_java_files_to_txt(repo_clone_path, relative_path, example_output, config)
            for key in stats:
                stats[key] += folder_stats[key]

        return stats
    return {'processed': 0, 'skipped': 0, 'errors': 1}
    """
    Process a single repository by cloning and extracting files.
    
    Args:
        repo_url (str): Repository URL
        target_folder (str): Target folder for file extraction
        output_dir (str): Output directory
        clone_base_dir (str): Base directory for clones
        config (Config): Configuration object
        
    Returns:
        Dict: Processing statistics
        
    Notes:
        - Handles repository cloning/updating
        - Manages file extraction and processing
        - Returns processing statistics
    """
    repo_name = repo_url.rstrip('.git').split('/')[-1]
    repo_clone_path = os.path.join(clone_base_dir, repo_name)
    output_file = os.path.join(output_dir, f"{repo_name}_java_files")

    if clone_or_update_repository(repo_url, repo_clone_path, config):
        return save_java_files_to_txt(repo_clone_path, target_folder, output_file, config)
    return {'processed': 0, 'skipped': 0, 'errors': 1}

def process_repositories(repositories: List[str], target_folder: str, output_dir: str,
                         clone_base_dir: str, config: Config, threads: int = 4):
    """
    Process multiple repositories in parallel.

    Args:
        repositories (List[str]): List of repository URLs
        target_folder (str): Target folder for extraction
        output_dir (str): Output directory
        clone_base_dir (str): Base directory for clones
        config (Config): Configuration object
        threads (int): Number of parallel threads

    Notes:
        - Uses thread pool for parallel processing
        - Tracks overall progress and statistics
        - Creates necessary directories
        - Provides progress updates during processing
    """
    os.makedirs(clone_base_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    total_stats = {'processed': 0, 'skipped': 0, 'errors': 0}
    total = len(repositories)
    completed = 0

    def update_progress(future):
        nonlocal completed
        completed += 1
        stats = future.result()
        for key in total_stats:
            total_stats[key] += stats[key]
        logging.info(f"Progress: {completed}/{total} repositories processed")

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for repo_url in repositories:
            future = executor.submit(
                process_repository,
                repo_url,
                target_folder,
                output_dir,
                clone_base_dir,
                config
            )
            future.add_done_callback(update_progress)
            futures.append(future)

    logging.info(f"Final stats - Processed: {total_stats['processed']}, "
                 f"Skipped: {total_stats['skipped']}, Errors: {total_stats['errors']}")
    return total_stats

def main():
    """
    Main entry point for the script.

    Notes:
        - Handles command line argument parsing
        - Sets up logging configuration
        - Loads configuration
        - Initiates repository processing
    """
    parser = argparse.ArgumentParser(description="Extract Java files from GitHub repositories.")
    parser.add_argument("--repo_url", help="URL of a single GitHub repository")
    parser.add_argument("--target_folder", default="example", help="Target folder for extraction")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    parser.add_argument("--clone_base_dir", default="cloned_repos", help="Base clone directory")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--config", default="config.json", help="Config file path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('repository_processor.log'),
            logging.StreamHandler()
        ]
    )

    config = load_config(args.config)
    repositories = [args.repo_url] if args.repo_url else config.default_repositories
    process_repositories(
        repositories,
        args.target_folder,
        args.output_dir,
        args.clone_base_dir,
        config,
        args.threads
    )

if __name__ == "__main__":
    main()