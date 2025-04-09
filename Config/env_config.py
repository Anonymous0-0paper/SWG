import os

def configure_langchain_environment():
    """
    Configures the environment variables required to connect to LangChain Smith.
    """
    # Set LangChain environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = "YOUR_API_KEY"
    os.environ["LANGCHAIN_PROJECT"] = "SWG"


    # Verify configuration (optional: remove in production)
    if all(
            os.getenv(var)
            for var in [
                "LANGCHAIN_TRACING_V2",
                "LANGCHAIN_ENDPOINT",
                "LANGCHAIN_API_KEY",
                "LANGCHAIN_PROJECT",
            ]
    ):
        print("LangChain environment successfully configured.")
    else:
        print("Error: LangChain environment configuration failed. Check your setup.")


def configure_llms_environment():
    """
        Configures the environment variables required to connect to LLMs.
    """
    # Note: Setting API keys directly in code is not recommended for security reasons.
    # Consider using environment variables or a secrets management system instead, especially in production.
    # For demonstration purposes, we'll keep them here, but make sure to handle them securely in real-world applications.
    os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
    os.environ["ANTHROPIC_API_KEY"] = "YOUR_API_KEY"  # Replace with your actual Anthropic API key if you want to test that model
    os.environ["MISTRAL_API_KEY"] = "YOUR_API_KEY" # Replace with your actual Mistral API key
    os.environ["GROQ_API_KEY"] = "YOUR_API_KEY"#
    os.environ["COHERE_API_KEY"] = ""

    # Verify configuration (optional: remove in production)
    if all(
            os.getenv(var)
            for var in [
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "MISTRAL_API_KEY",
                "GROQ_API_KEY",
                "COHERE_API_KEY",
            ]
    ):
        print("Setting API KEY for LLMs in environment successfully configured.")
    else:
        print("Error: Setting API KEY for LLMs in environment configuration failed. Check your setup.")

if __name__ == "__main__":
    configure_langchain_environment()
    # Proceed with your app initialization
    print("App is now ready to use LangChain.")

    print("="*80)

    configure_llms_environment()
    print("App is now ready to use LLMS.")
