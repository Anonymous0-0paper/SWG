import os

def configure_langchain_environment():
    """
    Configures the environment variables required to connect to LangChain Smith.
    """
    # Set LangChain environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = "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"
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
    os.environ["OPENAI_API_KEY"] = "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"
    os.environ["ANTHROPIC_API_KEY"] = "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"  # Replace with your actual Anthropic API key if you want to test that model
    os.environ["MISTRAL_API_KEY"] = "QaxMSLno2alGaYiyYzhxSdojqQpEsVtv" # Replace with your actual Mistral API key
    os.environ["GROQ_API_KEY"] = "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"#"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"
    os.environ["COHERE_API_KEY"] = "q8ZssJ9hNEtVVp4Xs4mWV2S5LT6TNQ2llJrXksNP"

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

# Example usage
if __name__ == "__main__":
    configure_langchain_environment()
    # Proceed with your app initialization
    print("App is now ready to use LangChain.")

    print("="*80)

    configure_llms_environment()
    print("App is now ready to use LLMS.")
