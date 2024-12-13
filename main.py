import getpass
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq

from Config.env_config import configure_llms_environment

configure_llms_environment()

# Choose the model you want to use by uncommenting the corresponding line:
model = ChatMistralAI(model="mistral-large-latest") # MistralAI
# model = ChatOpenAI(model="gpt-4o")
# model = ChatAnthropic(model="claude-3-sonnet")
# model = ChatGroq(model="llama3-8b-8192") # Grok
# model = ChatCohere(model="command-r-plus") #Cohere

def get_streaming_system():
    """Get the streaming system from the user."""
    print("Select a streaming system:")
    print("1. Flink")
    print("2. Storm")
    print("3. Spark")
    print("4. Kafka Stream")
    # print("5. Other (please specify)")

    choice = input("Enter your choice (1/2/3/4): ")

    if choice == "1":
        return "Apache Flink"
    elif choice == "2":
        return "Apache Storm"
    elif choice == "3":
        return "Apache Spark"
    elif choice == "4":
        return "Kafka Stream"
    # elif choice == "5":
    #     return input("Please enter the streaming system by your specifics: ")
    else:
        print("Invalid choice. Defaulting to Flink.")
        return "Flink"

def main():
    streaming_system = get_streaming_system()
    user_message = input("Please enter your message: ")
    print("please be patient we are preparing your code")

    messages = [
        SystemMessage(content=f"you are a stream processing professional and please write a code for {streaming_system} with it's latest version based on user request"),
        HumanMessage(content=user_message),
    ]

    # Invoke the model and print the response
    response = model.invoke(messages)
    print(response.content)

if __name__ == "__main__":
    main()