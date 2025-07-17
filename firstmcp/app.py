import asyncio

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use.agents import MCPAgent
from mcp_use.client import MCPClient
import os

async def run_memory_chat():
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

    config_file = "firstmcp/browser_mcp.json"

    print("Starting MCPU server...")

    client = MCPClient.from_config_file(config_file)
    llm = ChatGroq(model="llama3-8b-8192")

    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=15,
        memory_enabled=True,
    )

    print("\n===== Interactive MCP Chat =====")
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'clear' to clear conversation history")
    print("=====================================\n")

    try:
        while True:
            user_input = input("\nYou: ")

            # Exit command
            if user_input.lower() in ["exit", "quit"]:
                print("Ending conversation...")
                break

            # Clear memory
            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("Conversation history cleared.")
                continue

            # Get response
            print("\nAssistant: ", end="", flush=True)
            try:
                response = await agent.run(user_input)
                print(response)
            except Exception as e:
                print(f"[Error] {str(e)}")

    except KeyboardInterrupt:
        print("\n[!] Exiting...")

if __name__ == "__main__":
    asyncio.run(run_memory_chat())