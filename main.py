import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
tools = [TavilySearch()]

#@tool
#def search(query: str) -> str:
    #"""Tool that searches over internet
    #Args:
    #    query: The query to search for
    #Returns:
    #    The search result
    #"""
    #print(f"Searching for {query}...")
    #return tavily_client.search(query=query)


def main():
    print("Hello from langchain-course!")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    agent = create_agent(model=llm, tools=tools)

    result = agent.invoke({"messages": [HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()

