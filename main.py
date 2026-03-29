import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from tavily import TavilyClient
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from typing import List

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent's response"""
    answer: str = Field(description="Answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of the sources used to generate the answer")

load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
tools = [TavilySearch(max_results=5, topic="general")]

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

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    #llm = ChatOllama(model="llama3.2:3b")

    agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

    result = agent.invoke({"messages": [HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()

