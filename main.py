from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from tavily import TavilyClient
import os


load_dotenv()

class Source(BaseModel):
    """Schema for a source by the agent"""
    name: str = Field(description="Name of the source")
    url: str = Field(description="URL of the source")

class AgentResonse(BaseModel):
    """Schema for the agent response and sources"""
    answer: str = Field(description="Final answer to the user")
    sources: List[Source] = Field(default_factory=list, description="Sources used to generate the answer")

@tool
def search(query: str) -> str:
    """Search the web for a query"""
    print(f"\n\nSearching for {query} \n\n")
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    return tavily.search(query=query)

tools = [search, TavilySearch()]
llm = ChatOpenAI()
agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResonse
)


def main(): 
    print("Hello from langchain-course!")
    response = agent.invoke({
        "messages":HumanMessage(content="Search for job openeings in India with 1 year experience and skill as AI Engineer and Langchain")
    })
    print(response)


if __name__ == "__main__":
    main()
