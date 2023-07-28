# packages
import os
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool, AgentType, load_tools
from langchain.utilities import GoogleSearchAPIWrapper

# set environment variables
os.environ['GOOGLE_CSE_ID'] = "75f72c8150c954652"
os.environ['GOOGLE_API_KEY'] = "AIzaSyB2F_hGlm_nB3xjy3TUZBE_LG8hFbwGrlw"

# create LLM instance
llm = OpenAI()
print("LLM Name: ", llm.model_name)
print("LLM Temperature: ", llm.temperature)

# define google search wrapper
search = GoogleSearchAPIWrapper()

# create tools
tools = [
    Tool(
        name="google-search",
        func=search.run,
        description="when you need to search google to answer questions about current events."
    ),
]

# create agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    max_iterations=6,
)

response = agent("What is the latest news about Chandrayaan-3?")
print(response['output'])
