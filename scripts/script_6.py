# packages
import os
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.utilities import GoogleSearchAPIWrapper

# set environment variables
os.environ['GOOGLE_CSE_ID'] = "75f72c8150c954652"
os.environ['GOOGLE_API_KEY'] = "AIzaSyB2F_hGlm_nB3xjy3TUZBE_LG8hFbwGrlw"

# get instance of LLM
llm = OpenAI()
print("LLM Name: ", llm.model_name)
print("LLM Temperature: ", llm.temperature)

# define prompt
prompt = PromptTemplate(
    input_variables=['query'],
    template="Write a summary of the following text: {query}",
)

# define the chain
chain = LLMChain(llm=llm, prompt=prompt)

# create google search instance
search = GoogleSearchAPIWrapper()

# create tools
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for finding about recent events",
    ),
    Tool(
        name="Summarizer",
        func=chain.run,
        description="useful for summarizing texts",
    ),
]

# create an agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)

# interact
response = agent("Summarize the concept of Quantum Mechanics in 50 words. Then, tell me the date and title of latest breakthrough in Quantum Mechanics?")
print(response['output'])
