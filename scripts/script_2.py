# packages
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# get OpenAI instance
llm = OpenAI()
print("LLM Name: ", llm.model_name)
print("LLM Temperature: ", llm.temperature)

# define PromptTemplate
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# define chain
chain = LLMChain(llm=llm, prompt=prompt)

# run the chain
print(chain.run("eco-friendly water bottles"))
