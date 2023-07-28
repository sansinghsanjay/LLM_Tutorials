# packages
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

# get instance of OpenAI
llm = OpenAI()

# set the value of temperature to 0 to get deterministic responses for summarization
llm.temperature = 0

# update the llm name and temperature value
print("LLM Name: ", llm.model_name)
print("LLM Temperature: ", llm.temperature)

# load the summarization chain
summarize_chain = load_summarize_chain(llm)

# load the document using PyPDFLoader
doc_loader = PyPDFLoader(file_path="./../data/Floating Islands.pdf")
document = doc_loader.load()

# summarize the document
summary = summarize_chain(document)

# print summary
print(summary['output_text'])
