# packages
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentType

##################################################################################################
# make sure that you have created your ACTIVELOOP_TOKEN and also set it as an environment variable
##################################################################################################

# create an instance of LLM
llm = OpenAI()
print("LLM Name: ", llm.model_name)
print("LLM Temperature: ", llm.temperature)

# define embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# create our document
texts = [
    "Napoleon Bonaparte was born in 15 August 1769.",
    "Louis XIV was born in 5 September 1638.",
]

# process the document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# create deeplake dataset
my_org_id = "sansinghsanjay"
my_db_name = "db_0"
dataset_path = f"hub://{my_org_id}/{my_db_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
db.add_documents(docs)

# create a retrieval qna
retrieval_qa = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever = db.as_retriever(),
)

# create a tool
tools = [
    Tool(
        name = "Retrieval QA System",
        func = retrieval_qa.run,
        description = "Useful for answering question",
    ),
]

# create an agent
agent = initialize_agent(
    tools,
    llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = False,
)

usr_msg = ""
while(usr_msg != "quit"):
    usr_msg = input("User: ")
    if(usr_msg != "quit"):
        response = agent.run(usr_msg)
        response = response.strip()
        print("AI: " + response)
        print("")
