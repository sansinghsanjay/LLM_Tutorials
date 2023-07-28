# packages
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentType

# get llm instance
llm = OpenAI()
print("LLM Name: ", llm.model_name)
print("LLM Temperature: ", llm.temperature)

# define embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# load documents
f_ptr = open("./../data/story_1.txt", "r")
data1 = f_ptr.read()
f_ptr.close()
f_ptr = open("./../data/story_2.txt", "r")
data2 = f_ptr.read()
f_ptr.close()
data = [
    data1,
    data2,
]

# define text splitter and create documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=64, chunk_overlap=0)
docs = text_splitter.create_documents(data)

# create DeepLake dataset
org_id = "sansinghsanjay"
db_name = "my_story_db"
db_path = f"hub://{org_id}/{db_name}"
db = DeepLake(dataset_path = db_path, embedding_function = embeddings)
db.add_documents(docs)

# create a chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
)

# create a tool
tools = [
    Tool(
        name="my story tool",
        func=chain.run,
        description="a tool to refer my stories",
    ),
]

# create an agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)

# interaction with user
usr_msg = ""
while(usr_msg != "quit"):
    usr_msg = input("User: ")
    if(usr_msg != "quit"):
        llm_response = agent.run(usr_msg)
        llm_response = llm_response.strip()
        print("AI: " + llm_response)
        print("")
