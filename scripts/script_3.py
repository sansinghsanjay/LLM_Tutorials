# packages
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# get the LLM
llm = OpenAI()
print("LLM Name: ", llm.model_name)
print("LLM Temperature: ", llm.temperature)

# define conversation chain
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

# start the conversation
conversation.predict(input="Tell me about the south pole in 50 words.")

# further continue the conversation
conversation.predict(input="How far North Pole is from South Pole in Kilometers?")
conversation.predict(input="Which of the two poles is good for vacation in 5 words?")

# show the entire conversation
print(conversation)
