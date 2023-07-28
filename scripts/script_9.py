# packages
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# get instance of ChatOpenAI
chat = ChatOpenAI()

# define the template
system_template = "You are an assistant that helps user find information about movies."
system_msg_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_template = "Find information about the movie {movie_title}."
human_msg_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_msg_prompt, human_msg_prompt])

# get response
response = chat(chat_prompt.format_prompt(movie_title="Inception").to_messages())
print(response.content)
