# packages
from langchain.llms import OpenAI

# define OpenAI Key
## OpenAI key is already defined in environment variables
# defining parameters
llm = OpenAI()

# update the status
print("LLM Name: ", llm.model_name)
print("LLM Temperature: ", llm.temperature)

# define a new value for temperature
llm.temperature = 0.9
print("New value of LLM Temperature: ", llm.temperature)

# define the prompt
text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
print("Prompt: ", text)
print("")

# get and print the response of LLM
llm_response = llm(text)
llm_response = llm_response.strip()
print("LLM Response:\n", llm_response)
