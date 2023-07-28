# packages
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# define template
template = """
Question: {question}
Answer: Let's think step by step."""

# build the prompt
prompt = PromptTemplate(
    input_variables = ['question'],
    template = template,
)

# define the callback manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# load the llm
llm = GPT4All(model="C:/Users/sanjaysingh5/Documents/sandocs/github_projects/ActiveLoop_projects/models/gpt4all-lora-quantized-ggml.bin", callback_manager=callback_manager, verbose=True)

# define the chain
chain = LLMChain(llm=llm, prompt=prompt)

# interaction
question = "What happens when it rains somewhere?"
chain.run(question)
