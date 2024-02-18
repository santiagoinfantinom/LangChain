#1st create a template
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv 
load_dotenv()


prompt = (
    PromptTemplate.from_template("Tell me a joke about {topic}")
    + ", make it funny"
    + "\n\nand in {language}"
)

PromptTemplate(input_variables=['language', 'topic'], output_parser=None, partial_variables={}, template='Tell me a joke about {topic}, make it funny\n\nand in {language}', template_format='f-string', validate_template=True)

prompt.format(topic="life under water", language="spanish")

#2.Use an LLMChain
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model = "gpt-3.5-turbo-1106", 
                   temperature=0.8,
                   base_url="https://polite-bags-shave.loca.lt/v1")

chain = LLMChain(llm=model, prompt=prompt) #Then create a Chain

#print(chain.run(topic="sports", language="spanish")) #Then run the chain

###### Chat prompt composition
from langchain.schema import AIMessage, HumanMessage, SystemMessage

#First, let’s initialize the base ChatPromptTemplate with a system message
prompt = SystemMessage(content="You are a nice pirate")

#Use a Message when there is no variables to be formatted, 
#use a MessageTemplate when there are variables to be formatted.

new_prompt = (
    prompt + HumanMessage(content="hi") + AIMessage(content="what?") + "{input}"
)

#Under the hood, 
#this creates an instance of the ChatPromptTemplate class, so you can use it just as you did before!

new_prompt.format_messages(input="i said hi")

from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

#U can use this in an LLM Chain, just as before
model = ChatOpenAI(model = "gpt-3.5-turbo-1106", 
                   temperature=0.8,
                   base_url="https://polite-bags-shave.loca.lt/v1")

#Then run the chain
chain = LLMChain(llm=model, prompt=new_prompt)

#print(chain.run("i said hi"))

#### Select by length
"""
This is useful when you are worried about constructing a prompt that will go over the length
 of the context window. For longer inputs, it will select fewer examples to include, 
 while for shorter inputs it will select more
 """

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector #This is the thing!

# Examples of a pretend task of creating antonyms.
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

#Example_prompt
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)
example_selector = LengthBasedExampleSelector(
    examples=examples,
    # The PromptTemplate being used to format the examples.
    example_prompt=example_prompt,
    max_length=25,
    # The function used to get the length of a string, which is used
    # to determine which examples to include. It is commented out because
    # it is provided as a default value if none is specified.
    # get_text_length: Callable[[str], int] = lambda x: len(re.split("\n| ", x))
)
dynamic_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples. This takes examples in/out depending on the number
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)

# An example with small input, so it selects all examples.
#print(dynamic_prompt.format(adjective="big"))

# An example with long input, so it selects only one example.
long_string = "big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else"
#print(dynamic_prompt.format(adjective=long_string))

# You can add an example to an example selector as well.
new_example = {"input": "big", "output": "small"}
dynamic_prompt.example_selector.add_example(new_example)
print(dynamic_prompt.format(adjective="enthusiastic"))