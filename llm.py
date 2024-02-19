from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

from langchain.prompts import ChatPromptTemplate

#A parser helps format outputs in a certain format
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

#Creates a prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

#An LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature= 0.7, #Between 0 (factual) and 1 (creative)
    max_tokens= 1000, #You are gonna be charged by the amount of tokens!
    verbose = False, #Allows easier debugging
    base_url="https://quick-rivers-enter.loca.lt/v1" #url del modelo local
    )

#The first chain
chain = prompt | llm | output_parser

# The first question. The print statement was essential!
#print(chain.invoke({"input": "How does the process of creating an LLM works?"}))

#Prompts: Quickstart
#PromptTemplate
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)
prompt_template.format(adjective="spicy", content="a grandma")

#ChatPromptTemplate
#The prompt to chat models is a list of chat messages
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a hateful AI bot. Your name is {name}."),
        ("human", "Hello, could i spit on your face?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ]
)
#ChatPromptTemplate.from_messages accepts a variety of message representations.

messages = chat_template.format_messages(name="Carlos", user_input="Did you like it?")
#print(messages) #Actually only repeats, what you give it

#You can do a lot more stuff with that tool. 
#F.e. like using (Human)MessagePromptTemplate or (System)BaseMessage
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful assistant that re-writes the user's text to "
                "sound more upbeat."
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)
messages = chat_template.format_messages(text="I don't like eating tasty things")
#print(messages) #That did'nt seem to have the output one would've expect

#Composition: You can do this with either string prompts or chat prompts.
#String prompt composition
from langchain.prompts import PromptTemplate

prompt = (
    PromptTemplate.from_template("Tell me a joke about {topic}")
    + ", make it funny"
    + "\n\nand in {language}"
)

PromptTemplate(input_variables=['language', 'topic'], output_parser=None, partial_variables={}, template='Tell me a joke about {topic}, make it funny\n\nand in {language}', template_format='f-string', validate_template=True)
prompt.format(topic="sports", language="spanish") #Actually creates only 'Tell me a joke about sports, make it funny\n\nand in spanish'

#But you can also use it with LLMChain
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run(topic="music", language="spanish"))



#Simple Prompt-response structure
response = llm.invoke("What day of the week is it today?") 
#print(response.content)


#2 prompts runned in Paralel
response2 = llm.batch(["What's in right now?","Tell me a dad joke about a pizza"])
#print(response2)

#3 Streams a response (i.e. prints out as its beeing created) and does'nt need a print statement
response3 = llm.invoke("Write a poem about organic life")
#print(response3.content)