from dotenv import load_dotenv 
load_dotenv()

from langchain_openai import ChatOpenAI

#The prompt to chat models is a list of chat messages.
#Each chat message is associated with content, and an additional parameter called role.
#(f.e "system","human","ai")
#Used for creating a template for prompting
from langchain.prompts import ChatPromptTemplate


#Parser define how the output of a function should look like
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.output_parsers import CommaSeparatedListOutputParser


#Pydantic is a data validation library
#Models are simply pydantic classes which inherit from BaseModel 
#and define Fields as annotated attributes
from langchain_core.pydantic_v1 import BaseModel,Field

model = ChatOpenAI(model = "gpt-3.5-turbo-1106", 
                   temperature=0.8,
                   base_url="https://polite-bags-shave.loca.lt") #Necesita el base_url 

def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        #Prompt global de todo el sistema, cual es el uso del LLM? 
        #Quien es/ Que papel debe jugar/ Como quien se debe comportar? Eso es el system
        ("system", "Tell me a joke about the following subject")
        ("human","{input}")
    ])
    
    #Transforms the data into an specific format
    parser = StrOutputParser() 

    chain = prompt | model | parser

    #Invoke takes an input and an optional config. 
    #Then processes it using the specified agent or chain
    #Finally, returns an output based on the applied computation

    return chain.invoke(
        {"input" : "cow"}
        ) 
    print(call_string_output_parser)

def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system","Generate 3 antonyms for the following word. Return the result as a comma separated list")
        ("human", {input})
    ])

    parser = CommaSeperatedListOutputParser()

    chain = prompt | model | parser

    return chain.invoke({
        "input": "sad"
        })

def call_json_output_parser():
    prompt = ChatPromptTemplate(
        [
            ("system", " Extract information from the following phrase.\nFormatting Instructions {format_instructions}")
            ("human", {phrase}) 
        ]
    )

    class Person(BaseModel): #Useful for creating a Json
        name: str = Field(description="the name of the person") #Annotated attributes
        age: int = Field(description="the age of the person")
        #Also possible adress: dict, ingredients: list, u.s.w

    parser = JsonOutputParser(pydantic_object=Person) 

    chain = prompt | model | parser

    return chain.invoke({
        "phrase" : "Maximo turned 27 yesterday",
        "format_instructions":parser.get_format_instructions()
    })

print(JsonOutputParser())