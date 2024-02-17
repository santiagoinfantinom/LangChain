from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeperatedListOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel,Field

model = ChatOpenAI(model = "gpt-3.5-turbo-1106", temperature=0.8)

def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tell me a joke about the following subject")
        ("human","{input}")
    ])
    
    parser = StrOutputParser()

    chain = prompt | model | parser

    return chain.invoke(
        "input" : "cow"
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
        name: str = Field(description="the name of the person")
        age: int = Field(description="the age of the person")
        #Also possible adress: dict, ingredients: list, u.s.w

    parser = JsonOutputParser(pydantic_object=Person) 

    chain = prompt | model | parser

    return chain.invoke({
        "phrase" : "Maximo turned 27 yesterday",
        "format_instructions":parser.get_format_instructions()
    })

print(JsonOutputParser())