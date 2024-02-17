from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptsTemplate

#Instantiate Model
llm = ChatOpenAI(
    temperature=0.7,
    model = "gpt-3.5-turbo"
)

# Prompt template
prompt = ChatPromptsTemplate.from_template("Tell me a joke about {subject}")

# Create LLM Chain
chain = prompt | llm # | operator is essential! its done with AltGr

response = chain.invoke({"subject":"cow"}) #Here is the {} op. also super important
print(response)

# Prompt template 2
prompt = ChatPromptsTemplate.from_messages(
    [
        ("system","You are an AI Chef. Create a creative recipe with the following ingredients")
        ("human", "{input}")
    ]
    )

response = chain.invoke({"input":"tomatoes"}) #Here is the {} op. also super important
print(response)

# Prompt template 3
prompt = ChatPromptsTemplate.from_messages(
    [
        ("system","Generate a list of 8 synonyms for the following word. Return the result as a comma separated list         ")
        ("human", "{input}")
    ]
    )

response = chain.invoke({"input":"funny"}) #Here is the {} op. also super important
print(response)
