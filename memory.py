from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

model = ChatOpenAI(
    model = "gpt-3.5-turbo",
    temperature = 0.7,
    base_url="https://polite-bags-shave.loca.lt"
)

prompt = ChatPromptTemplate.from_messages([
    ("system","You are a hardworking AI Assistant")
    MessagesPlaceholder(variable_name="chat_history")
    ("human","{input}"),
    base_url="https://polite-bags-shave.loca.lt"])

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

#chain = prompt | model
chain = LLMChain(
    llm=model,
    prompt=prompt,
    memory=memory,
    verbose=True
)

msg1 = {
    "input":"Hello"
}
response1 = chain.invoke(msg1)
print(response1)


msg2 = {
    "input":"My name is gallardo"
}
response2 = chain.invoke(msg2)
print(response2)