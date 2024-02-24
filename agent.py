from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent,AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool

from langchain.chains import HumanMessage,AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

url = "https://python.langchain.com/docs/expression_language/"
loader = WebBaseLoader(url)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, #Changed for improving model's performance
    chunk_overlap = 20
)
splitDocs = splitter.split_documents(docs) #Split the document into chunks of data
embeddings = OpenAIEmbeddings()
vectorStore = FAISS.from_documents(docs,embedding=embeddings) #Create a VectorStore for the doc
retriever = vectorStore.as_retriever(search_kwargs={"k":3}) #Changed to increase the amount of answers provided
    

model = ChatOpenAI(
        model = "gpt-3.5-turbo-1106",
        temperature = 0.4,
        base_url="https://polite-bags-shave.loca.lt"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a friendly assistent called Javier"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")

])


search = TavilySearchResults() #Needs an TavilyAPI Key
retriever_tools = create_retriever_tool(
    retriever,
    "lcel_search",
    "Use this tool when searching for langchain expression language (LCEL)"
)
tools = [search, retriever_tools] #A list of tools, for searching and retrieving

agent = create_openai_functions_agent( #This is where the agent is created!
    llm=model,
    prompt=prompt,
    tools=tools
)

agentExecutor = AgentExecutor( #Initialize the agent. Necessary! 
    agent=agent,
    tools=tools
)

def process_chat(agentExecutor, user_input,chat_history): #Uses the agent executor to create a response
    response = agentExecutor.invoke({
        "input":user_input,
        "chat_history":chat_history
        })
    return response["output"]

if __name__=='__main__':
    chat_history = [
    ] 
    """This can contain anything: F.e         
    HumanMessage(content= "Hello"),
        AIMessage(content = "Hello, how can I assist you?"),
        HumanMessage(content = "My name is Gonzalo")"""

    while True:
        user_input = input("You:  ")
        if user_input.lower() == 'exit':
            break

        response = process_chat(agentExecutor, user_input,chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant: ", response)