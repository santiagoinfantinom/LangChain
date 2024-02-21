from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

from langchain.chains import HumanMessage,AIMessage
from langchain.core import MessagesPlaceholder

#Version 1
"""
def get_doc_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs
"""

#Version 2 (TextSplitter)
def get_doc_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, #Changed for improving model's performance
        chunk_overlap = 20
    )
    splitDocs = splitter.split_documents(docs)
    print(len(splitDocs))
    return splitDocs

#print(get_doc_from_web())
def create_db(docs):
    embeddings = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs,embedding=embeddings)
    return vectorStore

def create_chain(vectorStore):
    docA = Document(
    page_content = "I'm havin fun learning Langchain")

    model = ChatOpenAI(
        model = "gpt-3.5-turbo-1106",
        temperature = 0.4,
        base_url="https://polite-bags-shave.loca.lt"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
        ("system","Answer the users question based on the context: {context}"),
        MessagesPlaceholder(variable_name = "chat_history")
        ("human", "{input}")
         ]
    )
    """
    prompt = ChatPromptTemplate.from_template(
       
        Answer the following question:
        Context: {context} #!NEEDS to be called context!
        Question: {input}
    )
    """
    retriever = vectorStore.as_retriever(search_kwargs={"k":3}) #Changed to increase the amount of answers provided
    retrieval_chain = create_retrieval_chain(
        retriever,
        chain #This only works because of create_stuff_documents_chain
    )

    return retrieval_chain 

#chain = prompt | model
chain = create_stuff_documents_chain(
    llm = model,
    prompt=prompt
    )


"""
response = chain.invoke(
    {
    "input":"Who's having fun with Langchain?",
    "context": [docA]
     }
)

print(response)
"""

def process_chat(chain, question, chat_history):
    response = chain.invoke(
        {
        "input":question,
        "context": docs,
        "chat_history" : chat_history
        }
    )

    return(response["answer"])

if __name__=='__main__':
    docs = get_doc_from_web(https://python.langchain.com/docs/modules/model_io/chat/)
    vectorStore = create_db(docs)
    chain = create_chain(vectorStore)

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

        response = process_chat(chain, user_input,chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant: ", response)
