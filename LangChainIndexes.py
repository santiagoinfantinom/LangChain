#First lets load the data to a vector database. The first step is to load into the memory
from langchain.document_loaders import DirectoryLoader, TextLoader

#Import the knowledge base
loader = DirectoryLoader(
    "./FAQ", glob="**/*.txt", loader_cls=TextLoader, show_progress=True
)
docs = loader.load()

#Define chunk_size and overlap
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

documents = text_splitter.split_documents(docs)
documents[0]
#Returns a Document-type element, with f.e. page_content

#Texts are not stored as text in the database, but as vector representations
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

#You always wanna know which embedding belongs to which text
from langchain.vectorstores.faiss import FAISS
import pickle

vectorstore = FAISS.from_documents(documents, embeddings)

#Then we "serialize" our vectorstore
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)

###Prompts
#Here you can "pre-prompt" the possible answers
from langchain.prompts import PromptTemplate

prompt_template = """You are a helpful assistant for our restaurant.

{context}

Question: {question}
Answer here:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

##Chains
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

chain_type_kwargs = {"prompt": PROMPT}

llm = OpenAI(openai_api_key=API_KEY)
qa = RetrievalQA.from_chain_type( #This is the magic
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs=chain_type_kwargs,
)

query = "When does the restaurant open?"
qa.run(query)

##Memory
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)

#Using memory in chains
from langchain.chains import ConversationalRetrievalChain

qa = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(model_name="text-davinci-003", temperature=0.7, base_url="https://polite-bags-shave.loca.lt"),
    memory=memory,
    retriever=vectorstore.as_retriever(),
    combine_docs_chain_kwargs={"prompt": PROMPT},
)


query = "Do you offer vegan food?"
qa({"question": query})
qa({"question": "How much does it cost?"})

