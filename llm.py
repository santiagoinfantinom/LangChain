from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature= 0.7, #Between 0 (factual) and 1 (creative)
    max_tokens= 1000, #You are gonna be charged by the amount of tokens!
    verbose = True #Allows easier debugging
    )

#Simple Prompt-response structure
response = llm.invoke("What day of the week is it today?") 
#print(response)

#2 prompts runned in Paralel
response2 = llm.batch(["Hello, how are you today?","Tell me a dad joke about a pizza"])
#print(response2)

#3 Streams a response (i.e. prints out as its beeing created) and does'nt need a print statement
response3 = llm.stream("Write a poem about organic life")
