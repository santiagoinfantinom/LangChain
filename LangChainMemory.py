import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.memory import ChatMessageHistory
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
        model = "gpt-3.5-turbo-1106",
        temperature = 0.4,
        base_url="https://polite-bags-shave.loca.lt"
)

history = ChatMessageHistory()

history.add_user_message("hi!")
history.add_ai_message("hello my friend!")
history.messages

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("hello my friend!")
memory.load_memory_variables({})

from langchain.llms import OpenAI
from langchain.chains import ConversationChain #Important for using Memory!

llm = OpenAI(temperature=0)
conversation = ConversationChain(
    llm=llm, verbose=True, memory=ConversationBufferMemory()
)
conversation.predict(input="Hi") #Creates a conversation

conversation.predict(input="I need to know the capital of france") #Adds the new question to the conversation

from langchain.memory import ConversationSummaryBufferMemory

review = "I ordered Pizza Salami for 9.99$ and it was awesome! \
The pizza was delivered on time and was still hot when I received it. \
The crust was thin and crispy, and the toppings were fresh and flavorful. \
The Salami was well-cooked and complemented the cheese perfectly. \
The price was reasonable and I believe I got my money's worth. \
Overall, I am very satisfied with my order and I would recommend this pizza place to others."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
#Use to reduce the number of tokens used
memory.save_context(
    {"input": "Hello, how can I help you today?"},
    {"output": "Could you analyze a review for me?"},
)
memory.save_context(
    {"input": "Sure, I'd be happy to. Could you provide the review?"},
    {"output": f"{review}"},
)
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

conversation.predict(input="Thank you very much!")

#This prints out a conversation between a human and an AI about the pizza review!

memory.load_memory_variables({})


