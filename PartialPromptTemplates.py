#Useful in case you have to wait for some variable to be calculated
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("{foo}{bar}")
partial_prompt = prompt.partial(foo="foo")
print(partial_prompt.format(bar="baz")) 
#Prints foobaz

#You can also just initialize the prompt with the partialed variables.

prompt = PromptTemplate(
    template="{foo}{bar}", input_variables=["bar"], partial_variables={"foo": "foo"}
)
print(prompt.format(bar="baz"))

## Partial with functions example
from datetime import datetime


def _get_datetime():
    now = datetime.now()
    return now.strftime("%m/%d/%Y, %H:%M:%S")

prompt = PromptTemplate(
    template="Tell me a {adjective} joke about the day {date}",
    input_variables=["adjective", "date"],
)
partial_prompt = prompt.partial(date=_get_datetime)
print(partial_prompt.format(adjective="funny"))

#You can also just initialize the prompt with the partialed variables, 
#which often makes more sense in this workflow.

prompt = PromptTemplate(
    template="Tell me a {adjective} joke about the day {date}",
    input_variables=["adjective"],
    partial_variables={"date": _get_datetime},
)
print(prompt.format(adjective="funny"))
