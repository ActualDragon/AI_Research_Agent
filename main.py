from dotenv import load_dotenv #load environment variables
from pydantic import BaseModel #create a pydantic model to define fields as annotated attributes
#Langchain is an open source framework for developing apps and agents using LLMs.
from langchain_anthropic import ChatAnthropic #Import anthropic ai models
from langchain_core.prompts import ChatPromptTemplate #template to send a prompt to the model
from langchain_core.output_parsers import PydanticOutputParser #parse the model's output to a specific format
from langchain.agents import create_tool_calling_agent, AgentExecutor #create and execute our ai agent
from tools import search_tool, wiki_tool, save_tool #tools created for the agent to use when executing the query

load_dotenv() #load environment variables

class ResearchResponse(BaseModel): #Format for our response
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatAnthropic(model="claude-sonnet-4-5-20250929") #Import Claude 4.5
parser = PydanticOutputParser(pydantic_object=ResearchResponse) #Parse the output to match ResearchResponse

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a search assistant that will help generate an academic paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions} 
            """, #format instructions determined by the parser created before
        ),
        ("placeholder", "{chat_history}"), #provided by the agent executor
        ("human", "{query}"), #user input
        ("placeholder", "{agent_scratchpad}"), #provided by the agent executor
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool] #tools we created for the agent
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can i help you search? ")
raw_response = agent_executor.invoke({"query": query})

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)
