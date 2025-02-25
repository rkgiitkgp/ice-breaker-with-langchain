import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain import hub
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from tools.tools import get_profile_url_tavily

load_dotenv()


def lookup(name: str) -> str:
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )
    template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                          Your answer should contain only a URL"""

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    # Tool is a class which invoked from langchain_core.tool interface. and this works as agent
    # which can search online. with that function "func" assigned.

    # this is array of tools. If there are multiple tools added. then agent decides which tool will be used for reasoning.
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page", # this must be meaningful, it is being used in reasoning for agent.
            func=get_profile_url_tavily, # function gets trigger
            description="useful for when you need get the Linkedin Page URL", # this is vvi, it is the determining factor for the llm to use exact tool.
        )
    ]

    # super popular prompt for ReAct prompting. This prompt is send to the llm. and langchain will take care of the rest.
    react_prompt = hub.pull("hwchase17/react")

    # create_react_agent is a built in function in langchain. which receives 1. llm, 2. tools, 3. react prompt
    # and this function will return an Agent based on ReAct algorithm
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    linked_profile_url = result["output"]
    return linked_profile_url


if __name__ == "__main__":
    linkedin_url = lookup(name="Rakesh Gupta | Sukshi | Senior Backend Developer")
    print(linkedin_url)
