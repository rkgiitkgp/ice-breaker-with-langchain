from typing import Tuple
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from output_parsers import Summary, summary_parser



from third_parties.linkedin import scrape_linkedin_profile
from third_parties.twitter import scrape_user_tweets_mock

api_key = "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"

def ice_break_with(name: str) -> Tuple[Summary, str]:
    linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username, mock=True)

    twitter_username = twitter_lookup_agent(name=name)
    tweets = scrape_user_tweets_mock(username=twitter_username)
    
    summary_template = """
    given the linkedin information {information}, and their latest posts {twitter_posts}
    about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them

    Use both information from twitter and linkedin
    \n {format_instruction}
    """

    # This method creates a template and also provide more feature where we can create template using different methodologies. 
    # and also do couple of validation for us
    summary_prompt_template = PromptTemplate(
        input_variables="information", 
        template=summary_template,
        partial_variables={"format_instruction": summary_parser.get_format_instructions}
        )

    # This method is from library "langchain-openai" which is created by OpenAI. 
    # It support lots of method which can be used to use chat gpt models.
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    # Chat with Ollama
    # llm = ChatOllama(temperature=0, model="llama3.2")

    # Effectively, this step creates a pipeline where input first goes through the prompt template, then into the LLM for response generation.
    # chain = summary_prompt_template | llm | StrOutputParser()

    # Here, output text from llm get into summary_parser pipeline. which format the output in json
    chain = summary_prompt_template | llm | summary_parser

    # Executes the chain and gets the model's response.
    response:Summary = chain.invoke(input={"information": linkedin_data, "twitter_posts": tweets})

    return response, linkedin_data.get("profile_pic_url")

if __name__ == "__main__":
    load_dotenv()

    # ice_break_with(name="Rakesh Gupta | Sukshi | Senior Backend Developer")
    ice_break_with(name="Eden Emarco")


