from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from third_parties.linkedin import scrape_linkedin_profile

api_key = "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"
# information = """
# Mohandas Karamchand Gandhi[c] (2 October 1869 – 30 January 1948)[2] was an Indian lawyer, anti-colonial nationalist, and political ethicist who employed nonviolent resistance to lead the successful campaign for India's independence from British rule. He inspired movements for civil rights and freedom across the world. The honorific Mahātmā (from Sanskrit, meaning great-souled, or venerable), first applied to him in South Africa in 1914, is now used throughout the world.[3]

# Born and raised in a Hindu family in coastal Gujarat, Gandhi trained in the law at the Inner Temple in London and was called to the bar at the age of 22. After two uncertain years in India, where he was unable to start a successful law practice, Gandhi moved to South Africa in 1893 to represent an Indian merchant in a lawsuit. He went on to live in South Africa for 21 years. Here, Gandhi raised a family and first employed nonviolent resistance in a campaign for civil rights. In 1915, aged 45, he returned to India and soon set about organising peasants, farmers, and urban labourers to protest against discrimination and excessive land tax.
# """

# commit 1:
# section 2: create a basic llm chain
# if __name__ == "__main__":
#     load_dotenv()

#     summary_template = """
#     given the information {information} about a person I want you to create:
#     1. a short summary
#     2. two interesting facts about them
#     """

#     # This method creates a template and also provide more feature where we can create template using different methodologies. 
#     # and also do couple of validation for us
#     summary_prompt_template = PromptTemplate(input_variables="information", template=summary_template)

#     # This method is from library "langchain-openai" which is created by OpenAI. 
#     # It support lots of method which can be used to use chat gpt models.
#     llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

#     # Chat with Ollama
#     # llm = ChatOllama(temperature=0, model="llama3.2")

#     # Effectively, this step creates a pipeline where input first goes through the prompt template, then into the LLM for response generation.
#     chain = summary_prompt_template | llm | StrOutputParser()

#     # Executes the chain and gets the model's response.
#     response = chain.invoke(input={"information": information})

#     print(response)


# commit 2:
# section 3.1: scrap data from linkedin
if __name__ == "__main__":
    load_dotenv()

    summary_template = """
    given the linkedin information {information} about a person I want you to create:
    1. a short summary
    2. two interesting facts about them
    """

    # This method creates a template and also provide more feature where we can create template using different methodologies. 
    # and also do couple of validation for us
    summary_prompt_template = PromptTemplate(input_variables="information", template=summary_template)

    # This method is from library "langchain-openai" which is created by OpenAI. 
    # It support lots of method which can be used to use chat gpt models.
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    # Chat with Ollama
    # llm = ChatOllama(temperature=0, model="llama3.2")

    # Effectively, this step creates a pipeline where input first goes through the prompt template, then into the LLM for response generation.
    chain = summary_prompt_template | llm | StrOutputParser()

    linkedin_data = scrape_linkedin_profile("www.linkedin.com/in/rakesh-gupta-iitkgp", True)

    # Executes the chain and gets the model's response.
    response = chain.invoke(input={"information": linkedin_data})

    print(response)