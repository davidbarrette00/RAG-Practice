import os

import openai
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

import constants

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

def main(query_text: str):
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=constants.CHROMA_PATH, embedding_function=embedding_function)

    #Search for relevant results
    results = db.similarity_search_with_relevance_scores(query=query_text, k=5)

    #Make sure we found relecant results
    if len(results) == 0 or results[0][1] < 0.5:
        print(f"Unable to find relevant results")
        if (len(results) > 0):
            print(f"Top relevancy: {results[0][1]}")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(constants.PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)



if __name__ == "__main__":
    print("What is your query?")
    main(input())