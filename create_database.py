import os
import shutil

import openai
from dotenv import load_dotenv

import constants

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

def main():
    save_to_chroma(split_documents(load_documents()))

def load_documents():
    loader = DirectoryLoader(constants.DATA_PATH, glob="**/*.txt")
    documents = loader.load()
    return documents


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True)
    chunks = text_splitter.split_documents(documents)

    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def save_to_chroma(chunks: list[Document]):

    if os.path.exists(constants.CHROMA_PATH):
        shutil.rmtree(constants.CHROMA_PATH)

    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(),
        persist_directory=constants.CHROMA_PATH
    )
    # db.persist()
    print(f"Saved {len(chunks)} chunks to \"./{constants.CHROMA_PATH}")


if __name__ == "__main__":
    main()
