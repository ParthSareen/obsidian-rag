import os
import glob

import argparse

from langchain import hub
from langchain.chat_models import ChatOllama 
from langchain.document_loaders import ObsidianLoader 
from langchain.embeddings import OllamaEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import chroma
from langchain.chains.question_answering import load_qa_chain


def get_args() -> str:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Get user file path")
    parser.add_argument('--filepath', help='Input file path')
    parser.add_argument('--vectorize', default=False, help='Whether to vectorize the file')
    return parser.parse_args() 

def load_markdown_files(directory: str) -> dict:
    files_dict: dict = {}
    for filename in glob.glob(os.path.join(directory, '*.md')):
        with open(filename, 'r') as f:
            files_dict[os.path.basename(filename)] = f.read()
    return files_dict

def format_docs(docs: list) -> str:
        return "\n\n".join(doc.page_content for doc in docs)


def main():

    args = get_args()
    loader: ObsidianLoader = ObsidianLoader(path=args.filepath)
    data: list = loader.load()
    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits: list = text_splitter.split_documents(data)
   
    # TODO: add a way to save the vectorstore 

    vectorstore: chroma.Chroma = chroma.Chroma.from_documents(documents=all_splits, embedding=OllamaEmbeddings(model='mistral'), persist_directory="vectorstore")
    vectorstore = vectorstore.load()

    vectorstore.persist()


    rag_prompt: str = hub.pull("rlm/rag-prompt")

    question: str = "explain cicd?"
    docs: list = vectorstore.similarity_search(question)
    print(len(docs))
    llm = ChatOllama(model="mistral")
    from langchain.chains import RetrievalQA

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": rag_prompt},
    )
    ans = qa_chain({"query": question})
    print(ans)
    


if __name__ == "__main__":
    main()
