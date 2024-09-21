import os
import glob
import argparse

from typing import Dict, List, Tuple, Union

from langchain import hub
from langchain_community.chat_models import ChatOllama 
from langchain_community.document_loaders import ObsidianLoader 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain_core.callbacks import StreamingStdOutCallbackHandler

import gradio as gr


def get_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Get user file path")
    parser.add_argument('--notes_dir', help='Input file path')
    parser.add_argument('--filepath', help='Alternative input file path')
    parser.add_argument('--vectorize', default=False, action=argparse.BooleanOptionalAction, help='Whether to vectorize the file')
    return parser.parse_args() 

def format_docs(docs: List[str]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

def remove_all_files_in_folder(directory: str) -> None:
    os.system(f"rm -rf {directory}/*")

def main(question: str) -> str:
    args = get_args()
    notes_dir = args.notes_dir or args.filepath  # Use filepath if notes_dir is not provided
    
    if args.vectorize:
        loader: ObsidianLoader = ObsidianLoader(path=notes_dir)
        data: List[str] = loader.load()
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits: List[str] = text_splitter.split_documents(data)

        # Hard reset cause LLM be weird
        remove_all_files_in_folder("vectorstore")

        vectorstore: chroma.Chroma = chroma.Chroma.from_documents(documents=all_splits, embedding=OllamaEmbeddings(model='mistral'), persist_directory="vectorstore")
        print('Vectorized!')

    else:
        vectorstore: chroma.Chroma = chroma.Chroma(embedding_function=OllamaEmbeddings(model='mistral'), persist_directory="vectorstore")
        print('Loaded vectorstore!')

    rag_prompt: str = hub.pull("rlm/rag-prompt")
    # print('rag prompt loaded!', rag_prompt)
    llm = ChatOllama(model="mistral", callbacks=[StreamingStdOutCallbackHandler()])

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        chain_type_kwargs={"prompt": rag_prompt},
    )
    return qa_chain({"query": question})['result']



# TODO: seperate out loading function for local vectorstore
# demo = gr.Interface(fn=main, inputs="text", outputs="text")

if __name__ == "__main__":
    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        answer = main(question)
        print("\nAnswer:", answer)
        print("\n" + "-"*50 + "\n")
    # demo.launch(show_api=False)   
