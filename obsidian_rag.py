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
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def get_args() -> str:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Get user file path")
    parser.add_argument('--notes_dir', help='Input file path')
    parser.add_argument('--vectorize', default=False, action=argparse.BooleanOptionalAction, help='Whether to vectorize the file')
    return parser.parse_args() 

def load_markdown_files(directory: str) -> dict:
    files_dict: dict = {}
    for filename in glob.glob(os.path.join(directory, '*.md')):
        with open(filename, 'r') as f:
            files_dict[os.path.basename(filename)] = f.read()
    return files_dict

def format_docs(docs: list) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

def remove_all_files_in_folder(directory: str) -> None:
    os.system(f"rm -rf {directory}/*")

def main():
    callbacks = [StreamingStdOutCallbackHandler()]
    args = get_args()
    if args.vectorize:
        loader: ObsidianLoader = ObsidianLoader(path=args.notes_dir)
        data: list = loader.load()
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits: list = text_splitter.split_documents(data)

        # Hard reset cause LLM be weird
        remove_all_files_in_folder("vectorstore")
        # os.mkdir("vectorstore")

        vectorstore: chroma.Chroma = chroma.Chroma.from_documents(documents=all_splits, embedding=OllamaEmbeddings(model='mistral'), persist_directory="vectorstore")
        print('Vectorized!')

    else:
        vectorstore: chroma.Chroma = chroma.Chroma(embedding_function=OllamaEmbeddings(model='mistral'), persist_directory="vectorstore")
        print('Loaded vectorstore!')

    print(vectorstore)

    rag_prompt: str = hub.pull("rlm/rag-prompt")
    print(rag_prompt)

    # question: str = "explain cicd?"
    # docs: list = vectorstore.similarity_search(question)

    # Document Q&A
    llm = ChatOllama(model="mistral", callbacks=callbacks)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        # retriever=vectorstore.as_retriever(
        #     search_type="similarity_score_threshold",
        #     search_kwargs={'score_threshold': 0.8}
        # ),
        chain_type_kwargs={"prompt": rag_prompt},
    )
    while True:
        question = input("Ask a question: ")
        ans = qa_chain({"query": question})
        print(ans)
    


if __name__ == "__main__":
    main()
