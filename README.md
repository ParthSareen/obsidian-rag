# obsidian-rag
Obsidian-Rag is a local first project that leverages Langchain to perform RAG on markdown files. It is particularly designed to work with the Obsidian note-taking app since I know we're all nerds (Waterloo strong 💪).

## Features
- Load markdown files from a given directory.
- Vectorize the loaded files for further processing.
- Perform similarity search on vectorized data.
- Utilize the Langchain library's functionalities such as `ChatOllama`, `ObsidianLoader`, `OllamaEmbeddings`, and `Chroma`.

## Dependencies
Must have installed `requirements.txt` and an `Ollama` instance with `Mistral` must be running

### Usage

The main script of the project is obsidian_rag.py. It accepts command-line arguments for the file path and a boolean flag to decide whether to vectorize the file or not.

`python obsidian_rag.py --filepath YOUR_FILE_PATH --vectorize`

This will open up a gradio interface which is still WIP and should be made into a chat interface.
