# obsidian-rag
# NOT MAINTAINED!
For the maintained version of the app, see [RAG-in-a-box](https://github.com/ParthSareen/RAG-in-a-box)

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
<img width="688" alt="image" src="https://github.com/ParthSareen/obsidian-rag/assets/29360864/13747e0b-78f8-495e-9f03-c80229d537a6">
<img width="1256" alt="image" src="https://github.com/ParthSareen/obsidian-rag/assets/29360864/f79e90e3-2624-46a9-90e0-12034c9afb42">

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=parthsareen/obsidian-rag&type=Date)](https://star-history.com/#parthsareen/obsidian-rag&Date)
