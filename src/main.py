import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def load_document(file):
    if file.endswith(".txt"):
        loader = TextLoader(file, encoding='UTF-8')
    elif file.endswith(".pdf"):
        loader = PyPDFLoader(file)
    elif file.endswith(".json"):
        loader = JSONLoader(file, jq_schema='.', text_content=False)
    else:
        raise Exception("Unknown file extension")
    document = loader.load()
    return document
    

def get_vector_storage():
    cwd = os.getcwd()
    file_path = 'src/data'

    documents = []

    for file in os.listdir(os.path.join(cwd, file_path)):
        doc = load_document(os.path.join(cwd, file_path, file))
        documents.extend(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(documents)

    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

    return vectorstore

def main():
    storage = get_vector_storage()
    prompt = "bee"
    docs = storage.similarity_search(prompt)
    print(docs)


if __name__ == "__main__":
    main()
