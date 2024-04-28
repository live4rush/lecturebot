
from langchain.text_splitter import RecursiveCharacterTextSplitter
from summary import generate_summary
from topics import generate_topics
from load_docs import load_docs
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import pickle
import os
from langchain.chains import RetrievalQA
import langchain
from langchain.prompts import PromptTemplate
import time
import argparse
import langchain.schema


# Create argument parser
parser = argparse.ArgumentParser(
    description="Run a query on the document set.")
parser.add_argument("--query", "-q", type=str,
                    required=True, help="The question to ask.")

# Parse arguments
args = parser.parse_args()

# Extract query from parsed arguments
query = args.query


class Document:
    def __init__(self, metadata, content):
        self.metadata = metadata
        self.content = content


def load_pkl_files_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(directory_path, filename)
            doc = pkl_to_document(file_path)
            documents.append(doc)

    return documents


def pkl_to_document(pkl_file_path):
    with open(pkl_file_path, 'rb') as file:
        pkl_data = pickle.load(file)
        #print("printing langchain schema")
        #print(dir(langchain.schema.Document))
        if isinstance(pkl_data, Document) or isinstance(pkl_data, langchain.schema.Document):
            return pkl_data

        elif isinstance(pkl_data, dict):
            content = pkl_data.get('content', '')
            metadata = pkl_data.get('metadata', {})
            return Document(metadata, content)

        else:
            raise TypeError(
                f"Unsupported data type in {pkl_file_path}: {type(pkl_data)}")


# Define constants
TOKEN_LIMIT = 4096 - 512
n_gpu_layers = 20
n_batch = 256
n_ctx = 4096
n_threads = 36
max_tokens = 4096
MODEL_PATH = "models/llama-2-13b-chat.Q4_K_M.gguf"
LLM_PATH = "models/"
QDRANT_PATH = "qdrant/"

llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_threads=n_threads,
    f16_kv=True,
    n_ctx=n_ctx,
    max_tokens=max_tokens,
    temperature=0,
    verbose=True
)

documents = load_docs()
# We want to turn pkl docs to List<Document>
pickle_documents = load_pkl_files_from_directory('./ClassTranscriptions')

if len(pickle_documents) < len(documents):
    # Loop over documents and add metadata so we can search it later
    for index, doc in enumerate(documents):
        print(index, doc.metadata["source"])
        if os.path.exists(doc.metadata["source"].replace(".txt", ".pkl")):
            print("file exists, no need to create pkl file:",doc.metadata["source"].replace(".txt", ".pkl"))
        else:
            print("file doesnt exists, creating pkl file:", doc.metadata["source"].replace(".txt", ".pkl"))
            print("Generating summaries and topics for document #",
                  (index+1), "out of ", len(documents))
            doc.page_content = doc.page_content.replace('\n', '')
            doc.metadata["summary"] = generate_summary(llm, doc.page_content)
            print(doc.metadata["summary"])
            doc.metadata["topics"] = generate_topics(llm, doc.metadata["summary"])
            print(doc.metadata["topics"])
            with open(doc.metadata["source"].replace(".txt", ".pkl"), 'wb') as file:
                pickle.dump(doc, file)


print("Number of documents:", len(pickle_documents))
print("Loading Embeddings model")
# Embeddings model
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'mps'}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    #model_kwargs=model_kwargs,
)
print("Embeddings model loaded")


# Define the path to the file where you want to save the splits
splits_file_path = "cache/splits.pkl"

print("Checking for existing splits file...")
# Check if the splits file already exists
if os.path.exists(splits_file_path):
    print("Loading splits from file...")
    # Load splits from the file
    with open(splits_file_path, 'rb') as file:
        splits = pickle.load(file)
    print("Splits loaded from file.")
else:
    print("Splitting texts...")
    # Split texts
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(pickle_documents)
    print("Texts split.")

    print("Saving splits to file...")
    # Save splits to the file
    with open(splits_file_path, 'wb') as file:
        pickle.dump(splits, file)
    print("Splits saved to file.")


if os.path.exists('qdrant/collection/IK_Classes'):
    start_time = time.time()
    client = QdrantClient(path=QDRANT_PATH)
    collection_name = "IK_Classes"
    qdrant = Qdrant(client, collection_name, embeddings)
    end_time = time.time()
    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Loaded Vector Store in {elapsed_time} seconds")
else:
    # Populate the vector store with documents
    start_time = time.time()
    qdrant = Qdrant.from_documents(
        documents=splits,
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name="IK_Classes",
        force_recreate=False,
    )
    end_time = time.time()
    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Created Vector Store in {elapsed_time} seconds")

# initialize base retriever
retriever = qdrant.as_retriever(search_type="mmr", search_kwargs={
                                'k': 6, 'fetch_k': 50, 'lambda_mult': 0.30})

prompt_template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Based on the transcriptions from IK classes below, provide a short answer to the query to the best of your ability. Only use information from the IK classes to answer the query.
<</SYS>>

{context}

{question}

[/INST]"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever)

result = qa({"query": query})

print(result["result"])
