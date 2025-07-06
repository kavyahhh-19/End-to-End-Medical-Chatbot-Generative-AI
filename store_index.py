from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
print(f"Pinecone API Key loaded: {PINECONE_API_KEY is not None}")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"

print(f"Creating index: {index_name}")
try:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Index '{index_name}' created successfully!")
except Exception as e:
    print(f"Error creating index: {e}")

print("Listing indexes after creation:")
print(pc.list_indexes())

# Continue with embeddings
extracted_data = load_pdf_file(data='C:\\Users\\Kavya Vempati\\End-to-End-Medical-Chatbot-Generative-AI\\Data')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# embed each chunk and upsert
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
print("Vector store created and documents upserted successfully!")


