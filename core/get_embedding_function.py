from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

URL = os.getenv("OLLAMA_URL")

def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(base_url=URL,model="nomic-embed-text")
    return embeddings
