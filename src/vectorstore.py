from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_vectorstore(documents, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", persist_dir="embeddings"):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": "cpu"})

    # If FAISS index exists, load it
    if os.path.exists(persist_dir):
        print(f"Loading existing FAISS vectorstore from '{persist_dir}'...")
        
        vectorstore = FAISS.load_local(persist_dir,embeddings,allow_dangerous_deserialization=True)
        
        if documents:  # only add new documents if there are any
            print(f"Adding {len(documents)} new documents to the existing vectorstore...")
            vectorstore.add_documents(documents)
            vectorstore.save_local(persist_dir)
            print(f"Vectorstore updated and saved at '{persist_dir}'")
            
    else:
        print("Creating FAISS vectorstore... this may take a few minutes.")
        
        vectorstore = FAISS.from_documents(documents, embeddings)
        os.makedirs(persist_dir, exist_ok=True)
        vectorstore.save_local(persist_dir)
        
        print(f"Vectorstore created and saved at '{persist_dir}'")

    return vectorstore


def load_vectorstore(persist_dir="embeddings", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": "cpu"})
    
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vectorstore = FAISS.load_local(persist_dir,embeddings,allow_dangerous_deserialization=True)
        
        print(f"FAISS vectorstore loaded from '{persist_dir}'")
    
    else:
        print(f"No existing vectorstore found in '{persist_dir}'. Returning empty FAISS vectorstore.")
        # Return an empty FAISS vectorstore
        vectorstore = FAISS.from_texts([], embeddings)
    
    return vectorstore