import json
import os
import numpy as np
import faiss
from openai import OpenAI
import concurrent.futures
import configparser

# Load configuration from config.ini file
config = configparser.ConfigParser()
config.read('config.ini')
openai_api_base = config.get('API', 'openai_api_base')
openai_api_key = config.get('API', 'openai_api_key')


class RAGDatabase:
    """
    A simple Retrieval-Augmented Generation database that stores documents and their embeddings
    for retrieval when answering questions.
    """
    def __init__(
        self, 
        client, 
        db_file=None, 
        faiss_index_file=None,
        dimension=768
    ):
        self.client = client
        self.db_file = db_file  # File to store document texts
        self.faiss_index_file = faiss_index_file  # File to store vector index
        self.documents = []
        self.dimension = dimension  # OpenAI dimension is 1536, LLM Factory is currently 768
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # Faster FAISS search
        self.is_db_modified = False
        self.is_index_modified = False
        
        # Load existing data if available
        if self.db_file and os.path.exists(db_file):
            self.load_database()
        if self.faiss_index_file and os.path.exists(faiss_index_file):
            self.load_faiss_index()

    def embed_text(self, texts, model=''):
        '''Generate embeddings for one or multiple texts.'''
        if isinstance(texts, str):  
            texts = [texts]  # Ensure batch format
        
        # Call the OpenAI-compatible embeddings endpoint
        response = self.client.embeddings.create(
            input=texts, 
            model=model
        )
        return np.array([res.embedding for res in response.data], dtype=np.float32)

    def add_document(self, doc):
        '''Add a document or a list of documents and their embeddings to the database.'''

        # Ensure doc is a list
        if isinstance(doc, str):
            doc = [doc]  # Convert single document to a list

        # Generate embeddings for each document
        with concurrent.futures.ThreadPoolExecutor() as executor:
            embeddings = list(executor.map(self.embed_text, [[d] for d in doc]))

        # Flatten embeddings if `embed_text` returns a list for each input
        embeddings = np.array([e[0] for e in embeddings], dtype=np.float32)

        # Add embeddings to vector index
        self.index.add(embeddings)  

        # Store document texts
        self.documents.extend(doc)  

        # Flag modifications
        self.is_db_modified = True
        self.is_index_modified = True

        # Save updates if file paths are set
        if self.db_file:
            self.save_database()
        if self.faiss_index_file:
            self.save_faiss_index()

    def save_database(self):
        '''Save documents to a JSON file.'''
        if not self.db_file:
            print('Document database file is not defined')
            return
        if not self.is_db_modified:
            return  

        with open(self.db_file, 'w') as f:
            json.dump({'documents': self.documents}, f)

        self.is_db_modified = False  

    def load_database(self):
        '''Load documents from a JSON file.'''
        if not self.db_file:
            print('Document database file is not defined')
            return
        
        with open(self.db_file, 'r') as f:
            self.documents = json.load(f)['documents']

    def save_faiss_index(self):
        '''Save FAISS index to a file if modifications made.'''
        if not self.faiss_index_file:
            print('Index file is not defined')
            return
        if not self.is_index_modified:
            return  

        faiss.write_index(self.index, self.faiss_index_file)
        self.is_index_modified = False  

    def load_faiss_index(self):
        '''Load FAISS index safely, rebuild if corrupted.'''
        if not self.faiss_index_file:
            print('Index file is not defined')
            return
        if os.path.exists(self.faiss_index_file):
            self.index = faiss.read_index(self.faiss_index_file)
            # Verify index matches document count
            if self.index.ntotal != len(self.documents):  
                print("FAISS index does not match document count! Rebuilding index...")
                self.rebuild_faiss_index()
        else:
            self.rebuild_faiss_index()

    def rebuild_faiss_index(self):
        '''Rebuild FAISS index from stored documents.'''
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)  
        
        if not self.documents:
            return
        
        embeddings = self.embed_text(self.documents)  
        self.index.add(embeddings)  
        self.save_faiss_index()

    def search(self, query, top_k=3):
        '''Retrieve the most relevant documents using FAISS.'''
        # Convert query to embedding
        query_embedding = self.embed_text([query]).reshape(1, -1)
        
        # Search for similar embeddings in the index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Return the corresponding documents
        return [self.documents[i] for i in indices[0] if i < len(self.documents)]

    def generate_response(self, query, top_k=3, prompt_template=None):
        '''Retrieve relevant documents and generate a response using OpenAI.'''
        # Get relevant documents
        relevant_docs = self.search(query, top_k)
        context = '\n'.join(relevant_docs)

        # Format prompt with retrieved context
        if prompt_template:
            prompt = prompt_template.format(context=context, query=query)
        else:
            prompt = f'Using the following information, answer the question:\n{context}\n\nQuestion: {query}\nAnswer:'

        # Generate response from LLM
        response = self.client.chat.completions.create(
            model='',  # Add your LLM Factory adapter ID here
            messages=[
                {'role': 'system', 'content': 'You are a helpful AI.'},
                {'role': 'user', 'content': prompt}
            ]
        )

        if response.choices and response.choices[0].message:
            return response.choices[0].message.content
        else:
            return "Error: No response generated."


# Example Usage
if __name__ == '__main__':
    # Initialize OpenAI client
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Create the RAG database
    db = RAGDatabase(client, db_file="rag_docs.json", faiss_index_file="rag_index.faiss")
    
    # Add some sample documents
    db.add_document('The capital of France is Paris.')
    db.add_document('The largest planet in our solar system is Jupiter.')

    # Define a custom prompt template
    custom_prompt = "Based on this information:\n{context}\nProvide an answer to:\n{query}"
    
    # Test the RAG system with a query
    query = 'What is the capital of France?'
    print('Search results:', db.search(query))
    print('Generated response:', db.generate_response(query, prompt_template=custom_prompt))