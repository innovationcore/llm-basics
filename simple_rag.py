import json
import os
import numpy as np
import faiss
from openai import OpenAI
import configparser

# PUBLIC SITE 
config = configparser.ConfigParser()
config.read('config.ini')
openai_api_base = config.get('API', 'openai_api_base')
openai_api_key = config.get('API', 'openai_api_key')


class RAGDatabase:
    def __init__(
        self, 
        client, 
        db_file='rag_db.json', 
        faiss_index_file='rag.index',
        dimension=768
    ):
        self.client = client
        self.db_file = db_file
        self.faiss_index_file = faiss_index_file
        self.documents = []
        self.dimension = dimension # OpenAI dimension is 1536, LLM Factory is currently 768
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # Faster FAISS search
        self.is_modified = False
        
        if os.path.exists(db_file):
            self.load_database()
        if os.path.exists(faiss_index_file):
            self.load_faiss_index()

    def embed_text(self, texts, model=''):
        '''Generate embeddings for one or multiple texts.'''
        if isinstance(texts, str):  
            texts = [texts]  # Ensure batch format
        
        response = self.client.embeddings.create(
            input=texts, 
            model=model
        )
        return np.array([res.embedding for res in response.data], dtype=np.float32)

    def add_document(self, doc):
        '''Add a document and its embedding to the database.'''
        embedding = self.embed_text([doc])  
        self.index.add(embedding) 
        self.documents.append(doc)
        self.is_modified = True  # Flag as modified
        self.save_database()
        self.save_faiss_index()

    def save_database(self):
        '''Save documents to a JSON file.'''
        if not self.is_modified:
            return  

        with open(self.db_file, 'w') as f:
            json.dump({'documents': self.documents}, f)

        self.is_modified = False  

    def load_database(self):
        '''Load documents from a JSON file.'''
        with open(self.db_file, 'r') as f:
            self.documents = json.load(f)['documents']

    def save_faiss_index(self):
        '''Save FAISS index to a file if modifications made.'''
        if not self.is_modified:
            return  

        faiss.write_index(self.index, self.faiss_index_file)
        self.is_modified = False  

    def load_faiss_index(self):
        '''Load FAISS index safely, rebuild if corrupted.'''
        if os.path.exists(self.faiss_index_file):
            self.index = faiss.read_index(self.faiss_index_file)
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
        query_embedding = self.embed_text([query]).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)  
        return [self.documents[i] for i in indices[0] if i < len(self.documents)]

    def generate_response(self, query, top_k=3, prompt_template=None):
    '''Retrieve relevant documents and generate a response using OpenAI.'''
    relevant_docs = self.search(query, top_k)
    context = '\n'.join(relevant_docs)

    if prompt_template:
        prompt = prompt_template.format(context=context, query=query)
    else:
        prompt = f'Using the following information, answer the question:\n{context}\n\nQuestion: {query}\nAnswer:'

    response = self.client.chat.completions.create(
        model='',
        messages=[{'role': 'system', 'content': 'You are a helpful AI.'},
                  {'role': 'user', 'content': prompt}]
    )

    if response.choices and response.choices[0].message:
        return response.choices[0].message.content
    else:
        return "Error: No response generated."


# Example Usage
if __name__ == '__main__':
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Create the RAG DB
    db = RAGDatabase(client)
    # Add documents to the DB
    db.add_document('The capital of France is Paris.')
    db.add_document('The largest planet in our solar system is Jupiter.')

    # Save database to local file and to FAISS index file
    # FAISS index is used to quickly lookup the vector relations
    # Found indexes are then matched with documents in database
    db.save_database()
    db.save_faiss_index()

    custom_prompt = "Based on this information:\n{context}\nProvide an answer to:\n{query}"
    
    query = 'What is the capital of France?'
    print('Search results:', db.search(query))
    print('Generated response:', db.generate_response(query, prompt_template=custom_prompt))
