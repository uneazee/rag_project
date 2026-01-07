from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path

class VectorStore:
    """Vector database for semantic search"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}...")
        self.embedder = SentenceTransformer(model_name)
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        print("model loaded!")
    
    def add_documents(self, chunks):
        """Add documents to the vector store"""
        print(f"\nGenerating embeddings for {len(chunks)} chunks...")
        
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings in batches
        embeddings = self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        self.documents = chunks
        
        print(f"added {len(chunks)} documents to vector store")
        return len(chunks)
    
    def search(self, query, top_k=3):
        """Search for most relevant chunks"""
        # Generate query embedding
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        
        # Search FAISS index
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        # Format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append({
                    **self.documents[idx],
                    'distance': float(distance),
                    'similarity_score': float(1 / (1 + distance))
                })
        
        return results
    
    def save(self, path='vector_store'):
        """Save index and documents to disk"""
        Path(path).mkdir(exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}/index.faiss")
        
        # Save documents
        with open(f"{path}/documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        
        print(f"vector store saved to {path}/")
    
    def load(self, path='vector_store'):
        """Load index and documents from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}/index.faiss")
        
        # Load documents
        with open(f"{path}/documents.pkl", 'rb') as f:
            self.documents = pickle.load(f)
        
        print(f"loaded {len(self.documents)} documents from {path}/")


# Build the vector store
if __name__ == "__main__":
    from document_processor import DocumentProcessor
    
    print("building vector store")
    
    # Process documents with optimal chunk size (500)
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    chunks = processor.process_directory('knowledge_base')
    
    if not chunks:
        print("no documents found! Make sure PDFs are in knowledge_base/")
        exit(1)
    
    # Create and populate vector store
    vector_store = VectorStore()
    vector_store.add_documents(chunks)
    
    # Save for later use
    vector_store.save('vector_store')
    
    # Test semantic search
    print("\n testing semantic search")
    
    # Test queries relevant to your documents
    test_queries = [
        "What is a graph algorithm?",
        "Explain sorting algorithms",
        "What is augmented reality?"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        
        results = vector_store.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result['similarity_score']:.3f})")
            print(f"Source: {result['source']}")
            print(f"Content: {result['content'][:200]}...")
    print("vector store ready!")
  