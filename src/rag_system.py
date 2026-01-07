import ollama
from vector_store import VectorStore

class RAGSystem:
    
    def __init__(self, model='phi3:3.8b', vector_store_path='vector_store'):
        self.model = model
        self.vector_store = VectorStore()
        print(f"Loading vector store from {vector_store_path}")
        self.vector_store.load(vector_store_path)
        
        self.system_prompt = """You are a helpful assistant that answers questions based on provided documentation.

Rules:
1. Use only information from the context below
2. If the answer is not in the context, say "I don't have information about that in the documentation"
3. Cite the source document when answering
4. Be concise and accurate

Context:
{context}

Question: {question}

Answer:"""
    
    def query(self, question, top_k=3, verbose=False):
        
        results = self.vector_store.search(question, top_k)
        
        context = "\n\n---\n\n".join([
            f"[{r['source']}]\n{r['content']}" 
            for r in results
        ])
        
        if verbose:
            print(f"\nRetrieved {len(results)} chunks:")
            for r in results:
                print(f"  {r['source']} (score: {r['similarity_score']:.3f})")
        
        prompt = self.system_prompt.format(
            context=context,
            question=question
        )
        
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={'temperature': 0.1}
        )
        
        return {
            'answer': response['response'],
            'sources': [r['source'] for r in results]
        }


if __name__ == "__main__":
    rag = RAGSystem()
    
    print("RAG System Ready (type 'quit' to exit)\n")
    
    while True:
        question = input("Question: ")
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        result = rag.query(question, verbose=True)
        print(f"\nAnswer: {result['answer']}\n")
        print(f"Sources: {', '.join(set(result['sources']))}")