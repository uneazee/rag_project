from rag_system import RAGSystem

out_of_scope = [
    "What is Bitcoin?",
    "Who won the 2024 Super Bowl?",
    "Explain quantum computing"
]

print("Testing hallucination control\n")

rag = RAGSystem(model='phi3:3.8b')

for question in out_of_scope:
    print(f"Question: {question}")
    result = rag.query(question)
    print(f"Answer: {result['answer'][:200]}...")
    
    refused = "don't have" in result['answer'].lower() or \
              "not in the documentation" in result['answer'].lower()
    
    status = "passed (correctly refused)" if refused else "failed (hallucinated)"
    print(f"Status: {status}\n")