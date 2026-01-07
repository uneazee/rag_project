import time
from rag_system import RAGSystem

test_question = "What is a sorting algorithm?"

print("Top-K performance analysis\n")

results = []

for k in [1, 3, 5, 10]:
    rag = RAGSystem(model='phi3:3.8b')
    
    start = time.time()
    response = rag.query(test_question, top_k=k)
    latency = time.time() - start
    
    results.append({
        'top_k': k,
        'latency': round(latency, 2),
        'answer_length': len(response['answer']),
        'sources': len(set(response['sources']))
    })
    
    print(f"Top-K={k}")
    print(f"  Latency: {latency:.2f}s")
    print(f"  Answer length: {len(response['answer'])} chars")
    print(f"  Unique sources: {len(set(response['sources']))}\n")

import json
with open('deliverables/topk_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Analysis saved to deliverables/topk_analysis.json")