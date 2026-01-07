import json
from rag_system import RAGSystem

class Evaluator:
    
    def __init__(self, model='gemma3:4b'):
        self.rag = RAGSystem(model=model)
    
    def score_answer(self, answer, keywords):
        answer_lower = answer.lower()
        
        refusal_phrases = ["don't have information", "not in the documentation"]
        refused = any(p in answer_lower for p in refusal_phrases)
        
        found = sum(1 for kw in keywords if kw.lower() in answer_lower)
        coverage = found / len(keywords)
        
        if refused:
            faithfulness = 5 if len(keywords) > 0 else 1
            relevance = 0
        else:
            faithfulness = 5 if coverage > 0.6 else (3 if coverage > 0.3 else 1)
            relevance = min(5, int(coverage * 6) + 1)
        
        return {
            'faithfulness': faithfulness,
            'relevance': relevance,
            'keyword_coverage': f"{found}/{len(keywords)}"
        }
    
    def run_evaluation(self, questions_file='questions.json'):
        with open(questions_file) as f:
            questions = json.load(f)
        
        results = []
        
        print("Running evaluation on golden questions\n")
        
        for q in questions:
            print(f"Question {q['id']}: {q['question']}")
            
            response = self.rag.query(q['question'])
            scores = self.score_answer(response['answer'], q['keywords'])
            
            results.append({
                **q,
                'answer': response['answer'],
                'sources': response['sources'],
                **scores
            })
            
            print(f"Faithfulness: {scores['faithfulness']}/5")
            print(f"Relevance: {scores['relevance']}/5")
            print(f"Coverage: {scores['keyword_coverage']}\n")
        
        avg_faith = sum(r['faithfulness'] for r in results) / len(results)
        avg_rel = sum(r['relevance'] for r in results) / len(results)
        
        print("\nEvaluation summary")
        print(f"Average faithfulness: {avg_faith:.2f}/5")
        print(f"Average relevance: {avg_rel:.2f}/5")
        
        with open('deliverables/evaluation_results.json', 'w') as f:
            json.dump({
                'questions': results,
                'summary': {
                    'avg_faithfulness': avg_faith,
                    'avg_relevance': avg_rel
                }
            }, f, indent=2)
        
        print("\nResults saved to deliverables/evaluation_results.json")
        return results


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.run_evaluation()