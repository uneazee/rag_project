import json
import time
from rag_system import RAGSystem

class ModelComparator:
    """Compare multiple LLM models for RAG performance"""
    
    def __init__(self):
        self.models = ['phi3:3.8b', 'llama3:8b', 'gemma3:4b']
        self.results = {}
    
    def score_answer(self, answer, keywords):
        """Score an answer for faithfulness and relevance"""
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
    
    def evaluate_model(self, model_name, questions_file='questions.json'):
        """Evaluate a single model and track timing"""
        print(f"evaluating: {model_name}")
        
        # Load questions
        with open(questions_file) as f:
            questions = json.load(f)
        
        # Initialize RAG system
        rag = RAGSystem(model=model_name)
        
        results = []
        question_times = []
        
        # Track total time
        total_start = time.time()
        
        for q in questions:
            print(f"Question {q['id']}: {q['question']}")
            
            # Track time for this question
            q_start = time.time()
            response = rag.query(q['question'])
            q_time = time.time() - q_start
            question_times.append(q_time)
            
            scores = self.score_answer(response['answer'], q['keywords'])
            
            results.append({
                **q,
                'answer': response['answer'],
                'sources': response['sources'],
                'time_seconds': round(q_time, 2),
                **scores
            })
            
            print(f"  Faithfulness: {scores['faithfulness']}/5")
            print(f"  Relevance: {scores['relevance']}/5")
            print(f"  Coverage: {scores['keyword_coverage']}")
            print(f"  Time: {q_time:.2f}s\n")
        
        total_time = time.time() - total_start
        
        # Calculate averages
        avg_faith = sum(r['faithfulness'] for r in results) / len(results)
        avg_rel = sum(r['relevance'] for r in results) / len(results)
        avg_time = sum(question_times) / len(question_times)
        
        summary = {
            'model': model_name,
            'avg_faithfulness': round(avg_faith, 2),
            'avg_relevance': round(avg_rel, 2),
            'total_time_seconds': round(total_time, 2),
            'avg_time_per_question': round(avg_time, 2),
            'min_time': round(min(question_times), 2),
            'max_time': round(max(question_times), 2),
            'total_questions': len(questions)
        }


        print(f"summary for {model_name}:")
        
        print(f"Average Faithfulness: {avg_faith:.2f}/5 ({avg_faith*20:.0f}%)")
        print(f"Average Relevance: {avg_rel:.2f}/5 ({avg_rel*20:.0f}%)")
        print(f"Total Time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        print(f"Average Time/Question: {avg_time:.2f}s")
        print(f"Min Time: {min(question_times):.2f}s")
        print(f"Max Time: {max(question_times):.2f}s")
        
        # Save model-specific results
        output_data = {
            'model': model_name,
            'questions': results,
            'summary': summary
        }
        
        # Save with model name in filename
        model_filename = model_name.replace(':', '_').replace('.', '_')
        output_file = f'deliverables/evaluation_{model_filename}.json'
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to: {output_file}\n")
        
        return output_data
    
    def run_comparison(self, questions_file='questions.json'):
        """Run evaluation on all models"""
        
        print("model comparison started\n")

        
        comparison_start = time.time()
        all_results = {}
        
        for model in self.models:
            try:
                result = self.evaluate_model(model, questions_file)
                all_results[model] = result
            except Exception as e:
                print(f"\nERROR testing {model}: {e}\n")
                continue
        
        total_comparison_time = time.time() - comparison_start
        
        # Generate comparison summary
        
        print("final comparison summary:\n")
        
        
        # Create comparison table
        print(f"{'Model':<15} {'Faithfulness':<15} {'Relevance':<15} {'Avg Time':<12} {'Total Time':<12}")
        print("-" * 80)
        
        for model in self.models:
            if model in all_results:
                summary = all_results[model]['summary']
                print(f"{model:<15} "
                      f"{summary['avg_faithfulness']:.2f}/5 ({summary['avg_faithfulness']*20:.0f}%){'':<4} "
                      f"{summary['avg_relevance']:.2f}/5 ({summary['avg_relevance']*20:.0f}%){'':<4} "
                      f"{summary['avg_time_per_question']:.2f}s{'':<6} "
                      f"{summary['total_time_seconds']:.2f}s")
        
        
        print(f"Total comparison time: {total_comparison_time:.2f}s ({total_comparison_time/60:.1f} minutes)")
        
        
        # Save consolidated comparison
        comparison_output = {
            'models_compared': self.models,
            'total_comparison_time': round(total_comparison_time, 2),
            'results': all_results,
            'ranking': self.rank_models(all_results)
        }
        
        with open('deliverables/model_comparison_complete.json', 'w') as f:
            json.dump(comparison_output, f, indent=2)
        
        print("complete comparison saved to: deliverables/model_comparison_complete.json\n")
        
        # Print ranking
        print("RANKING:")
        for i, rank_info in enumerate(comparison_output['ranking'], 1):
            medal = ['🥇', '🥈', '🥉'][i-1] if i <= 3 else f'{i}.'
            print(f"{medal} {rank_info['model']}: Combined Score = {rank_info['combined_score']:.2f}/5")
        
        return comparison_output
    
    def rank_models(self, results):
        """Rank models by combined score"""
        rankings = []
        
        for model, data in results.items():
            summary = data['summary']
            combined = (summary['avg_faithfulness'] + summary['avg_relevance']) / 2
            rankings.append({
                'model': model,
                'faithfulness': summary['avg_faithfulness'],
                'relevance': summary['avg_relevance'],
                'combined_score': round(combined, 2)
            })
        
        # Sort by combined score (descending)
        rankings.sort(key=lambda x: x['combined_score'], reverse=True)
        return rankings


if __name__ == "__main__":
    comparator = ModelComparator()
    comparator.run_comparison()
