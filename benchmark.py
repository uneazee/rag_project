import ollama
import time
import psutil
import os
import json

def benchmark_model(model_name):
    """Benchmark a model for TTFT and RAM usage"""
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Testing: {model_name}")
    
    # Test with a technical question
    start = time.time()
    response = ollama.generate(
        model=model_name,
        prompt="Explain exception handling in Python with examples."
    )
    ttft = time.time() - start
    
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    result = {
        'model': model_name,
        'ttft_seconds': round(ttft, 2),
        'ram_mb': round(mem_after - mem_before, 2),
        'response_length': len(response['response']),
        'response_preview': response['response'][:150] + "..."
    }
    
    print(f"Time to First Token: {result['ttft_seconds']}s")
    print(f"RAM Usage: {result['ram_mb']}MB")
    print(f"Response Length: {result['response_length']} characters")
    print(f"Preview: {result['response_preview']}")
    
    return result

# Main execution
if __name__ == "__main__":
    print("model benchmarking")
    
    models = ['llama3:8b', 'phi3']
    results = []
    
    for model in models:
        try:
            result = benchmark_model(model)
            results.append(result)
        except Exception as e:
            print(f"error testing {model}: {e}")
    
    # Save results
    with open('deliverables/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary table
    print("summary")
    print(f"{'Model':<15} {'TTFT (s)':<12} {'RAM (MB)':<12}")
    for r in results:
        print(f"{r['model']:<15} {r['ttft_seconds']:<12} {r['ram_mb']:<12}")
    
    print("\n results saved to: deliverables/benchmark_results.json")