import fitz
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

class DocumentProcessor:
    
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_pdf_text(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            doc.close()
            return text
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return ""
    
    def process_directory(self, directory):
        chunks_with_metadata = []
        pdf_files = list(Path(directory).rglob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {directory}")
            return []
        
        print(f"\nProcessing {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file.name}")
            text = self.extract_pdf_text(pdf_file)
            
            if text:
                chunks = self.splitter.split_text(text)
                
                for idx, chunk in enumerate(chunks):
                    chunks_with_metadata.append({
                        'content': chunk,
                        'source': pdf_file.name,
                        'chunk_id': idx,
                        'chunk_size': len(chunk)
                    })
        
        print(f"Created {len(chunks_with_metadata)} chunks")
        return chunks_with_metadata
    
    def get_stats(self, chunks):
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'total_documents': 0
            }
        
        chunk_lengths = [c['chunk_size'] for c in chunks]
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': round(sum(chunk_lengths) / len(chunk_lengths), 1),
            'min_chunk_size': min(chunk_lengths),
            'max_chunk_size': max(chunk_lengths),
            'total_documents': len(set(c['source'] for c in chunks))
        }


if __name__ == "__main__":
    print("\nDocument processing - chunking comparison test\n")
    
    configs = [
        {'size': 200, 'overlap': 20, 'name': 'Small'},
        {'size': 500, 'overlap': 50, 'name': 'Medium'},
        {'size': 1000, 'overlap': 100, 'name': 'Large'}
    ]
    
    comparison_results = []
    
    for config in configs:
        print(f"\nConfiguration: {config['name']} ({config['size']} characters)")
        
        processor = DocumentProcessor(
            chunk_size=config['size'],
            chunk_overlap=config['overlap']
        )
        
        chunks = processor.process_directory('knowledge_base')
        
        if not chunks:
            print("No chunks created")
            continue
        
        stats = processor.get_stats(chunks)
        
        result = {
            'config_name': config['name'],
            'chunk_size': config['size'],
            'chunk_overlap': config['overlap'],
            'stats': stats
        }
        
        comparison_results.append(result)
        
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Average size: {stats['avg_chunk_size']} characters")
    
    with open('deliverables/chunking_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print("\nResults saved to deliverables/chunking_comparison.json")