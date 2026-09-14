#!/usr/bin/env python3
"""
AST chunking script for example source code.
Uses the ASTChunkBuilder class from src/astchunk/astchunk_builder.py with max_chunk_size = 2000.
"""

from astchunk import ASTChunkBuilder


def main():
    """Main function to process input file and create AST chunks."""
    input_file = 'examples/source_code.txt'
    output_file = 'examples/outputs/ast_chunking_results.txt'
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        code = f.read()

    configs = {
        "max_chunk_size": 1800,
        "language": "python",
        "metadata_template": "default",
        "chunk_expansion": False
    }
    
    # Initialize AST chunk builder
    chunk_builder = ASTChunkBuilder(**configs)

    # Create chunks using AST chunking
    chunks = chunk_builder.chunkify(code, **configs)
    
    # Write results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"AST Chunking Results (max {configs['max_chunk_size']} non-whitespace chars per chunk)\n")
        f.write("=" * 80 + "\n\n")
        
        for i, chunk in enumerate(chunks, 1):
            # Extract content and metadata
            content = chunk.get('content', chunk.get('context', ''))
            metadata = chunk.get('metadata', {})

            # Count lines in the chunk
            line_count = len(content.split('\n'))
            header = f"{'-' * 25} Chunk {i} ({line_count} lines / {metadata.get('chunk_size', 0)} chars) {'-' * 25}\n"
            f.write(header)
            f.write(content)
            f.write("\n" + "-" * (len(header) - 1) + "\n\n")
    
    print(f"AST chunking completed!")
    print(f"Created {len(chunks)} chunks")
    print(f"Results written to: {output_file}")

if __name__ == "__main__":
    main()
