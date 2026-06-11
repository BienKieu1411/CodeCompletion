#!/usr/bin/env python3
"""
Fixed chunking script for example source code.
"""

def chunkify(code: str, max_chunk_size: int) -> list[str]:
    """
    A simple baseline chunking method that divides code into chunks where each chunk is less than max_chunk_size lines.
    
    Args:
        code: The input code as a string
        max_chunk_size: Maximum number of lines per chunk
        
    Returns:
        List of code chunks as strings
    """
    lines = code.split('\n')
    chunks = []
    current_chunk = []
    
    for line in lines:
        # If adding this line would exceed the limit, start a new chunk
        if len(current_chunk) >= max_chunk_size:
            if current_chunk:  # Only add non-empty chunks
                chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
        else:
            current_chunk.append(line)
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks


def main():
    """Main function to process input file and create fixed chunks."""
    input_file = 'examples/source_code.txt'
    output_file = 'examples/outputs/fixed_chunking_results.txt'
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Set max chunk size (in lines)
    max_chunk_size = 50 
    
    # Create chunks
    chunks = chunkify(code, max_chunk_size)
    
    # Write results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Fixed Chunking Results (max {max_chunk_size} lines per chunk)\n")
        f.write("=" * 80 + "\n\n")
        
        for i, chunk in enumerate(chunks, 1):
            header = f"{'-' * 25} Chunk {i} ({len(chunk.split(chr(10)))} lines) {'-' * 25}\n"
            f.write(header)
            f.write(chunk)
            f.write("\n" + "-" * (len(header) - 1) + "\n\n")
    
    print(f"Fixed chunking completed!")
    print(f"Created {len(chunks)} chunks")
    print(f"Results written to: {output_file}")


if __name__ == "__main__":
    main()
