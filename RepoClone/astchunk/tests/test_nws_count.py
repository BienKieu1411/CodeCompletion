#!/usr/bin/env python3
"""
Quick test to verify that non whitespace count functions work correctly.
"""

from astchunk import (
    ByteRange,
    preprocess_nws_count,
    get_nws_count,
    get_nws_count_direct
)


def test_renamed_functions():    
    # Test data
    test_code = "def foo():\n    print('hello world')\n    return 42"
    test_bytes = test_code.encode('utf-8')
    
    print("Testing nws count functions...")
    print(f"----- Test code -----\n{test_code}\n---------------------")
    
    # Test preprocess_nws_count
    nws_cumsum = preprocess_nws_count(test_bytes)
    print(f"nws_cumsum shape: {nws_cumsum.shape}")
    print(f"nws_cumsum: {nws_cumsum}")
    
    # Test get_nws_count 
    full_range = ByteRange(0, len(test_bytes))
    nws_count = get_nws_count(nws_cumsum, full_range)
    print(f"Non-whitespace count (using cumsum): {nws_count}")
    
    # Test get_nws_count_direct 
    # Note: Direct method works on string, cumsum method works on bytes
    nws_count_direct = get_nws_count_direct(test_bytes.decode('utf-8'))
    print(f"Non-whitespace count (direct): {nws_count_direct}")
    
    # For this simple test, they should be the same
    print(f"Cumsum result: {nws_count}, Direct result: {nws_count_direct}")
    assert nws_count == nws_count_direct, "Cumsum and direct count do not match!"
    print("✓ Functions are working correctly!")
    
    # Test a partial range
    partial_range = ByteRange(0, 11)  # First 10 bytes (note exclusive end)
    partial_nws_count = get_nws_count(nws_cumsum, partial_range)
    partial_direct = get_nws_count_direct(test_bytes[:10].decode('utf-8'))
    print(f"Partial range [0:10] - cumsum: {partial_nws_count}, direct: {partial_direct}")
    assert partial_nws_count == partial_direct, "Partial range count does not match!"
    print("✓ Partial range test completed!")
    
    print("All tests passed! ✓")

if __name__ == "__main__":
    test_renamed_functions()
