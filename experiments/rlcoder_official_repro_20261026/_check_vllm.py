
import sys
try:
    import vllm
    print('vllm imported successfully:', vllm.__version__)
except Exception as e:
    print('vllm import error:', type(e), e)
