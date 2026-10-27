
import os
print("bundle:", os.path.exists("/content/full_benchmark_8004_bundle.tar.gz"))
print("trace:", os.path.exists("/content/rlcoder_full_8004_retrieval_trace.jsonl.gz"))
try:
    import vllm
    print("setup_done: True")
except Exception:
    print("setup_done: False")
