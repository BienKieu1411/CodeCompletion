export CUDA_VISIBLE_DEVICES=0

VLLM_CONFIGURE_LOGGING=0 \
    vllm serve deepseek-ai/deepseek-coder-1.3b-base