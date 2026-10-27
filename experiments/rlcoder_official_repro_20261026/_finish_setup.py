
import os, tarfile, json
from pathlib import Path
from huggingface_hub import snapshot_download

# 1. Extract bundles if not yet extracted
if Path('/content/cadre_official.tar.gz').is_file():
    print('Extracting cadre_official.tar.gz...')
    with tarfile.open('/content/cadre_official.tar.gz', 'r:gz') as tar:
        tar.extractall('/content')

if Path('/content/full_benchmark_8004_bundle.tar.gz').is_file():
    if not Path('/content/cceval_python.jsonl').is_file():
        print('Extracting full_benchmark_8004_bundle.tar.gz...')
        with tarfile.open('/content/full_benchmark_8004_bundle.tar.gz', 'r:gz') as tar:
            tar.extractall('/content')

# 2. Tokenizer download & patch
generator_dir = Path('/content/models/deepseek-coder-1.3b-base')
generator_dir.mkdir(parents=True, exist_ok=True)
print('Downloading tokenizer files...')
snapshot_download(
    'deepseek-ai/deepseek-coder-1.3b-base',
    local_dir=str(generator_dir),
    revision='c919139c3a9b4070729c8b2cca4847ab29ca8d94',
    allow_patterns=['*.json', '*.model', '*.txt']
)

cfg_path = generator_dir / 'tokenizer_config.json'
if cfg_path.is_file():
    with cfg_path.open('r', encoding='utf-8') as f:
        cfg = json.load(f)
    if cfg.get('tokenizer_class') != 'PreTrainedTokenizerFast':
        cfg['tokenizer_class'] = 'PreTrainedTokenizerFast'
        with cfg_path.open('w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)
        print('Patched tokenizer_config.json -> PreTrainedTokenizerFast')

print('SETUP COMPLETE!')
