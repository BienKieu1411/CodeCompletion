
# AlignCoder

**Models:** The trained retriever `AlignRetriever` is available at [AlignRetriever](https://huggingface.co/AlignCoder/AlignRetriever).

**Datasets:** The training and evaluation datasets is available at [Data4AlignCoder](https://huggingface.co/datasets/AlignCoder/Data4AlignCoder). Please download and then extract to the /data folder of this repository. You can use the following function to download the datasets:
```bash
def download_dataset():
   try:
       local_dir = snapshot_download(repo_id="AlignCoder/Data4AlignCoder", local_dir="data", repo_type="dataset")
       print(f"Dataset downloaded to: {local_dir}")
       return True
   except Exception as e:
       print(f"Download failed: {e}")
       return False
success = download_dataset_as_is()
if success:
   print("Download completed!")
```

**Set up:** To install these dependencies, use the following command:

```bash
conda create --name align python=3.10
conda activate align
pip install -r requirements.txt
```
## Training
To train UniXcoder to obtain AlignRetriever, use the following command:
```bash
# AlignCoder
python main.py \
    --weighted_keywords \
    --enable_generation \
    --enable_prediction \
    --add_api_blocks \
    --number_sample 4 \
    --inference_type unixcoder_with_rl \
    --output_dir result/online_train_sample_4 \
    --retriever_batch_size_per_gpu 1024 \
    --batch_size 6 \
    --epoch 20 \
    --sample_number 20 \
    --data_per_epoch 3000 \
    2>&1 | tee log_infer/online_train_sample_4.log
```

## RQ1
Use the following command for inference. Please specify the name of the generator you are using in `generator_model_path`.
```bash
python main.py \
    --eval \
    --weighted_keywords \
    --enable_prediction \
    --enable_generation \
    --add_api_blocks \
    --inference_type unixcoder_with_rl \
    --generator_model_path "" \
    --retriever_model_path "AlignCoder/AlignRetriever" \
    --generator_max_crossfile_length 1536 \
    --generator_max_context_length 2048 \
    --generator_batch_size_per_gpu 16 \
    --output_dir "result_infer/AlignCoder_deepseekcoder_1.3b_crossfile_1536_infile_512" \
    2>&1 | tee "log_infer/AlignCoder_deepseekcoder_1.3b_crossfile_1536_infile_512.log"
```

## RQ2
To run the following commands, please replace `x` with a value from 1 to 6:
```bash
python main.py \
    --weighted_keywords \
    --enable_generation \
    --enable_prediction \
    --add_api_blocks \
    --number_sample x \
    --inference_type unixcoder_with_rl \
    --output_dir result/online_train_sample_x \
    --retriever_batch_size_per_gpu 1024 \
    --batch_size 6 \
    --epoch 20 \
    --sample_number 20 \
    --data_per_epoch 3000 \
    2>&1 | tee log_infer/online_train_sample_x.log
```

## RQ3
```bash
# w/o DC
python main.py \
    --weighted_keywords \
    --enable_generation \
    --enable_prediction \
    --number_sample 4 \
    --inference_type unixcoder_with_rl \
    --output_dir result/w_o_DC \
    --retriever_batch_size_per_gpu 1024 \
    --batch_size 6 \
    --epoch 20 \
    --sample_number 20 \
    --data_per_epoch 3000 \
    2>&1 | tee log_infer/w_o_DC.log
```
```bash
# w/o QH
python main.py \
    --weighted_keywords \
    --enable_generation \
    --add_api_blocks \
    --inference_type unixcoder_with_rl \
    --output_dir result/w_o_QH \
    --retriever_batch_size_per_gpu 1024 \
    --batch_size 6 \
    --epoch 20 \
    --sample_number 20 \
    --data_per_epoch 3000 \
    2>&1 | tee log_infer/w_o_QH.log
```
```bash
# w/o RL
python main.py \
    --eval \
    --weighted_keywords \
    --enable_generation \
    --enable_prediction \
    --number_sample 4 \
    --inference_type unixcoder_with_rl \
    --generator_model_path "" \
    --retriever_model_path "microsoft/unixcoder-base" \
    --generator_max_crossfile_length 1536 \
    --generator_max_context_length 2048 \
    --generator_batch_size_per_gpu 16 \
    --output_dir "result_infer/w_o_RL" \
    2>&1 | tee log_infer/w_o_RL.log
```
## RQ4
```bash
python main.py \
    --eval \
    --weighted_keywords \
    --enable_generation \
    --enable_prediction \
    --add_api_blocks \
    --number_sample 4 \
    --temperature1 0.8 \
    --top_p1 0.7 \
    --inference_type unixcoder_with_rl \
    --generator_model_path "" \
    --retriever_model_path "AlignCoder/AlignRetriever" \
    --generator_max_crossfile_length 1536 \
    --generator_max_context_length 2048 \
    --generator_batch_size_per_gpu 16 \
    --output_dir "result_infer/temperature_0.8_top_p_0.7" \
    2>&1 | tee "log_infer/temperature_0.8_top_p_0.7.log"
```

