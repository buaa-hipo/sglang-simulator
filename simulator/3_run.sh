export MODEL=/mnt/models/Qwen3-32B
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 8000 \
  --model $MODEL \
  --dataset-name mooncake \
  --random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.5 \
  --num-prompts 100 \
  --request-rate 10 \
  --max-concurrency 20 \
  --output-file sglang_random.jsonl --output-details \
  --profile --pd-separated --profile-prefill-url http://127.0.0.1:30000