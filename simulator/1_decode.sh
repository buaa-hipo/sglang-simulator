export MODEL=/mnt/models/Qwen3-32B
export ASCEND_MF_STORE_URL="tcp://127.0.0.0:26000"
export ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE=true
export ASCEND_NPU_PHY_ID=0
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m sglang.launch_server --model-path $MODEL \
    --disaggregation-mode decode \
    --port 30001 \
    --base-gpu-id 4 \
    --tp-size 4 \
    --mem-fraction-static 0.9 \
    --disable-cuda-graph \
    --disaggregation-transfer-backend ascend