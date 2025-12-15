export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m sglang_router.launch_router --pd-disaggregation \
	--prefill http://127.0.0.1:30000 \
	--decode http://127.0.0.1:30001 \
	--host 0.0.0.0 --port 8000