rm -rf ~/.cache/atc/
rm -rf ~/.cache/ops_cache/
rm -rf ~/.cache/Python-SDK/

export ASCEND_LAUNCH_BLOCKING=1 # comment to accelerate
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

source /usr/local/Ascend/ascend-toolkit/set_env.sh

if [ -z "$ASCEND_RT_VISIBLE_DEVICES" ]; then
    DEVICE_COUNT=0
else
    # 将逗号替换为空格，转换为数组
    IFS=',' read -r -a device_array <<< "$ASCEND_RT_VISIBLE_DEVICES"
    DEVICE_COUNT=${#device_array[@]}
fi
OUTPUT_FILE="test.mp4"

# torchrun --nproc_per_node=$DEVICE_COUNT generate.py \
# --task i2v-A14B \
# --size 832*480 \
# --frame_num 1 \
# --ckpt_dir /mnt/nvme1p1/pengyt/wan2.2/Wan2.2-I2V-A14B \
# --image examples/i2v_input.JPG \
# --dit_fsdp \
# --t5_fsdp \
# --ulysses_size $DEVICE_COUNT \
# --offload_model True \
# --save_file $OUTPUT_FILE \
# --prof \
# --prompt "The cat down its head and catches a golden fish from the water."


torchrun --nproc_per_node=4 generate.py \
--task i2v-A14B \
--size 1280*720 \
--ckpt_dir /mnt/nvme1p1/pengyt/wan2.2/Wan2.2-I2V-A14B \
--lora_weight_dir /mnt/nvme1p1/pengyt/wan2.2/Wan2.2-I2V-A14B/lora_weight \
--image examples/girl.png \
--dit_fsdp \
--t5_fsdp \
--ulysses_size 4 \
--offload_model True \
--save_file $OUTPUT_FILE \
--prompt_file ./prompt.txt