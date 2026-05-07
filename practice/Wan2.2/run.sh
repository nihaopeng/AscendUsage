rm -rf ~/.cache/atc/
rm -rf ~/.cache/ops_cache/
rm -rf ~/.cache/Python-SDK/

export ASCEND_LAUNCH_BLOCKING=1 # comment to accelerate
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

source /usr/local/Ascend/ascend-toolkit/set_env.sh

torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir /mnt/nvme1p1/pengyt/wan2.2/Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --offload_model True --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

# ASCEND_RT_VISIBLE_DEVICES=4 python generate.py --task i2v-A14B --size 832*480 --frame_num 121 --ckpt_dir /mnt/nvme1p1/pengyt/wan2.2/Wan2.2-I2V-A14B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

# ASCEND_RT_VISIBLE_DEVICES=0 python generate.py --task ti2v-5B --size 704*1280 --frame_num 25 --ckpt_dir /mnt/nvme1p1/pengyt/wan2.2/Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/girl.jpg