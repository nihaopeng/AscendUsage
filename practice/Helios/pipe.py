import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from diffusers import AutoModel, HeliosPyramidPipeline
from diffusers.utils import export_to_video, load_video, load_image

vae = AutoModel.from_pretrained("BestWishYSH/Helios-Base", subfolder="vae", torch_dtype=torch.float32)

pipeline = HeliosPyramidPipeline.from_pretrained(
    "BestWishYsh/Helios-Base",
    vae=vae,
    torch_dtype=torch.bfloat16
)

print(pipeline.scheduler.ori_start_sigmas)
print(type(pipeline.scheduler.ori_start_sigmas))
print(pipeline.scheduler.config.stages)
# print(pipeline.stage_idx)

pipeline.to("npu", dtype=torch.bfloat16)

negative_prompt = """
Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality,
low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured,
misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
"""

# --- I2V ---
# i2v_prompt = """
# A towering emerald wave surges forward, its crest curling with raw power and energy. Sunlight glints off the translucent water, 
# illuminating the intricate textures and deep green hues within the wave’s body. A thick spray erupts from the breaking crest, 
# casting a misty veil that dances above the churning surface. As the perspective widens, the immense scale of the wave becomes 
# apparent, revealing the restless expanse of the ocean stretching beyond. The scene captures the ocean’s untamed beauty and 
# relentless force, with every droplet and ripple shimmering in the light. The dynamic motion and vivid colors evoke both awe and 
# respect for nature’s might.
# """
# image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/wave.jpg"

i2v_prompt = """
Wisps of violent steam surged rapidly from below, drifting upward before vanishing. The anime character's hair whipped wildly up and down in the turbulent airflow.
"""

image_path = "sddefault.jpg"

output = pipeline(
    prompt=i2v_prompt,
    negative_prompt=negative_prompt,
    image=load_image(image_path).resize((640, 384)),
    num_frames=240,
    pyramid_num_inference_steps_list=[6],
    guidance_scale=1.0,
    is_amplify_first_chunk=True,
    generator=torch.Generator("npu").manual_seed(42),
).frames[0]

print("video dtype:", output.dtype)
print("video min:", output.min())
print("video max:", output.max())
print("has nan:", torch.isnan(output).any())
print("has inf:", torch.isinf(output).any())

export_to_video(output, "helios_distilled_i2v_output.mp4", fps=24)