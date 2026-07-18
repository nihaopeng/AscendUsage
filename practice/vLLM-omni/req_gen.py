from openai import OpenAI
import base64
import time


# vLLM-Omni FLUX 服务地址
client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:59085/v1"
)


prompt = """
A futuristic city at night,
cyberpunk style,
neon lights,
high quality,
cinematic lighting,
8k resolution
"""


print("Sending request...")

start = time.time()

response = client.images.generate(
    model="black-forest-labs/FLUX.1-dev",
    prompt=prompt,
    size="1024x1024",
    n=1,
    response_format="b64_json"
)


print(f"Generation finished: {time.time()-start:.2f}s")


# 获取 base64 图片
image_base64 = response.data[0].b64_json


# 解码
image_bytes = base64.b64decode(image_base64)


# 保存
output_file = "gen_res/flux_output.png"

with open(output_file, "wb") as f:
    f.write(image_bytes)


print(f"Saved: {output_file}")
