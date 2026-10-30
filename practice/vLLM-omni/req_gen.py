from openai import OpenAI
import base64
import csv
import os
import time
from statistics import mean, median, stdev

# ==============================
# 配置
# ==============================

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:59085/v1"
)

MODEL = "black-forest-labs/FLUX.1-dev"

PROMPT = """
A futuristic city at night,
cyberpunk style,
neon lights,
high quality,
cinematic lighting,
8k resolution
"""

# 根据你的服务支持情况修改
SIZES = [
    "1024x1024",
    "768x1344",
    "1344x768",
    "896x1152",
    "1152x896",
]

# 每个 Shape 测试次数
REPEAT_PER_SIZE = 6

SAVE_IMAGE = False          # 是否保存图片
OUTPUT_DIR = "gen_res"
CSV_FILE = "benchmark.csv"

# ==============================
# 初始化
# ==============================

if SAVE_IMAGE:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

records = []

total_requests = len(SIZES) * REPEAT_PER_SIZE

print("=" * 70)
print(f"Start Benchmark ({total_requests} Requests)")
print("=" * 70)

# ==============================
# Benchmark
# ==============================

req_id = 0

for r in range(REPEAT_PER_SIZE):

    print(f"\n========== Round {r+1}/{REPEAT_PER_SIZE} ==========")

    for size in SIZES:

        req_id += 1

        start = time.perf_counter()

        response = client.images.generate(
            model=MODEL,
            prompt=PROMPT,
            size=size,
            n=1,
            response_format="b64_json"
        )

        latency = (time.perf_counter() - start) * 1000

        records.append({
            "id": req_id,
            "round": r + 1,
            "size": size,
            "latency_ms": latency,
        })

        print(
            f"[{req_id:02d}/{total_requests}] "
            f"{size:<10} "
            f"{latency:8.2f} ms"
        )

        if SAVE_IMAGE:
            image_bytes = base64.b64decode(response.data[0].b64_json)

            filename = os.path.join(
                OUTPUT_DIR,
                f"{req_id:02d}_{size.replace('x','_')}.png"
            )

            with open(filename, "wb") as f:
                f.write(image_bytes)

# ==============================
# 保存CSV
# ==============================

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "round", "size", "latency_ms"])

    for r in records:
        writer.writerow([
            r["id"],
            r["round"],
            r["size"],
            f"{r['latency_ms']:.3f}"
        ])

# ==============================
# Overall Statistics
# ==============================

latencies = [r["latency_ms"] for r in records]

lat_sorted = sorted(latencies)

p50 = lat_sorted[int(len(lat_sorted) * 0.5)]
p95 = lat_sorted[int(len(lat_sorted) * 0.95)]

print("\n")
print("=" * 70)
print("Overall Statistics")
print("=" * 70)

print(f"Requests : {len(latencies)}")
print(f"Average  : {mean(latencies):.2f} ms")
print(f"Median   : {median(latencies):.2f} ms")
print(f"P50      : {p50:.2f} ms")
print(f"P95      : {p95:.2f} ms")
print(f"Std      : {stdev(latencies):.2f} ms")
print(f"Min      : {min(latencies):.2f} ms")
print(f"Max      : {max(latencies):.2f} ms")

# ==============================
# Statistics By Shape
# ==============================

print("\n")
print("=" * 70)
print("Statistics By Size")
print("=" * 70)

for size in SIZES:

    xs = [r["latency_ms"] for r in records if r["size"] == size]

    print(
        f"{size:<10}"
        f" Avg={mean(xs):8.2f} ms"
        f"  Min={min(xs):8.2f}"
        f"  Max={max(xs):8.2f}"
    )

print("\nCSV saved to:", CSV_FILE)

# =====================================
# 保存 CSV
# =====================================