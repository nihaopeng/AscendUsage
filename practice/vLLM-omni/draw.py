import csv

SIZES = [
    "1024x1024",
    "768x1344",
    "1344x768",
    "896x1152",
    "1152x896",
]

CSV_FILE = "benchmark.csv"
records = []

with open("benchmark.csv", "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        records.append({
            "id": int(row["id"]),
            "round": int(row["round"]),
            "size": row["size"],
            "latency_ms": float(row["latency_ms"]),
        })

print(records[:3])

# =====================================
# 绘图
# =====================================

import matplotlib
import matplotlib.pyplot as plt
from statistics import mean

# 中文字体
matplotlib.rcParams["font.family"] = [
    "Noto Sans CJK JP",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

# -----------------------------
# Figure 1
# 每次请求Latency
# -----------------------------

request_ids = [r["id"] for r in records]
latencies = [r["latency_ms"] for r in records]

plt.figure(figsize=(10, 5))

colors = {
    "1024x1024": "tab:blue",
    "768x1344": "tab:orange",
    "1344x768": "tab:green",
    "896x1152": "tab:red",
    "1152x896": "tab:purple",
}

# 先画连接线
plt.plot(
    request_ids,
    latencies,
    color="gray",
    linewidth=1,
    alpha=0.5,
    zorder=1,
)


# 每个size单独画点
for size in SIZES:

    xs = [
        r["id"]
        for r in records
        if r["size"] == size
    ]

    ys = [
        r["latency_ms"]
        for r in records
        if r["size"] == size
    ]

    plt.scatter(
        xs,
        ys,
        s=80,
        color=colors[size],
        label=size,
        edgecolors="black",
        linewidths=0.5,
        zorder=2,
    )


plt.xlabel("Request ID")
plt.ylabel("Latency (ms)")
plt.title("Latency of Sequential Requests")

plt.grid(
    linestyle="--",
    alpha=0.3
)

plt.legend(
    title="Image Size",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    "latency_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()