import pandas as pd
import plotly.express as px
import json
import os

def analyze_wan_ultimate_vis(jsonl_path="transfer_log.jsonl", filter_min_mb=1.0):
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found.")
        return

    with open(jsonl_path, 'r') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    
    # --- 1. 颜色与分类逻辑定义 ---
    # 定义一套高辨识度的配色方案
    COLOR_MAP = {
        "cpu->npu": "#FF5733",      # 鲜艳橙：数据加载到加速器
        "npu->cpu": "#3498DB",      # 亮蓝色：数据回传内存
        "npu_reformat": "#9B59B6",  # 紫色：NPU内部精度转换（显存抖动）
        "cpu_reformat": "#2ECC71",  # 绿色：CPU内部精度转换
        "other": "#95A5A6"          # 灰色：其他
    }

    def categorize(row):
        d = row['dir']
        if d in COLOR_MAP: return d
        if '->npu' in d: return "cpu->npu"
        if 'npu->' in d: return "npu->cpu"
        if 'reformat' in d: return "npu_reformat"
        return "other"

    df['category'] = df.apply(categorize, axis=1)

    # --- 2. 数据过滤与处理 ---
    df_plot = df[df['new_mb'] >= filter_min_mb].copy()
    df_plot['time'] = df_plot['time'] - df_plot['time'].min()
    
    # 让名字更短一点，方便显示
    df_plot['short_name'] = df_plot['name'].apply(lambda x: x[-30:] if len(x)>30 else x)

    # --- 3. 绘图 ---
    fig = px.scatter(
        df_plot,
        x="time",
        y="short_name",
        size="new_mb",
        color="category",  # 强制使用分类颜色
        symbol="category", # 形状也区分开：搬运点 vs 转换点
        hover_data=["dtype", "diff_mb", "old_mb", "new_mb"],
        title="Wan2.2 Heterogeneous Memory & Precision Profiler",
        color_discrete_map=COLOR_MAP,
        category_orders={"category": ["cpu->npu", "npu->cpu", "npu_reformat", "cpu_reformat"]},
        template="plotly_dark"
    )

    # --- 4. 样式调整 ---
    fig.update_traces(
        marker=dict(line=dict(width=0.5, color='white')),
        selector=dict(mode='markers')
    )

    fig.update_layout(
        height=900,
        xaxis_title="Time (seconds) - ⏳ 进度轴",
        yaxis_title="Tensor ID / Logic Name - 📂 张量标识",
        legend_title="Operation Type",
        # 增加右侧统计
        annotations=[dict(
            x=1.1, y=1.05, xref="paper", yref="paper",
            text=f"Total Events: {len(df)}<br>Filter: >{filter_min_mb}MB",
            showarrow=False, align="left"
        )]
    )

    fig.show()

    # --- 5. 打印分项总量报表 ---
    print("\n" + "统计报表".center(40, '='))
    summary = df.groupby(['category', 'dtype'])['new_mb'].agg(['sum', 'count'])
    summary['sum'] = (summary['sum'] / 1024).round(3) # GB
    summary.columns = ['Total_GB', 'Count']
    print(summary)
    print("="*44)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_wan_ultimate_vis(sys.argv[1], filter_min_mb=2.0)
    else:
        # 默认路径
        analyze_wan_ultimate_vis("transfer_log.jsonl", filter_min_mb=2.0)