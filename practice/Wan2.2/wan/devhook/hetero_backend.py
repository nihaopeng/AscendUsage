import torch
from typing import List

def my_hetero_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("--- 成功拦截到 PyTorch 静态图 (FX Graph) ---")
    
    # 打印图的拓扑结构（这就是你需要的 DAG）
    gm.graph.print_tabular()
    
    # 遍历图中的所有节点（算子）
    for node in gm.graph.nodes:
        # node.op 表示操作类型，如 'call_function', 'placeholder', 'output'
        # node.target 表示具体的算子，如 aten.add.Tensor
        # node.meta 可以获取 Tensor 的 shape 和 dtype（需要配合 AOTAutograd）
        print(f"Node Name: {node.name}, Op: {torch.typename(node.target) if node.op == 'call_function' else node.op}")
        
    # 注意：这里需要返回一个可执行的 callable。
    # 原生默认是丢给 Inductor，你后面要替换成你自己的分发逻辑。
    return gm.forward

if __name__ == "__main__":
    from wan.modules.model import WanModel
    explanation = torch._dynamo.explain(WanModel, a, b)
    print(explanation.graph_print)