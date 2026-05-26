
import importlib
from pathlib import Path

from .registry import RuntimeRegistry

_pkg_path = Path(__file__).resolve().parent
for py in _pkg_path.glob("wf_*.py"):
    mod_name = f"{__name__}.{py.stem}"
    importlib.import_module(mod_name)

def get_workflow(name: str):
    """
    根据工作流名称获取 create_pipeline_graph 工厂方法。

    Args:
        name (str): 工作流名称（注册名）

    Returns:
        Callable: 用于构建该工作流图的工厂函数
    """
    return RuntimeRegistry.get(name)

async def run_workflow(name: str, state):
    factory = get_workflow(name)
    graph_builder = factory()

    graph = graph_builder.build()       

    return await graph.ainvoke(state)

list_workflows = RuntimeRegistry.all