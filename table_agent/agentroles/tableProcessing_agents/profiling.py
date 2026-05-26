from __future__ import annotations

from typing import Any, Dict, Optional

from table_agent.states.tableState import TableState
from table_agent.toolkits.tool_manager import ToolManager
from table_agent.utils.logger import get_logger
from table_agent.agentroles.cores.base_agent import BaseAgent
from table_agent.agentroles.cores.registry import register

log = get_logger(__name__)

# ----------------------------------------------------------------------
# Agent Definition
# ----------------------------------------------------------------------
@register("data_profiling")
class DataProfiling(BaseAgent):
    """TODO: 描述 data_profiling 的职责"""

    # ---------- 工厂 ----------
    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    # ---------- 基本配置 ----------
    @property
    def role_name(self) -> str:  # noqa: D401
        return "data_profiling"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_data_profiling"
    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_data_profiling"

    # ---------- Prompt 参数 ----------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """根据前置工具结果构造 prompt 参数
        提示词中的占位符：
        """
        # TODO: 按需补充
        return {
            "raw_table_paths": pre_tool_results.get("raw_table_paths", []),
            "operation": pre_tool_results.get("operation", ""),
            'user_refine_input': pre_tool_results.get('user_refine_input', ''),
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """若调用方未显式传入，返回默认前置工具结果"""
        return {}

    # ---------- 结果写回 ----------
    def update_state_result(
        self,
        state: TableState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """将推理结果写回 TableState"""
        state['data_profiling'] = result['answer']
        state['execution_time'] = result['execution_time']
        # TODO：关注dfa中messages会做什么
        super().update_state_result(state, result, pre_tool_results)

async def data_profiling(
    state: TableState, 
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    use_agent: bool = False,
    **kwargs,
) -> TableState:
    """数据分析的入口函数"""
    generator = DataProfiling(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await generator.execute(state, use_agent=use_agent, **kwargs)


def create_data_profiling_agent(tool_manager: Optional[ToolManager] = None, **kwargs) -> DataProfiling:
    """创建数据分析代理实例"""
    return DataProfiling(tool_manager=tool_manager, **kwargs)