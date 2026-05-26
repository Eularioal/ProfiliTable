from __future__ import annotations

from typing import Any, Dict, Optional

from table_agent.states.tableState import TableState
from table_agent.toolkits.tool_manager import ToolManager
from table_agent.utils.logger import get_logger
from table_agent.agentroles.cores.base_agent import BaseAgent
from table_agent.agentroles.cores.registry import register
from table_agent.utils.node_helpers import extract_python_code_block

log = get_logger(__name__)

# ----------------------------------------------------------------------
# Agent Definition
# ----------------------------------------------------------------------
@register("summarizer")
class Summarizer(BaseAgent):
    """TODO: 描述 summarizer 的职责"""

    # ---------- 工厂 ----------
    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    # ---------- 基本配置 ----------
    @property
    def role_name(self) -> str:  # noqa: D401
        return "summarizer"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_summarizer"
    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_summarizer"

    # ---------- Prompt 参数 ----------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """根据前置工具结果构造 prompt 参数
        提示词中的占位符：
        """
        # TODO: 按需补充
        return {
            "processed_file_paths": pre_tool_results.get("processed_file_paths", []),
            "task_objective": pre_tool_results.get("task_objective", ""),
            "raw_file_paths": pre_tool_results.get("raw_file_paths", []),
            "score": pre_tool_results.get("score", []),
            "score_rule": pre_tool_results.get("score_rule", []),
            "summarizing_trace_summary": pre_tool_results.get("summarizing_trace_summary", ""),
            "task_meta": pre_tool_results.get("task_meta", []),
            "MAX_REACT_STEPS": pre_tool_results.get("MAX_REACT_STEPS", 7),
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
        super().update_state_result(state, result, pre_tool_results)

async def summarizer(
    state: TableState, 
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    use_agent: bool = False,
    **kwargs,
) -> TableState:
    """数据分析的入口函数"""
    summarizer = Summarizer(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await summarizer.execute(state, use_agent=use_agent, **kwargs)


def create_summarizer_agent(tool_manager: Optional[ToolManager] = None, **kwargs) -> Summarizer:
    """创建总结器代理实例"""
    return Summarizer(tool_manager=tool_manager, **kwargs)