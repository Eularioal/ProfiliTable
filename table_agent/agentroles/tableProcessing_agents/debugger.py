from __future__ import annotations

from typing import Any, Dict, Generator, Optional

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
@register("debugger")
class Debugger(BaseAgent):
    """TODO: 描述 debugger 的职责"""

    # ---------- 工厂 ----------
    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    # ---------- 基本配置 ----------
    @property
    def role_name(self) -> str:  # noqa: D401
        return "debugger"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_debugger"
    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_debugger"

    # ---------- Prompt 参数 ----------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """根据前置工具结果构造 prompt 参数
        提示词中的占位符：
        """
        # TODO: 按需补充
        return {
            "code": pre_tool_results.get("code", ""),
            "error": pre_tool_results.get("error", ""),
            'target': pre_tool_results.get('target', ''),
            'input_file_paths': pre_tool_results.get('input_file_paths', ''),
            'debug_history': pre_tool_results.get('debug_history', ''),
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

async def debugger(
    state: TableState, 
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    use_agent: bool = False,
    **kwargs,
) -> TableState:
    """数据分析的入口函数"""
    debugger = Debugger(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await debugger.execute(state, use_agent=use_agent, **kwargs)


def create_debugger_agent(tool_manager: Optional[ToolManager] = None, **kwargs) -> Generator:
    """创建调试器代理实例"""
    return Debugger(tool_manager=tool_manager, **kwargs)