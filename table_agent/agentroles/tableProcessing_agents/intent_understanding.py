
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
@register("intent_understanding")
class IntentUnderstanding(BaseAgent):
    """TODO: 描述 intent_understanding 的职责"""

    # ---------- 工厂 ----------
    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    # ---------- 基本配置 ----------
    @property
    def role_name(self) -> str:  # noqa: D401
        return "intent_understanding"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_intent_understanding"
    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_intent_understanding"

    # ---------- Prompt 参数 ----------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """根据前置工具结果构造 prompt 参数
        提示词中的占位符：
        """
        return {
            'user_input': pre_tool_results.get('user_input', []),
            'task_meta': pre_tool_results.get('data_profiling', {}),

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

        state.user_query = result
        state.task_type = result.get("task_type")
        super().update_state_result(state, result, pre_tool_results)

async def intent_understanding(
    state: TableState, 
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    use_agent: bool = False,
    **kwargs,
) -> TableState:
    """意图理解的入口函数"""
    generator = IntentUnderstanding(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await generator.execute(state, use_agent=use_agent, **kwargs)


def create_intent_understanding_agent(tool_manager: Optional[ToolManager] = None, **kwargs) -> IntentUnderstanding:
    """创建意图理解代理实例"""
    return IntentUnderstanding(tool_manager=tool_manager, **kwargs)