from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataflow.cli_funcs.paths import DataFlowPath
current_file = Path(__file__).resolve()

BASE_DIR = DataFlowPath.get_dataflow_dir()
DATAFLOW_DIR = BASE_DIR.parent
STATICS_DIR = DataFlowPath.get_dataflow_statics_dir()
PROJDIR = current_file.parent.parent

from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


# ==================== 最基础的 Request ====================
@dataclass
class MainRequest:
    """所有Request的基类，只包含核心字段"""
    # ① 用户偏好的自然语言
    language: str = "en"  # "en" | "zh" | ...

    # ② LLM 接口
    chat_api_url: str = os.getenv("API_URL")
    api_key: str = os.getenv("API_KEY")
    chat_api_key: str = os.getenv("API_KEY") 

    # ③ 选用的 LLM 名称
    model: str = "gpt-4o"

    # ④ 需求描述
    target: str = ""

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)


# ==================== 最基础的 State（所有State的祖先）====================
@dataclass
class MainState:
    """所有State的基类，只包含核心字段"""
    request: MainRequest = field(default_factory=MainRequest)
    messages: Annotated[list[BaseMessage], add_messages] = field(default_factory=list)
    # 通用字段
    agent_results: Dict[str, Any] = field(default_factory=dict)
    temp_data: Dict[str, Any] = field(default_factory=dict)
    llm_tracker: Any = None  # 可选的共享LLMTracker 实例(从llm_callers导入)

    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

