from typing import List
from langchain_core.messages import BaseMessage,AIMessage
from langchain_openai import ChatOpenAI

from .base import BaseLLMCaller
from table_agent.utils.logger import get_logger
from table_agent.utils.utils import calculate_money
import time

log = get_logger(__name__)

class TextLLMCaller(BaseLLMCaller):
    """文本LLM调用器 - 集成LLMTracker功能"""
    
    def __init__(self, state, model_name, **kwargs):
        super().__init__(state=state, model_name=model_name, **kwargs)
        self.input_tokens = 0
        self.output_tokens = 0
        self.completion_time = 0.0
        self.call_count = 0
        self.llm = ChatOpenAI(
            openai_api_base=self.state.request.chat_api_url,
            openai_api_key=self.state.request.api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        log.info(f"temperature: {self.temperature}, max_tokens: {self.max_tokens}")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_cost(self) -> float:
        return calculate_money(
            self.model_name,
            self.input_tokens,
            self.output_tokens
        )

    def _extract_usage(self, response: AIMessage) -> dict:
        """统一提取 usage 信息（兼容 OpenAI / Anthropic）"""
        usage = response.response_metadata.get("token_usage") or {}
        meta = getattr(response, "usage_metadata", {})
        if meta:
            return {
                "input_tokens": meta.get("input_tokens", 0),
                "output_tokens": meta.get("output_tokens", 0),
            }
        usage = response.response_metadata.get("usage", {})
        if usage:
            return {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            }
        return {"input_tokens": 0, "output_tokens": 0}

    async def __call__(self, messages: List[BaseMessage], bind_post_tools: bool = False) -> AIMessage:
        log.info(f"TextLLM调用，模型: {self.model_name}")
        
        start = time.time()
        try:
            response = await self.llm.ainvoke(messages)
        except Exception as e:
            response = self.llm.invoke(messages)  # 回退同步调用
        finally:
            elapsed = time.time() - start
            self.completion_time += elapsed

        usage = self._extract_usage(response)
        self.input_tokens += usage["input_tokens"]
        self.output_tokens += usage["output_tokens"]
        self.call_count += 1

        return response

    def bind_tools(self, post_tools, tool_choice):
        """绑定后处理工具（目前无实现）"""
        if post_tools:
            self.llm = self.llm.bind_tools(post_tools, tool_choice=tool_choice)
            log.info(f"[create_llm]:为LLM绑定了 {len(post_tools)} 个后置工具: "
                    f"{[t.name for t in post_tools]}")
        return self

    def reset(self):
        """重置统计（用于多轮子任务）"""
        self.input_tokens = 0
        self.output_tokens = 0
        self.completion_time = 0.0
        self.call_count = 0

    def summary(self) -> dict:
        return {
            "model": self.model_name,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "completion_time_sec": round(self.completion_time, 3),
            "call_count": self.call_count,
            "total_cost_usd": round(self.total_cost, 6),
        }