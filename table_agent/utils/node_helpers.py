"""Helper utilities to reduce duplication inside node implementations."""
from __future__ import annotations
import os
from pathlib import Path
import json
import re
from typing import List, Tuple, Dict, Any
from langchain_core.messages import AIMessage, BaseMessage
from table_agent.utils.refactor_constants import (
    EVAL_RESULT_FILENAME,
    THINK_TAG_PATTERN,
    ACTION_TAG_PATTERN,
    ANSWER_TAG_PATTERN,
    OBS_TAG_WRAPPER,
    CODE_LOG_TRUNCATE,
    EMPTY_CODE_FALLBACK,
)
from table_agent.utils.logger import get_logger

logger = get_logger()

def parse_react_output(raw: str) -> Dict[str, Any]:
    """Parse ReAct style output into structured parts.

    Returns dict with keys: thinks(List[str]), action_code(str|None), answer_obj(dict|None), errors(List[str])
    """
    result = {"thinks": [], "action_code": None, "answer_obj": None, "errors": []}
    think_blocks = re.findall(THINK_TAG_PATTERN, raw, re.DOTALL | re.IGNORECASE)
    result["thinks"] = [t.strip() for t in think_blocks if t.strip()]

    answer_match = re.search(ANSWER_TAG_PATTERN, raw, re.DOTALL)
    if answer_match:
        ans_raw = answer_match.group(1)
        try:
            result["answer_obj"] = json.loads(ans_raw)
        except Exception as e:
            logger.info(f"Answer is a string, not JSON")
            result["answer_obj"] = ans_raw.strip()
            
    action_match = re.search(ACTION_TAG_PATTERN, raw, re.DOTALL | re.IGNORECASE)
    if action_match and result["answer_obj"] is None:  # only consider action if no final answer
        act_raw = action_match.group(1)
        try:
            result["action_code"] = extract_python_code_block(act_raw)
        except Exception as e:
            result["errors"].append(f"action_json_parse_failed: {e}")

    return result

def build_observation_payload(status: str, **kwargs) -> Dict[str, Any]:
    payload = {"status": status}
    payload.update(kwargs)
    return payload

def observation_to_message(obs: Dict[str, Any]) -> str:
    return OBS_TAG_WRAPPER.format(obs=json.dumps(obs, ensure_ascii=False, separators=(",", ":")))

def truncate_for_log(text: str, limit: int = CODE_LOG_TRUNCATE) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"

def extract_python_code_block(content: str) -> str:
    match = re.search(r"```python\s*(.*?)\s*```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip() or EMPTY_CODE_FALLBACK

def write_code_file(res_path: str, code: str) -> Path:
    code_path, _ = _ensure_paths(res_path)
    Path(code_path).write_text(code, encoding="utf-8")
    return Path(code_path)

def _ensure_paths(res_path: str) -> Tuple[str, str]:
    # Re-use existing get_paths logic indirectly if needed.
    from table_agent.utils.utils import get_paths
    return get_paths(res_path)

def _write_eval_result(state: Dict):

    llm_tracker = state["llm_tracker"]
    summary = llm_tracker.summary()
    money_cost = summary["total_cost_usd"]
    input_tokens = summary["input_tokens"]
    output_tokens = summary["output_tokens"]
    completion_time = summary["completion_time_sec"]
    task_name = state["task_name"]
    profiling = state["data_profiling"]


    eval_result_path = os.path.join(state["res_path"], EVAL_RESULT_FILENAME)
    summary_lines = [
        f"task_name: {task_name}",
        f"Score: {state.get('current_best_score_and_code',[0])[0]:.3f}",
        f"input_tokens: {input_tokens}",
        f"output_tokens: {output_tokens}",
        f"completion_time: {completion_time:.3f}",
        f"execution_time: {state.get('execution_time', 0):.3f}",
        f"Money Cost: {money_cost:.3f}",
        "",
        f"generated_attempts: {state.get('attempts', 0)}",
        f"debug_total_attempts: {state.get('debug_total_attempts', 0)}",
        f"script_generated_total: {state.get('script_generated_total', 0)}",
        f"script_runnable_total: {state.get('script_runnable_total', 0)}",
        "",
    ]
    data_profiling_path = os.path.join(state["res_path"], "data_profiling.json")
    with open(data_profiling_path, "w", encoding="utf-8") as f:
        json.dump(profiling, f, ensure_ascii=False, indent=2)
    error_logs = state.get("error_logs", [])
    if error_logs:
        summary_lines.append("Error Logs:")
        summary_lines.extend(error_logs)
    content = "\n".join(summary_lines) + "\n"
    with open(eval_result_path, "w", encoding="utf-8") as f:
        f.write(content)

__all__ = [
    'parse_react_output', 'build_observation_payload', 'observation_to_message',
    'truncate_for_log', 'extract_python_code_block', 'write_code_file', '_write_eval_result'
]
