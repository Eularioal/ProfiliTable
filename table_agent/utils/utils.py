#!/usr/bin/env -S python -u
from pathlib import Path
import os
from typing import Union, List, Dict, Any
import re
import subprocess
from json import JSONDecodeError, JSONDecoder
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# 确保先导入 torch
from typing import List, Dict, Any
import shutil
from sentence_transformers import SentenceTransformer
import sys
import numpy as np
import json
import yaml
import time
from table_agent.utils.constants import MODEL_RATES
from table_agent.utils.logger import get_logger

log = get_logger()

_EMBEDDING_MODEL = None

def _get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        model_path = r"" # fill in your model path here
        _EMBEDDING_MODEL = SentenceTransformer(model_path, device='cuda:0')
        log.info("Embedding model loaded successfully.")
    return _EMBEDDING_MODEL


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def robust_parse_json(
    text: str,
    *,
    merge_dicts: bool = False,
    strip_double_braces: bool = False
) -> Union[Dict[str, Any], List[Any]]:
    """
    尽量从 LLM / 日志 / jsonl / Markdown 片段中提取合法 JSON。

    参数
    ----
    text : str
        输入原始文本
    merge_dicts : bool, default False
        提取到多个对象且全部是 dict 时，是否用 dict.update 合并返回
    strip_double_braces : bool, default False
        把 '{{' / '}}' 替换成 '{' / '}'（某些模板语言会加双层花括号）

    返回
    ----
    Dict / List / List[Dict | List]
    """
    s = text.strip()

    # ---------- 预处理：剥去外层包裹 ----------
    s = _remove_markdown_fence(s)          # ```json ... ```
    s = _remove_outer_triple_quotes(s)     # ''' ... ''' / """ ... """
    s = _remove_leading_json_word(s)       # 开头一个 json/JSON 标记

    if strip_double_braces:
        s = s.replace("{{", "{").replace("}}", "}")

    # ---------- 清理注释 & 尾逗号 ----------
    s = _strip_json_comments(s)

    # ---------- 新增：清理非法控制字符 ----------
    # 移除所有 JSON 规范不允许的 ASCII 控制字符。
    # 合法的 \n, \r, \t, 和 \f, \b, \" 都不会被移除，但这里只针对不可打印的控制码。
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)

    log.debug(f'清洗完之后内容是： {s}')

    # ---------- Step-1：整体解析 ----------
    # Step-1
    try:
        result = json.loads(s)
        log.info(f"整体解析成功，类型: {type(result)}")
        return result
    except JSONDecodeError as e:
        log.warning(f"整体解析失败: {e}")

    # ---------- Step-2：尝试 JSON Lines ----------
    objs = _parse_json_lines(s)
    if objs is not None:
        return _maybe_merge(objs, merge_dicts)

    # ---------- Step-3：流式提取多个对象 ----------
    objs = _extract_json_objects(s)
    log.warning(f"提取到 {len(objs)} 个对象")
    if not objs:
        raise ValueError("Unable to locate any valid JSON fragment.")

    return _maybe_merge(objs, merge_dicts)



# ======================================================================
#                            工具函数
# ======================================================================

_fence_pat = re.compile(r'```[\w-]*\s*([\s\S]*?)```', re.I)


def _remove_markdown_fence(src: str) -> str:
    """提取 ``` … ``` 内文本；若没找到则原样返回"""
    blocks = _fence_pat.findall(src)
    return "\n".join(blocks).strip() if blocks else src


def _remove_outer_triple_quotes(src: str) -> str:
    if (src.startswith("'''") and src.endswith("'''")) or (
        src.startswith('"""') and src.endswith('"""')
    ):
        return src[3:-3].strip()
    return src


def _remove_leading_json_word(src: str) -> str:
    return src[4:].lstrip() if src.lower().startswith("json") else src


def _strip_json_comments(src: str) -> str:
    # /* ... */  块注释
    src = re.sub(r'/\*[\s\S]*?\*/', '', src)
    # // ...     行注释，排除 URL 中的 :// 和字符串内的 //）
    src = re.sub(r'(?<![:\"\'])//.*', '', src)
    # 尾逗号 ,}
    src = re.sub(r',\s*([}\]])', r'\1', src)
    return src.strip()


# ----------------  JSON Lines ----------------
def _parse_json_lines(src: str) -> Union[List[Any], None]:
    lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
    if len(lines) <= 1:          # 只有 0/1 行就不是 jsonl
        return None

    objs: List[Any] = []
    for ln in lines:
        try:
            objs.append(json.loads(ln))
        except JSONDecodeError:
            return None  # 某行不是合法 JSON，放弃 jsonl 方案
    return objs


# ------------  多对象提取（改进版） ------------
def _extract_json_objects(src: str) -> List[Any]:
    dec = JSONDecoder()
    idx, n = 0, len(src)
    objs: List[Any] = []

    while idx < n:
        m = re.search(r'[{\[]', src[idx:])
        if not m:
            break
        idx += m.start()
        try:
            obj, end = dec.raw_decode(src, idx)
            # ========== 严格性检查 ==========
            tail = src[end:].lstrip()
            # 允许结束、逗号、换行、右括号、右中括号
            if tail and tail[0] not in ',]}>\n\r':
                idx += 1  # 可能是误判，如  {"a":1  <-- 缺 }
                continue
            objs.append(obj)
            idx = end
        except JSONDecodeError:
            idx += 1
    return objs


def _maybe_merge(objs: List[Any], merge_dicts: bool) -> Union[Any, List[Any]]:
    if len(objs) == 1:
        return objs[0]
    if merge_dicts and all(isinstance(o, dict) for o in objs):
        merged: Dict[str, Any] = {}
        for o in objs:
            merged.update(o)
        return merged
    return objs


# ========================================================================

#                           For tableProcessing

# ========================================================================


def find_csv_or_path_in_subdir(root_dir: str, target_subdir_name: None) -> list[Path]:
    root = Path(root_dir)
    csv_files = []

    # 遍历 root_dir 下的每一个直接子目录
    for child_dir in root.iterdir():
        if child_dir.is_dir():
            if not target_subdir_name:
                # 不进入子目录，遍历eval.py， task_json，返回两个，第一个eval.py，第二个task_meat(任务类别)
                return child_dir / 'eval.py', child_dir, 'task_meta.json'
            target_dir = child_dir / target_subdir_name
            if target_dir.exists() and target_dir.is_dir():
                # 查找该 target_dir 下所有 .csv 文件
                for csv_file in target_dir.glob("*.csv"):
                    csv_files.append(csv_file)
    return csv_files


def get_paths(base_dir: str) -> tuple[str, str]:
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    code_path = os.path.join(base_dir, 'generated_code.py')
    # 不需要后缀了，因为可能写入多组文件，多种不同后缀格式，所以这里不限制后缀，只指定文件夹
    processed_table_path = os.path.join(base_dir, "results") 
    if os.path.exists(processed_table_path):
        # Remove existing directory and its contents
        shutil.rmtree(processed_table_path)
    os.makedirs(processed_table_path)
    return code_path, processed_table_path



def safe_exec_code(py_path, output_path: list, input_path=None):
    time_before = time.time()
    py_path = Path(py_path)  # 确保是 Path 对象（即使传入的是字符串）
    if not os.path.isfile(py_path):
        raise FileNotFoundError(f"Script not found: {py_path}")
    if py_path.suffix.lower() != '.py':
        raise ValueError("Only .py files are allowed")
    if input_path:
        result = subprocess.run(
            [sys.executable, py_path, "--input", *input_path, "--output", output_path],
            capture_output=True,
            text=True,
            timeout=600  
        )
    else:
        output_args = [arg for out in output_path for arg in ("--output", out)]
        result = subprocess.run(
            [sys.executable, py_path, *output_args],
            capture_output=True,
            text=True,
            timeout=600
        )

    if result.returncode != 0:
        raise RuntimeError(f"Script failed:\n{result.stderr}")

    return result.stdout.strip(), time.time() - time_before



def retrive_operators(json_path: str, task_type: str, user_input: str) -> list:
    model = _get_embedding_model()
    json_data = json.load(open(json_path, 'r', encoding='utf-8'))
    task_type_1, task_type_2 = task_type.split('-')
    query_embedding = model.encode(user_input)
    descriptions = [
        member.get("description", "")
        for member in json_data.get(task_type_1, {}).get(task_type_2, [])
    ]
    if not descriptions:
        return []
    # 计算每个description的embedding
    descrip_embeddings = model.encode(descriptions)
    # 对向量进行归一化
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    descrip_norm = descrip_embeddings / np.linalg.norm(descrip_embeddings, axis=1, keepdims=True)
    # 计算余弦相似度
    similarities = np.dot(descrip_norm, query_norm)
    similarity_threshold = 0.5
    top_2_indices = np.argsort(similarities)[-2:][::-1]
    top_2_indices = [i for i in top_2_indices if similarities[i] >= similarity_threshold]
    if not top_2_indices:
        log.info("未找到相似度高于阈值的operator, 当前最高相似度为: {:.4f}".format(np.max(similarities)))
        return []
    # 返回对应的operator信息，可以根据需要修改返回内容
    top_2 = [json_data.get(task_type_1, {}).get(task_type_2, [])[i] for i in top_2_indices]
    return top_2




def load_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    

def calculate_money(model_name, input_tokens, output_tokens):
    # 按量计费费用 = 分组倍率 × 模型倍率 × （提示token数 + 补全token数 × 补全倍率）/ 500000 （单位：美元）  
    rates = MODEL_RATES.get(model_name)
    if not rates:
        raise ValueError(f"未知模型名称: {model_name}") 
    model_rate = rates["模型倍率"]
    completion_rate = rates["补全倍率"]
    group_rate = rates["分组倍率"]
    cost = group_rate * model_rate * (input_tokens + output_tokens * completion_rate) / 500000
    return cost


def extract_eval_score(stdout_str):
    # 使用正则查找可能存在的 JSON 对象，包含 "eval_score" 键
    # 匹配：{"eval_score": "0.1234"} 或 {"eval_score": "1.0000"} 等
    u_logger = get_logger()
    u_logger.info("开始匹配eval_score")
    u_logger.debug(f"stdout内容：{stdout_str}")
    match = re.search(r"[\"']eval_score[\"']\s*:\s*[\"']([^\"']*)[\"']", stdout_str)
    u_logger.info("完成匹配eval_score")
    if match:
        score_str = match.group(1)
        try:
            return float(score_str)
        except ValueError:
            pass
    # 如果没找到或转换失败，可选：尝试更宽松的匹配（比如数值不带引号）
    match2 = re.search(r'\{\s*"eval_score"\s*:\s*([0-9]*\.?[0-9]+)\s*\}', stdout_str)
    if match2:
        try:
            return float(match2.group(1))
        except ValueError:
            pass
    u_logger.error("未能提取eval_score")
    raise ValueError("Failed to extract eval_score from stdout")



