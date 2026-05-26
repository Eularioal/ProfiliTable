from table_agent.states.base_state import MainRequest, MainState
from dataclasses import dataclass, field
from typing_extensions import TypedDict
from typing import Annotated, Literal, List


@dataclass
class TableRequest(MainRequest):
    """主流程的Request，继承自MainRequest"""
    # ⑤ 测试样例文件（仅 CLI 批量跑用）
    json_file: str = ""

    # ⑥ Python 代码文件位置
    python_file_path: str = ""

    # ⑦ Debug 相关
    need_debug: bool = False
    max_debug_rounds: int = 3

    # ⑧ 本地模型相关
    use_local_model: bool = False
    local_model_path: str = ""

    # ⑨ 缓存和会话
    session_id: str = "default_session"

    # embeddings url
    chat_api_url_for_embeddings : str = ""
    embedding_model_name: str = "text-embedding-3-small"
    update_rag_content: bool = True



@dataclass
class TableState(MainState):
    # 重写request类型为DFRequest
    request: TableRequest = field(default_factory=TableRequest)
    task_type: str = field(default_factory=str)
    task_objective: str = field(default_factory=str)
    score_threshold: float = field(default=0.0)
    res_path: str = field(default="")
    data_profiling: str = field(default="")
    gt_table_path: str = field(default="")  # 本地 CSV/Parquet 路径（不传给 LLM！）
    raw_table_paths: str = field(default="")  # noisy 表路径
    score_func_path: str = field(default="")  # 评分代码路径
    user_query: str = field(default="")  # 意图识别用户的结果
    is_dag: bool = field(default=False) # 手动指定Dag
    is_op: bool = field(default=False)  # 手动指定op
    score_rule: str = field(default="")  # 评分规则
    profiling_trace_summary: str = field(default="")  # profiling 过程的 profiling_trace_summary
    summarizing_trace_summary: str = field(default="")  # summarizing 过程的 summarizing_trace_summary
    score: float = field(default=0.0)
    valid: bool = field(default=False)
    attempts: int = field(default=0)
    use_rag: bool = field(default=False)  # 是否启用 RAG（检索增强生成）
    execution_time: float = field(default=0.0)
    debug_attempts: int = field(default=0)
    model: str = field(default="")
    task_name: str = field(default="")
    operator_json_path: str = field(default="")
    summary: str = field(default="")  # 最终生成的 summary 结果
    processed_file_paths: List[str] = field(default_factory=list)  # 处理好的文件列表路径

    # === 新增：用于维护节点记忆 ===
    generated_codes: Annotated[
        List[str], lambda x, y: x + y
    ] = field(default_factory=list)  # 每轮生成的代码（字符串）
    error_logs: Annotated[List[str], lambda x, y: x + y] = field(default_factory=list)  # 每轮执行错误（字符串）
    evaluation_feedbacks: Annotated[
        List[dict], lambda x, y: x + y
    ] = field(default_factory=list)  # 每轮评分详情（如 {"score": 0.5, "reason": "..."})

    script_generated_total: int = field(default=0)  # 生成代码脚本总数
    script_runnable_total: int = field(default=0)  # 可执行代码脚本总数
    debug_total_attempts: int = field(default=0)  # 调试总次数
    debug_reasons: Annotated[List[str], lambda x, y: x + y] = field(default_factory=list)  # 每轮调试原因（字符串）
    current_best_score_and_code: tuple = field(default=(0.0, ""))  # 当前最佳 (score, code) 对
    
    # 新增：用于 decomposition_codes agent 角色
    decomposition_result: str = field(default="")  # 任务分解结果
    decomposition_codes: str = field(default="")  # 任务分解代码结果
    retrieved_operators: List[str] = field(default_factory=list)  # 检索到的相关操作代码
