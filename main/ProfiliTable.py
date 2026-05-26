
from __future__ import annotations
import asyncio
import json
import pytest
import argparse
import os
import time
import datetime
import traceback
from pathlib import Path

from table_agent.utils.utils import load_config

from table_agent.utils.logger import setup_logging, get_logger, switch_log_file, suppress_noisy_loggers
# ------------ 依赖 -------------
from table_agent.states.tableState import TableState, TableRequest
from table_agent.workflow import run_workflow

from table_agent.utils.data_profiling import profile_multiple_csvs
from table_agent.llm_callers.text import TextLLMCaller

# ============ 核心异步流程 ============
async def run_my_workflow_pipeline(
    raw_table_paths: list,
    user_input: str,
    res_path: str,
    eval_path: str,
    score_rule: str,
    model: str,
    task_name: str,
    gt_table_path: str = None,
    score_threshold: float = 0,
    task_type: str = "DataImputation",
    use_rag: bool = True,
    is_dag: bool = False,
    is_op: bool = False,
    llm_tracker = None,
    operator_json_path: str = None,
) -> TableState:
    """
    执行 my_workflow 工作流的测试流程
    """
    logger.info(f"📁 数据路径: {raw_table_paths}")
    logger.info(f"🎯 用户输入: {user_input}")
    # data_profiling
    logger.info("🔍 开始默认data profiling...")
    data_profiling = profile_multiple_csvs(raw_table_paths, res_path)
    if "error" in data_profiling.values():
        logger.error("Data profiling 失败。")
    user_input = user_input
    
    # 2) 初始化状态
    state = TableState(
        task_objective=user_input,
        score_threshold=score_threshold,
        task_type=task_type,
        res_path=res_path,
        data_profiling=data_profiling,
        raw_table_paths=raw_table_paths,
        score_func_path=eval_path,
        gt_table_path=gt_table_path,
        profiling_trace_summary="",
        summarizing_trace_summary="",
        score=-1,
        valid=True,
        attempts=0,
        use_rag=use_rag,
        is_dag=is_dag,
        is_op=is_op,
        error_logs=[],
        execution_time=0.0,
        score_rule=score_rule,
        debug_attempts=0,
        model=model,
        task_name=task_name,
        operator_json_path=operator_json_path,
        current_best_score_and_code=(0.0, ""),  
    )
    llm_trakcer = TextLLMCaller(state, model_name=state['model'], temperature=0.3, max_tokens=10000)
    state['llm_tracker'] = llm_trakcer

    logger.info("🚀 启动 my_workflow 工作流...")
    final_state: TableState = await run_workflow("table_processing_workflow", state)
    logger.info(
        "🏆 任务结束。最终状态:"
        f"\nScore: {final_state.get('current_best_score_and_code')[0]}"
        f"\nValid: {final_state.get('valid')}"
        f"\nAttempts: {final_state.get('attempts')}"
    )
    return final_state






def parse_args():
    parser = argparse.ArgumentParser(description="Choose evaluation mode for TableAgent")
    parser.add_argument('--score_threshold', type=float, default=0.1)
    parser.add_argument("--task", type=str, help="(e.g., T0001, T0011, ...) for specific evaluation mode")
    parser.add_argument("--tasks", nargs="+", type=str, help="Specify multiple tasks at once (e.g., --tasks T0001 T0002 T0100)")
    parser.add_argument("--task-range", nargs=2, type=int, metavar=("START", "END"), help="Run tasks from START to END inclusive (e.g., --task-range 82 100)")
    parser.add_argument("--use_rag", action="store_true", help="Whether to use RAG (Retrieval-Augmented Generation) strategy")
    parser.add_argument("--use_local", action="store_true", help="Whether to use local LLM for LLM calls")
    parser.add_argument("--model", type=str, default="gpt-4o", help="deepseek-r1/v3, qwen-max-latest, gpt-5, claude-4-opus, gemini-2.5-pro, grok-4, llama-4-scout")
    parser.add_argument("--is_dag", action="store_true")
    parser.add_argument("--is_op", action="store_true")
    parser.add_argument("--output_path", type=str, default="outcome")
    parser.add_argument("--input_path", type=str, default="NL2Op", help='NL2Op, NL2Dag')
    return parser.parse_args()

async def run_task(child_dir: Path, use_rag: bool, is_dag: bool, is_op: bool, score_threshold: float, model: str, output_root_path: str, operator_json_path: str):
    task_name = child_dir.name
    logger.info(f"🚀 启动任务: {task_name}")
    res_path = os.path.join(output_root_path, task_name)
    os.makedirs(res_path, exist_ok=True)
    with open(child_dir / "task_meta.json", "r", encoding="utf-8") as f:
        task_data = json.load(f)
    user_input = task_data.get("target_en", "")
    score_rule = task_data.get("score_rule", "")
    eval_path = child_dir / "eval.py"
    raw_paths = list((child_dir / "raw").glob("*"))
    if not raw_paths:
        logger.warning(f"⚠️ {task_name} 没有匹配到 raw 文件，跳过。")
        return
    gt_path = next((child_dir / "expected").glob("gt.*"), None)

    start_ts = time.perf_counter()
    try:
        await run_my_workflow_pipeline(
            raw_table_paths=raw_paths,
            user_input=user_input,
            res_path=res_path,
            gt_table_path=gt_path,
            eval_path=eval_path,
            use_rag=use_rag,
            is_dag=is_dag,
            is_op=is_op,
            score_threshold=score_threshold,
            score_rule=score_rule,
            model=model,
            task_name=task_name,
            operator_json_path=operator_json_path,
        )
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"任务 {task_name} 执行时抛出异常: {e}")
        logger.error(f"Traceback:\n{tb}")
    finally:
        elapsed = time.perf_counter() - start_ts
        readable = str(datetime.timedelta(seconds=int(elapsed)))
        logger.info(f"⏱️ 任务 {task_name} 耗时：{elapsed:.3f}s ({readable})")

# ============ pytest 入口 ============
@pytest.mark.asyncio
async def test_my_workflow_pipeline():
    """
    测试 my_workflow 工作流的完整流程
    """
    final_state = await run_my_workflow_pipeline()

    assert final_state is not None, "final_state 不应为 None"
    assert hasattr(final_state, "agent_results"), "state 应包含 agent_results"
    

    print("\n=== agent_results ===")
    print(final_state.agent_results)
    
    if hasattr(final_state, "messages") and final_state.messages:
        print("\n=== messages ===")
        for msg in final_state.messages:
            print(f"- {msg}")


if __name__ == "__main__":
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    config = load_config("config.yaml")
    OPERATOR_JSON_PATH = config['paths']['operator_json_path']
    model = args.model
    is_dag = args.is_dag
    is_op = args.is_op
    score_threshold = args.score_threshold
    selected_tasks = set()
    selected_tasks_str = ''
    if args.task:
        selected_tasks.add(args.task)
        selected_tasks_str += args.task
    elif args.task_range:
        selected_tasks_str += str(args.task_range)
        start, end = args.task_range
        if start > end:
            raise ValueError("--task-range START must be <= END")
        selected_tasks.update({f"T{num:04d}" for num in range(start, end + 1)})
    elif args.tasks:
        selected_tasks.update(args.tasks)
        selected_tasks_str += '_'.join(args.tasks)
    else:
        selected_tasks_str += 'all'
        selected_tasks = None  
    if selected_tasks is not None and len(selected_tasks) > 10:
        selected_tasks_str = selected_tasks_str[:10] + '_etc'
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    use_rag = args.use_rag
    log_path = os.path.join(
        config["paths"]["log_path"],
        f"task_type{args.input_path}_RAG{use_rag}_tasks{selected_tasks_str}_model_{model}_{timestamp}.log",
    )
    setup_logging(
    log_path=log_path,
    console_level="INFO",
    file_level="DEBUG"
)
    logger = get_logger(__name__)
    task_root_path = Path(config["paths"]["task_root_path"]) / args.input_path
    output_root_path = args.output_path
    if not os.path.exists('outcomes'):
        os.mkdir('outcomes')
    output_root_path = 'outcomes/' + output_root_path + '_' + model + '_' + args.input_path
    if use_rag:
        output_root_path += '_rag'
    os.makedirs(output_root_path, exist_ok=True)
    durations_file_path = os.path.join(output_root_path, "task_durations.csv")
    for child_dir in task_root_path.iterdir():
        task_name = child_dir.name
        res_path = os.path.join(output_root_path, task_name)
        if selected_tasks is not None:
            task_code = task_name.split("_")[0]
            if task_code not in selected_tasks:
                continue
        log_path = res_path + f'/{task_name}_log.log'
        if os.path.exists(log_path):
            os.remove(log_path)
        switch_log_file(res_path + f'/{task_name}_log.log', console_level="INFO", file_level="DEBUG")
        suppress_noisy_loggers()
        asyncio.run(run_task(
            child_dir,
            use_rag=use_rag,
            is_dag=is_dag,
            is_op=is_op,
            score_threshold=score_threshold,
            model=model,
            output_root_path=output_root_path,
            operator_json_path=OPERATOR_JSON_PATH,
        ))
