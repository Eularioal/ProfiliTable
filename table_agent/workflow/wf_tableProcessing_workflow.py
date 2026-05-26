
from __future__ import annotations
import json
import os
import traceback
from pathlib import Path
from typing import Literal
from dataclasses import Field

from sklearn import decomposition
from table_agent.graphbuilder.message_history import HumanMessage
from pydantic import BaseModel
from table_agent.states.tableState import TableState
# from dataflow_agent.state import xxState
from table_agent.graphbuilder.graph_builder import GenericGraphBuilder
from table_agent.workflow.registry import register
from table_agent.agentroles import (
    create_agent,
    create_simple_agent,
    create_react_agent,
    create_table_agent,
    create_graph_agent,
    create_vlm_agent,
    SimpleConfig,
    ReactConfig,
    GraphConfig,
    VLMConfig,
    ExecutionMode,
)

from table_agent.toolkits.tool_manager import get_tool_manager
from langchain.tools import tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from table_agent.llm_callers import TextLLMCaller
from table_agent.graphbuilder.graph_builder import GenericGraphBuilder
from table_agent.utils.logger import get_logger
from table_agent.utils.utils import retrive_operators, get_paths, safe_exec_code, extract_eval_score
from table_agent.utils.node_helpers import extract_python_code_block, write_code_file, _write_eval_result
from table_agent.utils.refactor_constants import MAX_DEBUG_ATTEMPTS, EVAL_RESULT_FILENAME, MAX_REACT_STEPS, MAX_GENERATE_ATTEMPTS

logger = get_logger(__name__)
CURRENT_DIR = Path(__file__).parent.parent.resolve()

@register("table_processing_workflow")
def create_my_workflow_graph() -> GenericGraphBuilder: 
    builder = GenericGraphBuilder(state_model=TableState,
                                  entry_point="intent_understanding")  
    # ----------------------------------------------------------------------
    # TOOLS (pre_tool definitions)
    # ----------------------------------------------------------------------
    @builder.pre_tool("user_input", "intent_understanding")
    def _user_input(state: TableState):
        user_input = state['task_objective']
        return user_input
    
    @builder.pre_tool("data_profiling", "intent_understanding")
    def _data_profiling(state: TableState):
        return state.get("data_profiling", {})
    
    # ----------------------------------------------------------------------
    @builder.pre_tool("raw_table_paths", "data_profiling")
    def _raw_table_paths(state: TableState):
        return state.raw_table_paths
    
    @builder.pre_tool("operation", "data_profiling")
    def _operation(state: TableState):
        return state['user_query']['operation']
    
    @builder.pre_tool("MAX_REACT_STEPS", "data_profiling")
    def _max_react_steps(state: TableState):
        return MAX_REACT_STEPS

    @builder.pre_tool("user_refine_input", "data_profiling")
    def _user_refine_input(state: TableState):
        score = state['score']
        score_rule = state.get('score_rule', '')
        profiling_trace_summary = state.get('profiling_trace_summary', '')
        insight = state.get('summary', '')
        return """
        "Your previous score was {score}. The score_rule is: {score_rule}.\n   Based on the previous profiling trace summary: {profiling_trace_summary}, \n  
        Based on the previous insight(about why agent don't do well): {insight}, try your best to improve the score to 1.0.
        """.format(
            score=score,
            score_rule=score_rule,
            profiling_trace_summary=profiling_trace_summary,
            insight=insight
        )
    # ----------------------------------------------------------------------
    @builder.pre_tool('user_query', 'decompositer')
    def _user_query(state: TableState):
        return state.get('user_query', '')

    @builder.pre_tool('benchmark_task_types', 'decompositer')
    def _benchmark_task_types(state: TableState):
        from table_agent.utils.constants import BENCHMARK_TASK_TYPES
        return BENCHMARK_TASK_TYPES
    # ------------------------------------------------------
    @builder.pre_tool("decomposition_result", "decomposition_codes")
    def _decomposition_result(state: TableState):
        return state.get('decomposition_result', {})
    
    @builder.pre_tool("retrieved_operators", "decomposition_codes")
    def _retrieved_operators(state: TableState):
        return state.get('retrieved_operators', [])

    # ----------------------------------------------------------------------
    @builder.pre_tool("task_meta", "generator")
    def _task_meta(state: TableState):
        return state['data_profiling']
    
    @builder.pre_tool("user_query", "generator")
    def _user_query(state: TableState):
        return state['user_query']
    @builder.pre_tool("user_input", "generator")
    def _user_input(state: TableState):
        user_inputs = [state['task_objective']]
        input_paths = " ".join(p.name for p in state["raw_table_paths"])
        user_inputs.append(f"输入文件的顺序与类型：{input_paths}")
        if state.get('debug_reasons'):
            # reasons似乎不需要压缩，除非是对整个debug历史进行总结
            user_inputs.append(
                f"Debug Reasons:{state['debug_reasons']} You need to avoid the previous mistakes based on the summary above."
            )
        if state.get("summary"):
                user_inputs.append(
                f"Additional Context from previous steps: {state.get('summary')}"
            )
                logger.info(f"Additional Context from previous steps added to generation: {state.get('summary')}")
        return user_inputs

    @builder.pre_tool("retrieved_operators", "generator")
    def _retrieved_operators(state: TableState):
        task_type = state["task_type"]
        user_query = state["user_query"]
        task_type_cate, task_type_sub_cate = task_type.split("-")
        operator_codes = []
        if state["use_rag"]:
            retrieved_operators = retrive_operators(state.get("operator_json_path"), task_type, user_query["operation"])
            operator_names = [op["name"] for op in retrieved_operators]
            for op_name in operator_names:
                op_path = CURRENT_DIR / 'utils' / 'operators' / task_type_cate / task_type_sub_cate / f"{op_name}.py"
                if op_path.is_file():
                    operator_codes.append((task_type, op_path.read_text(encoding='utf-8')))
            if len(operator_codes) == 0:
                logger.warning("RAG enabled but no operator codes found.")
        else:
            operator_names = []

        logger.info(f"GenerationStrategy: {task_type}")
        logger.debug(f"Retrieved operators: {operator_names}")
        operator_codes.extend(state.get('retrieved_operators', []))
        logger.info(f"Retrieved code snippets: {len(operator_codes)}")
        return operator_codes
    
    # @builder.pre_tool("decomposition_codes", "generator")
    # def _decomposition_codes(state: TableState):
    #     return state.get('decomposition_codes', '')
    # ----------------------------------------------------------------------
    @builder.pre_tool("code", "debugger")
    def _code(state: TableState):
        previews_generated_codes = state['generated_codes']
        return previews_generated_codes[-1] if previews_generated_codes else ""
    
    @builder.pre_tool("error", "debugger")
    def _error(state: TableState):
        error_logs = state['error_logs']
        return error_logs[-1] if error_logs else ""

    @builder.pre_tool("target", "debugger")
    def _target(state: TableState):
        return state['user_query']['operation']
    
    @builder.pre_tool("input_file_paths", "debugger")
    def _input_file_paths(state: TableState):
        input_paths = " ".join(p.name for p in state["raw_table_paths"])
        return input_paths
    
    @builder.pre_tool("debug_history", "debugger")
    def _debug_history(state: TableState):
        error_logs = state['error_logs']
        return error_logs[:-1] if len(error_logs) > 1 else "No previous debug history."

    # ----------------------------------------------------------------------
    @builder.pre_tool("processed_file_paths", "summarizer")
    def _processed_file_paths(state: TableState):
        return state.get("processed_file_paths", '')
    
    @builder.pre_tool("task_objective", "summarizer")
    def _task_objective(state: TableState):
        return state.get("task_objective", '')
    
    @builder.pre_tool("raw_file_paths", "summarizer")
    def _raw_file_paths(state: TableState):
        return [str(p) for p in state["raw_table_paths"]]

    @builder.pre_tool("score", "summarizer")
    def _score(state: TableState):
        return state.get('score', 0.0)

    @builder.pre_tool("score_rule", "summarizer")
    def _score_rule(state: TableState):
        return state.get('score_rule', '')
    
    @builder.pre_tool("summarizing_trace_summary", "summarizer")
    def _summarizing_trace_summary(state: TableState):
        return state.get('summarizing_trace_summary', '')

    @builder.pre_tool("task_meta", "summarizer")
    def _task_meta(state: TableState):
        return state.get('data_profiling', {})
    
    @builder.pre_tool("MAX_REACT_STEPS", "summarizer")
    def _max_react_steps(state: TableState):
        return MAX_REACT_STEPS

    # ==============================================================
    # NODES
    # ==============================================================
    async def intent_understanding(state: TableState) -> TableState:
        """
        示例节点 1: 使用新的策略模式创建和执行
        """
        logger.info("🔍 开始意图识别...")
        model = state["request"]["model"]
        # 实际使用：创建一个简单的代码审查 Agent
        agent = create_simple_agent(
            name="intent_understanding",       
            model_name=model,
            temperature=0.3,
            max_tokens=20480,
            parser_type="json",
        )
        state = await agent.execute(state=state)
        agent_result = state.agent_results.get(agent.role_name, {})
        user_query = agent_result.get("results", "")
        logger.info(f"意图识别原始输出：{user_query}")
        try:
            required = {"operation", "reason", "task_type", "suffix"}
            if not required.issubset(user_query.keys()):
                raise ValueError("Missing required fields")
            logger.success(f"✅ 意图解析成功: {user_query}")
        except Exception as e:
            logger.error(f"❌ 意图解析失败: {e}")
            raise ValueError(f"Failed to parse user query: {e}")
        
        logger.info(f"Agent {agent.role_name} 执行结果: {user_query}")
        
        return state
    
    async def data_profiling(state: TableState) -> TableState:
        """
        示例节点 2: 数据分析节点
        """
        logger.info("📊 开始profiling...")
        model = state["request"]["model"]
        agent = create_table_agent(
            name="data_profiling",        
            model_name=model,
            temperature=0.1,
            max_tokens=20480,
            parser_type="json",
        )
        
        state = await agent.execute(state=state)
        agent_result = state.agent_results.get(agent.role_name, {})
        data_profiling = agent_result['results']['answer']
        react_trace = agent_result['results'].get('react_trace', [])
        logger.info(f"Profiling 原始输出：{data_profiling}")
        summary_messages = [
            {"role": "system", "content": "Summarize the following ReAct trace briefly."},
            {"role": "user", "content": f"ReAct Trace: {json.dumps(react_trace, ensure_ascii=False)}"},
        ]
        local_summarizer = state['llm_tracker']
        local_summarizer_response = await local_summarizer(summary_messages)
        local_summarizer_response = local_summarizer_response.content.strip()
        logger.info(f"📊 Local Summarizer ReAct Trace Summary:\n{local_summarizer_response}")
        return {"profiling_trace_summary": local_summarizer_response,
                "data_profiling": data_profiling,
                "execution_time": agent_result['results'].get('execution_time', 0.0),}

        # return state

    async def decompositer(state: TableState) -> TableState:
        logger.info("🔄 开始任务分解...")
        model = state["request"]["model"]
        agent = create_simple_agent(
            name="decompositer",        # 替换为已注册的 agent 名称
            model_name=model,
            temperature=0.1,
            max_tokens=20480,
            parser_type="text",
        )

        state = await agent.execute(state=state)
        agent_result = state.agent_results.get(agent.role_name, {})
        decomposition_result = agent_result['results']['text']
        logger.info(f"Decompositer 原始输出：{decomposition_result}")
        operator_codes = []
        for sub_task, task_desc in json.loads(decomposition_result).items():
            logger.info(f"子任务: {sub_task} 描述: {task_desc}")
            task_type_cate, task_type_sub_cate = sub_task.split("-")
            retrieved_operators = retrive_operators(state.get("operator_json_path"), sub_task, task_desc)
            operator_names = [op["name"] for op in retrieved_operators]
            temp_operator_codes = []
            for op_name in operator_names:
                op_path = CURRENT_DIR / 'utils' / 'operators' / task_type_cate / task_type_sub_cate / f"{op_name}.py"
                if op_path.is_file():
                    temp_operator_codes.append((sub_task, op_path.read_text(encoding='utf-8')))
            if len(temp_operator_codes) == 0:
                logger.warning("RAG enabled but no operator codes found.")
            operator_codes.extend(temp_operator_codes)
        logger.info(f"Retrieved code snippets for decomposition: {len(operator_codes)}")
        return {"decomposition_result": decomposition_result, "retrieved_operators": operator_codes}

    async def decomposition_codes(state: TableState) -> dict:
        logger.info("🛠️ 开始代码片段检索与整合...")
        model = state["request"]["model"]
        agent_codes = create_agent(
            name="decomposition_codes",        # 替换为已注册的 agent 名称
            model_name=model,
            temperature=0.1,
            max_tokens=20480,
            parser_type="text",
        )
        state = await agent_codes.execute(state=state)
        agent_result_codes = state.agent_results.get(agent_codes.role_name, {})
        decomposition_codes = agent_result_codes['results']['text']
        logger.info(f"DecompositionCodes 原始输出：{decomposition_codes}")
        return {"decomposition_codes": decomposition_codes}

    async def generator(state: TableState) -> TableState:
        logger.info("🛠️ 开始代码生成...")
        model = state['model']
        
        agent = create_simple_agent(
            name="generator",        # 替换为已注册的 agent 名称
            model_name=model,
            temperature=0.1,
            max_tokens=20480,
            parser_type="text",
        )
        state = await agent.execute(state=state)
        code = extract_python_code_block(state["agent_results"]["generator"]["results"]['text'])
        code_path = write_code_file(state["res_path"], code)

        logger.info(f"GenerationStrategy 生成代码 (尝试 {state.get('attempts', 0)+1})")
        logger.info(f"代码已写入: {code_path}")
        logger.debug(f"Generated code length: {len(code)}")
        logger.debug(f"Generated code content:\n{code}")

        return state
    
    async def evaluator(state: TableState) -> TableState:
        logger.info("🧪 开始执行与评估...")
        execution_time = state['execution_time']
        script_generated_total = state.get('script_generated_total', 0) + 1
        script_runnable_total = state.get('script_runnable_total', 0)
        current_best_score_and_code = state.get("current_best_score_and_code", (0.0, ""))
        try:
            code_path, process_table_path = get_paths(state["res_path"])
            logger.info(f"Raw table paths: {state['raw_table_paths']}")
            execution_time += safe_exec_code(code_path, process_table_path, state["raw_table_paths"])[1]
            score_func_path = state["score_func_path"]
            processed_files = [str(f) for f in Path(process_table_path).iterdir() if f.is_file()]
            score_output, e_t = safe_exec_code(score_func_path, processed_files)
            execution_time += e_t
            score = extract_eval_score(score_output)
            if score >= current_best_score_and_code[0]:
                current_best_score_and_code = (score, state['generated_codes'][-1] if state['generated_codes'] else "")
            logger.success(f"✅ 评估成功 | Score: {score:.3f}")
            feedback = {"score": score, "status": "success", "reason": "Execution succeeded."}
            script_runnable_total += 1
            return {
                "messages": [AIMessage(content=f"[Evaluator] Score={score:.3f} feedback={feedback}")],
                "score": score,
                "valid": True,
                "current_best_score_and_code": current_best_score_and_code,
                "evaluation_feedbacks": [feedback],
                "execution_time": execution_time,
                "script_generated_total": script_generated_total,
                "script_runnable_total": script_runnable_total,
                "processed_file_paths": processed_files,
            }
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"[Evaluator] 执行失败: {e}\n{tb}"
            logger.error(f"💥 执行失败 | Error: {e}")
            feedback = {"score": 0.0, "status": "error", "reason": str(e), "traceback": tb}
            return {
                "messages": [AIMessage(content="Error:" + error_msg)],
                "score": 0.0,
                "valid": False,
                "error_logs": [error_msg],
                "evaluation_feedbacks": [feedback],
                "execution_time": execution_time,
                "script_generated_total": script_generated_total,
            }
        
    async def debugger(state: TableState) -> TableState:
        logger.info("🐞 开始调试代码...")
        code = ""
        reason = ""
        last_raw = ""
        agent = create_simple_agent(
            name="debugger",
            model_name=state['model'],
            temperature=0.1,
            max_tokens=20480,
            parser_type="text",
        )
        for attempt in range(MAX_DEBUG_ATTEMPTS):
            state = await agent.execute(state=state)
            last_raw = state["agent_results"]["debugger"]["results"]['text'].strip()
            try:
                parsed = json.loads(last_raw)
                code = parsed.get('code', '')
                reason = parsed.get('reason', '')
            except json.JSONDecodeError as e:
                logger.warning(f"调试 JSON 解析失败: {e}")
                continue
            if code:
                break

        if not code:
            eval_result_path = os.path.join(state["res_path"], EVAL_RESULT_FILENAME)
            with open(eval_result_path, "w", encoding="utf-8") as f:
                f.write("Score: 0.0\nValid: False\nError: Debugger 未能生成有效代码（多次失败）。\n")
            raise ValueError("Debugger 未能生成有效代码（多次失败）。")

        code_path = write_code_file(state["res_path"], code)
        logger.info(f"Debugger 生成调试后代码: {code_path}")

        return {
            "generated_codes": [code],
            "debug_attempts": state.get("debug_attempts", 0) + 1,
            "debug_total_attempts": state.get("debug_total_attempts", 0) + 1,
            "debug_reasons": [reason],
        }
    
    async def summarizer(state: TableState) -> TableState:
        logger.info("🔍🔍 Summarizer Node (ReAct THINK → ACTION → OBSERVE)")
        model = state["request"]["model"]
        agent = create_table_agent(
            name="summarizer",        # 替换为已注册的 agent 名称
            model_name=model,
            temperature=0.1,
            max_tokens=20480,
            parser_type="json",
        )

        state = await agent.execute(state=state)
        agent_result = state.agent_results.get(agent.role_name, {})
        summary = agent_result['results']['answer']
        react_trace = agent_result['results'].get('react_trace', [])
        local_summarizer = state['llm_tracker']
        summary_messages = [
            {"role": "system", "content": "Summarize the following ReAct trace briefly."},
            {"role": "user", "content": f"ReAct Trace: {json.dumps(react_trace, ensure_ascii=False)}"},
        ]
        local_summarizer_response = await local_summarizer(summary_messages)
        logger.info(f"🔍🔍 Local Summarizer ReAct Trace Summary:\n{local_summarizer_response.content.strip()}")
        logger.info(f"Summarizer 原始输出：{summary}")
        
        return {
            'summary': summary,
            "summarizing_trace_summary": local_summarizer_response.content.strip(),
            'execution_time': agent_result['results'].get('execution_time', 0.0),
        }



    async def finalizer(state: TableState) -> TableState:
        logger.info("✅️ 取分数最高的代码并执行...")
        execution_time = state['execution_time']
        current_best_score, best_code = state.get("current_best_score_and_code", (0.0, ""))
        # 把best_code写入文件并执行
        code_path, process_table_path = get_paths(state["res_path"])
        code_path = write_code_file(state["res_path"], best_code)
        logger.info(f"Finalizer 生成最终代码: {code_path} | Score: {current_best_score}")
        # 执行并生成processed文件
        execution_time += safe_exec_code(code_path, process_table_path, state["raw_table_paths"])[1]
        _write_eval_result(state)
        return {
            "execution_time": execution_time,
        }

    # ------------------------------------------------------------------
    # Ⅲ. 条件边
    # ------------------------------------------------------------------
    def should_debug(state: TableState) -> Literal["debugger", "summarizer", "finalizer"]:
        """根据评分与有效性决定是否进入调试节点(最多MAX_DEBUG_ATTEMPTS次调试尝试)。"""
        debug_attemps = state.get("debug_attempts", 0)
        valid = state.get("valid", False)
        if not valid and debug_attemps < MAX_DEBUG_ATTEMPTS:
            logger.info(f"🔄 任务未通过验证，进入调试节点（debug_attempts={debug_attemps}）")
            return "debugger"

        elif not valid and debug_attemps >= MAX_DEBUG_ATTEMPTS:
            logger.warning(f"⚠️ 达到最大调试次数 ({debug_attemps})，强制终止（valid={valid}）")
            return "finalizer"
        else:
            logger.info("✅ 任务通过验证，进入总结节点")
            if state.get("attempts", 0) >= MAX_GENERATE_ATTEMPTS:
                logger.warning(f"⚠️ 达到最大重试次数 ({state.get('attempts', 0)})，强制终止")
                return "finalizer"
            return "summarizer"


    def should_retry(state: TableState) -> Literal["data_profiling", "finalizer"]:

        attempts = state.get("attempts", 0)
        score = state.get("score", 0.0)
        threshold = state.get("score_threshold", 0.0)

        if attempts >= MAX_GENERATE_ATTEMPTS:
            logger.warning(f"⚠️ 达到最大重试次数 ({attempts})，强制终止")
            return "finalizer"
        # 依据分数判断是否结束或再次生成
        if score >= threshold:
            logger.success("🏁 任务成功完成！")
            return "finalizer"
        else:
            logger.warning(f"🔄 分数不足 ({score:.3f})，触发第 {attempts + 1} 次重试")
            return "data_profiling" 
    
    def should_decomposite(state: TableState) -> Literal["decompositer", "generator"]:
        """根据任务复杂度决定是否进入分解节点。"""
        is_dag = state.get("is_dag", {})
        is_op = state.get("is_op", {})
        if is_op:
            logger.info("✅ 任务简单，跳过分解节点")
            return "generator"
        elif is_dag:
            logger.info("🔄 任务复杂，进入分解节点")
            return "decompositer"
        else:
            # 用户没有手动指定，基于意图识别结果自动判断
            if state["user_query"].get("is_dag"):
                logger.info("🔄 任务复杂，进入分解节点")
                return "decompositer"
            else:
                logger.info("✅ 任务简单，跳过分解节点")
                return "generator"
                
    # ==============================================================
    # 注册 nodes / edges
    # ==============================================================
    nodes = {
        "intent_understanding": intent_understanding,
        "data_profiling": data_profiling,
        "decompositer": decompositer,
        "generator": generator,
        'evaluator': evaluator,
        "summarizer": summarizer,
        'debugger': debugger,
        'finalizer': finalizer,
        '_end_': lambda state: state,  # 终止节点
    }


    edges = [
        ("intent_understanding", "data_profiling"),
        ("decompositer", "generator"),
        ("generator", "evaluator"),
        ("debugger", "evaluator"),
        ("finalizer", "_end_"),
    ]

    
    conditional_edges = {
        "evaluator": should_debug,
        "summarizer": should_retry,
        "data_profiling": should_decomposite,
    }
    builder.add_nodes(nodes).add_edges(edges).add_conditional_edges(conditional_edges)
    return builder