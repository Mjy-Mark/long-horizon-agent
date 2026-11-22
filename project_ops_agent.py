import json
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from smolagents import OpenAIModel, ToolCallingAgent, tool

from project_ops_env import ProjectOpsEnv


# 两个工具共享一个环境。每次启动脚本时，环境从初始状态开始。
env = ProjectOpsEnv()


@tool
def prepare_dataset(dataset_id: str) -> str:
    """准备指定的数据集。

    Args:
        dataset_id: 数据集编号，可选 A 或 B。
    """
    result = env.prepare_dataset(dataset_id)
    print(f"[工具执行] prepare_dataset({dataset_id!r}) -> {result}", flush=True)
    return json.dumps(result, ensure_ascii=False)


@tool
def configure_experiment(dataset_id: str) -> str:
    """将当前实验配置为使用指定的数据集。该数据集必须已准备好。

    Args:
        dataset_id: 本次实验要使用的数据集编号，可选 A 或 B。
    """
    result = env.configure_experiment(dataset_id)
    print(f"[工具执行] configure_experiment({dataset_id!r}) -> {result}", flush=True)
    return json.dumps(result, ensure_ascii=False)


def snapshot(env: ProjectOpsEnv) -> dict:
    return {
        "dataset_ready": env.dataset_ready.copy(),
        "configured_dataset": env.configured_dataset,
    }


if __name__ == "__main__":
    model = OpenAIModel(
        model_id="longagent-qwen",
        api_base="http://127.0.0.1:8080/v1",
        api_key="local",
        temperature=0,
        max_tokens=256,
    )

    agent = ToolCallingAgent(
        model=model,
        tools=[prepare_dataset, configure_experiment],
        max_steps=5,
    )
    agent.prompt_templates["system_prompt"] = (
        "根据任务和工具反馈选择动作。使用 API 提供的工具调用格式。"
        "完成后调用 final_answer，用中文回答。"
    )

    task = input("请输入任务：")

    runs_dir = Path(__file__).resolve().parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    trace_path = runs_dir / f"react-{uuid4().hex}.json"

    trace = {
        "method": "react",
        "task": task,
        "model_id": model.model_id,
        "smolagents_version": version("smolagents"),
        "max_steps": agent.max_steps,
        "initial_state": snapshot(env),
    }

    try:
        result = agent.run(task, return_full_result=True)
    except Exception as exc:
        trace["agent_state"] = "exception"
        trace["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        trace["steps"] = agent.memory.get_full_steps()
        raise
    else:
        trace["agent_state"] = result.state
        trace["final_answer"] = str(result.output)
        trace["steps"] = result.steps
    finally:
        trace["final_state"] = snapshot(env)
        trace_path.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("[轨迹文件]", trace_path)

    print("[Agent 结束状态]", result.state)
    print("最终回答：", result.output)
    print("[环境终态]", trace["final_state"])
