import argparse
import json
from pathlib import Path


def verify_configure(final_state: dict, target_dataset: str) -> dict:
    """检查实验是否已配置为使用任务指定的数据集。"""
    actual_dataset = final_state["configured_dataset"]
    checks = {
        "target_dataset_ready": final_state["dataset_ready"][target_dataset] is True,
        "experiment_uses_target": actual_dataset == target_dataset,
    }

    return {
        "target_dataset": target_dataset,
        "actual_configured_dataset": actual_dataset,
        "checks": checks,
        "task_success": all(checks.values()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证一条实验配置任务轨迹")
    parser.add_argument("trace", type=Path, help="运行生成的 JSON 轨迹")
    parser.add_argument("--target", required=True, choices=["A", "B"])
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    verdict = verify_configure(trace["final_state"], args.target)

    print("原始任务：", trace["task"])
    print("框架结束状态：", trace["agent_state"])
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
