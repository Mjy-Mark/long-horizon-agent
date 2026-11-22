"""通过本地 Qwen 查询周五的模拟实验室排期。"""

from smolagents import OpenAIModel, ToolCallingAgent, tool


# None 表示空闲，字符串表示已预约的课题组。
schedule = {
    "动捕室": {"周五上午": None, "周五下午": None},
    "采集设备": {"周五上午": "其他课题组", "周五下午": None},
}


@tool
def get_availability(resource: str) -> str:
    """查询指定实验资源在周五的空闲时段。

    Args:
        resource: 实验资源名称，可选“动捕室”或“采集设备”。
    """
    print(f"[工具执行] get_availability(resource={resource!r})", flush=True)

    slots = schedule.get(resource)
    if slots is None:
        return f"未找到资源“{resource}”。可查询的资源有：动捕室、采集设备。"

    available = [slot for slot, owner in slots.items() if owner is None]
    if not available:
        return f"{resource}周五没有空闲时段。"
    return f"{resource}的空闲时段：{'、'.join(available)}。"


def main() -> None:
    task = input("请输入任务：").strip()
    if not task:
        print("请输入要查询的任务。")
        return

    model = OpenAIModel(
        model_id="longagent-qwen",
        api_base="http://127.0.0.1:8080/v1",
        api_key="local",
        temperature=0,
        max_tokens=256,
    )
    agent = ToolCallingAgent(model=model, tools=[get_availability], max_steps=3)
    # 工具的名称和参数由 API 提供。这里说明何时调用每个工具。
    agent.prompt_templates["system_prompt"] = (
        "你负责查询实验室周五的模拟排期。"
        "查询资源排期时，先调用 get_availability。"
        "拿到工具结果后，调用 final_answer，在 answer 参数中用中文回答。"
        "询问其他日期时，用 final_answer 说明目前只有周五的排期。"
        "使用 API 提供的工具调用格式。"
    )
    result = agent.run(task, return_full_result=True)
    if result.state != "success":
        raise SystemExit("查询未完成，Agent 达到了执行步数上限。")
    print("最终回答：", result.output)


if __name__ == "__main__":
    main()
