"""Interactive CLI: python main.py --session-id experiment-1."""
import argparse
import logging

from agent import AgentError, ResearchAgent
from config import Settings, configure_logging
from memory import MemoryCapacityError
from multi_agent import MultiAgentResearch


def main() -> None:
    parser = argparse.ArgumentParser(description="Research Agent 科研助手")
    parser.add_argument("--session-id", default="default", help="会话 ID（进程内保存）")
    parser.add_argument("--mode", choices=("single", "multi"), default="single",
                        help="选择单 Agent 或多 Agent 编排")
    args = parser.parse_args()
    try:
        settings = Settings.from_env()
        configure_logging()
        primary = ResearchAgent(settings)
        agent = primary if args.mode == "single" else MultiAgentResearch(primary)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print("Research Agent | /exit 退出 | /clear 清空当前会话")
    if not settings.llm_configured:
        print("尚未配置模型，请填写 .env 中的 OPENAI_BASE_URL、OPENAI_API_KEY、OPENAI_MODEL。")
    try:
        while True:
            try:
                question = input(f"\n[{args.session_id}] 你: ").strip()
                if question in {"/exit", "/quit"}:
                    break
                if question == "/clear":
                    primary.memory.clear(args.session_id)
                    print("当前会话已清空。")
                elif question:
                    print(f"\n科研助手: {agent.chat(question, args.session_id)}")
            except (AgentError, MemoryCapacityError, ValueError) as exc:
                print(f"错误: {exc}")
            except (EOFError, KeyboardInterrupt):
                print("\n已退出。")
                break
            except Exception as exc:
                logging.getLogger(__name__).exception("Unexpected CLI error (%s)", type(exc).__name__)
                print(f"遇到未预期错误（{type(exc).__name__}）。详细堆栈已写入控制台，请把最后 10 行日志发给我。")
    finally:
        agent.close()


if __name__ == "__main__":
    main()
