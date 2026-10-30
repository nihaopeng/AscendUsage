#!/usr/bin/env python3
"""Interactive client for an OpenAI-compatible vLLM server."""

import argparse
import os
import time

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a vLLM server")
    parser.add_argument(
        "--base-url",
        default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:59079/v1"),
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("VLLM_API_KEY", "EMPTY"),
        help="API key, if the server requires one",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VLLM_MODEL"),
        help="Model name; defaults to the first model returned by /v1/models",
    )
    parser.add_argument(
        "--system",
        default="你是一个有帮助的中文助手。",
        help="System prompt",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=131072,
        help="Context window used for usage display (default: 131072)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens generated per reply (default: 4096)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    model = args.model
    if not model:
        models = client.models.list().data
        if not models:
            raise RuntimeError("服务器没有返回可用模型")
        model = models[0].id

    messages: list[dict[str, str]] = [
        {"role": "system", "content": args.system}
    ]

    print(f"已连接：{args.base_url}")
    print(f"使用模型：{model}")
    print(f"上下文窗口：{args.context_window} tokens")
    print("输入 /clear 清空对话，输入 /exit 退出。\n")

    while True:
        try:
            user_text = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "exit", "quit"}:
            print("再见！")
            break
        if user_text.lower() == "/clear":
            messages = [{"role": "system", "content": args.system}]
            print("对话已清空。\n")
            continue

        messages.append({"role": "user", "content": user_text})

        try:
            request_started_at = time.perf_counter()
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                temperature=0.7,
                max_tokens=args.max_tokens,
            )

            print("助手：", end="", flush=True)
            answer_parts: list[str] = []
            first_token_at: float | None = None
            completion_tokens: int | None = None
            total_tokens: int | None = None
            for chunk in stream:
                if chunk.usage is not None:
                    completion_tokens = chunk.usage.completion_tokens
                    total_tokens = chunk.usage.total_tokens
                if not chunk.choices:
                    continue
                text = chunk.choices[0].delta.content or ""
                if text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    answer_parts.append(text)
                    print(text, end="", flush=True)
            request_finished_at = time.perf_counter()
            print()

            elapsed = request_finished_at - request_started_at
            if completion_tokens is not None:
                tokens_per_second = completion_tokens / elapsed if elapsed else 0.0
                speed_text = (
                    f"{completion_tokens} tokens | {elapsed:.2f}s | "
                    f"{tokens_per_second:.2f} tokens/s"
                )
            else:
                speed_text = f"耗时 {elapsed:.2f}s（服务端未返回 token 数）"

            if first_token_at is not None:
                ttft = first_token_at - request_started_at
                speed_text += f" | 首 token {ttft:.2f}s"
            print(f"[{speed_text}]\n")

            if total_tokens is not None:
                context_percent = total_tokens / args.context_window * 100
                print(
                    f"[上下文：{total_tokens}/{args.context_window} tokens "
                    f"({context_percent:.1f}%)]\n"
                )

            messages.append(
                {"role": "assistant", "content": "".join(answer_parts)}
            )
        except Exception as exc:
            messages.pop()
            print(f"请求失败：{exc}\n")


if __name__ == "__main__":
    main()
