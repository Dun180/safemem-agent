from collections.abc import Callable

from agent import run_agent


def run_cli(
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    output_fn("SafeMem Agent")
    output_fn("-------------")

    while True:
        try:
            user_input = input_fn("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("\nBye!")
            break

        if user_input.lower() in {"exit", "quit"}:
            output_fn("Bye!")
            break

        if not user_input:
            continue

        result = run_agent(user_input)

        for event in result.tool_events:
            output_fn(
                f"Tool: {event.name}({event.arguments})"
            )
            output_fn(
                f"Result: {event.result}"
            )

        output_fn(
            f"Agent: {result.answer}"
        )
        output_fn("")


if __name__ == "__main__":
    run_cli()