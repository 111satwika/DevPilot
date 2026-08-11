"""Smallest working example for DevPilot's first LLM integration.

Run from the project root: python -m llm.demo
"""

from llm.client import chat


def main() -> None:
    print("Prompt 1: 'Say hello in exactly 3 words.'")
    print("Response:", chat("Say hello in exactly 3 words."))

    print("\nPrompt 2: 'What is 17 + 25? Answer with just the number.'")
    print("Response:", chat("What is 17 + 25? Answer with just the number."))

    print("\nPrompt 3: asking about this project specifically (should NOT hallucinate)")
    print(
        "Response:",
        chat(
            "Name one MCP server that the project 'DevPilot AI' has built, "
            "in one sentence. If you don't actually know, say you don't know "
            "-- don't make one up."
        ),
    )


if __name__ == "__main__":
    main()
