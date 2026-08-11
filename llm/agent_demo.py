"""Test the tool-selecting agent with a real free-text request that
genuinely needs both servers to answer -- no hardcoded plan, unlike
Stage 9's planner.py. Expect this to take a few minutes: this model
runs at roughly 50-60 seconds per turn on this hardware.

Run from the project root: python -m llm.agent_demo
"""

import asyncio

from llm.agent import ask


async def main() -> None:
    # Explicitly decomposed into two numbered parts -- the first attempt at
    # this as one compound question got only the database half answered;
    # this is a real, documented adjustment, not a rerun hoping for luck.
    question = (
        "Answer both of these using the available tools -- don't guess, and "
        "answer both, not just one:\n"
        "1. What Python dependencies does this project use? (check requirements.txt)\n"
        "2. What tables exist in this project's demo database?"
    )
    print(f"Question: {question}\n")

    answer = await ask(question)

    print("\n" + "=" * 60)
    print("Final answer:")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
