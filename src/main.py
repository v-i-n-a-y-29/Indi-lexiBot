"""
main.py — CLI entry point for Indi-lexiBot.

Run from the project root:
    python src/main.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from rag_chain import ask


def main():
    print("=" * 55)
    print("   Indi-lexiBot — Indian Legal Research Assistant")
    print("=" * 55)
    print("Domains: Consumer Protection | Harassment | Traffic")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        print("\nSearching legal documents...")
        ask(question, stream=True)
        print("\n" + "-" * 55 + "\n")


if __name__ == "__main__":
    main()
