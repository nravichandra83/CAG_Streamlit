"""Entry point.

    python main.py                      # start the FastAPI server (default)
    python main.py --port 9000 --reload # dev server on another port
    python main.py --ask "question"     # one-shot CLI answer, no server
"""

import argparse

import uvicorn


def ask_once(question: str) -> None:
    from src.api.dependencies import get_cag_service

    service = get_cag_service()
    service.start()
    try:
        result = service.ask(question)
        print(f"\n{result.text}\n")
        print(
            f"[stats] latency={result.latency_seconds:.2f}s  "
            f"input_tokens={result.input_tokens}  "
            f"cached_tokens={result.cached_tokens}  "
            f"total_tokens={result.total_tokens}"
        )
    finally:
        service.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CAG demo: cache-augmented Q&A over the HR leave policy document."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev).")
    parser.add_argument("--ask", dest="question", help="Answer one question from the CLI and exit.")
    args = parser.parse_args()

    if args.question:
        ask_once(args.question)
        return

    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
