from __future__ import annotations
import argparse
import sys
import os
from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lmscan",
        description="\U0001f50d Detect AI-generated text and fingerprint which LLM wrote it",
    )
    parser.add_argument("input", nargs="?", help="Text string or file path to scan. Use '-' for stdin.")
    parser.add_argument("--file", "-f", help="Path to text file to scan")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--sentences", action="store_true", help="Show per-sentence analysis")
    parser.add_argument("--threshold", type=float, default=0.0, help="Exit with code 1 if AI probability exceeds threshold (for CI)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args(argv)

    # Determine input text
    text = None
    if args.file:
        if not os.path.isfile(args.file):
            print(f"Error: file '{args.file}' not found", file=sys.stderr)
            return 1
        with open(args.file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    elif args.input == "-":
        text = sys.stdin.read()
    elif args.input:
        # Could be a file path or direct text
        if os.path.isfile(args.input):
            with open(args.input, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        else:
            text = args.input
    else:
        # Try stdin
        if not sys.stdin.isatty():
            text = sys.stdin.read()
        else:
            parser.print_help()
            return 1

    if not text or not text.strip():
        print("Error: no text provided", file=sys.stderr)
        return 1

    from .scanner import scan
    from .report import format_report, format_json

    result = scan(text)

    if args.format == "json":
        print(format_json(result))
    else:
        output = format_report(result, show_sentences=args.sentences)
        print(output)

    if args.threshold > 0 and result.ai_probability > args.threshold:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
