#!/usr/bin/env python3
"""Call a local GUI-Actor grounding container using only the Python standard library."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("description")
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--top-k", type=int, choices=range(1, 6), default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.dumps(
        {
            "screenshot_base64": base64.b64encode(args.image.read_bytes()).decode("ascii"),
            "instruction": args.description,
            "top_k": args.top_k,
            "request_id": uuid4().hex,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        f"{args.url.rstrip('/')}/v1/ground",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        result = json.load(response)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
