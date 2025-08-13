import os
import sys

ENV_VAR = "GITHUB_TOKEN"

def mask(s: str, show: int = 4) -> str:
    if not s:
        return ""
    return s[:show] + "…" + s[-show:] if len(s) > (show * 2) else "…" * len(s)

def main() -> int:
    token = os.environ.get(ENV_VAR)
    if not token:
        print(f"{ENV_VAR} env var not set (set it in your shell/CI; do not hard-code secrets).", file=sys.stderr)
        return 1
    print("Token loaded.")
    print("Length:", len(token))
    print("Preview:", mask(token))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
