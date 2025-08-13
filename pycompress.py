import json
import re
import zlib

def compress_to_string(file_path):
    """
    Compress a Python file into a short, pasteable string
    that can be exactly reconstructed later.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        original_code = f.read()

    # Split into lines and build metadata
    lines = original_code.split("\n")
    metadata = [(len(line) - len(line.lstrip(" \t")), line) for line in lines]

    # Minify: remove comments and indentation for size reduction
    minified_lines = []
    for line in lines:
        stripped = re.sub(r"#.*", "", line.strip())
        if stripped:
            minified_lines.append(stripped)

    payload = {
        "code": "\n".join(minified_lines),
        "meta": metadata
    }

    json_data = json.dumps(payload, separators=(",", ":"))
    compressed = zlib.compress(json_data.encode("utf-8"), level=9)

    # Latin-1 encoding for pasteable text without Base64 bloat
    encoded_str = compressed.decode("latin1", errors="ignore")

    print(f"Original characters: {len(original_code)}")
    print(f"Compressed characters: {len(encoded_str)}\n")
    print("---- COPY BELOW THIS LINE ----")
    print(encoded_str)
    print("---- COPY ABOVE THIS LINE ----")

def decompress_from_string(encoded_str):
    """
    Reconstruct the exact original Python code from a pasteable string.
    """
    compressed = encoded_str.encode("latin1")
    json_data = zlib.decompress(compressed).decode("utf-8")
    payload = json.loads(json_data)

    # Rebuild original code using stored metadata
    meta = payload["meta"]
    restored_lines = [full_line for indent, full_line in meta]

    restored_code = "\n".join(restored_lines)
    return restored_code
