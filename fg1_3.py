import hashlib
import subprocess

def format_markdown_and_checksum(file_path: str):
    """
    Formats a Markdown file using `mdformat` and generates a SHA-256 checksum.

    Args:
        file_path (str): Path to the local file.

    Returns:
        str: SHA-256 checksum of the formatted content.
    """
    try:
        # 1. Format the content using `mdformat`
        subprocess.run(["mdformat", file_path], check=True)

        # 2. Read the formatted content
        with open(file_path, "r", encoding="utf-8") as f:
            formatted_content = f.read()

        # 3. Generate SHA-256 checksum
        sha256_hash = hashlib.sha256(formatted_content.encode('utf-8')).hexdigest()

        print(f"SHA-256 Checksum: {sha256_hash}")
        return sha256_hash

    except subprocess.CalledProcessError as e:
        print(f"Error formatting file: {e}")
        return None

    except Exception as ex:
        print(f"Error: {ex}")
        return None

# ✅ Example Usage
file_path = "README.md"   # Replace with the path to your local Markdown file
checksum = format_markdown_and_checksum(file_path)

if checksum:
    print(f"Checksum: {checksum}")
else:
    print("Failed to generate checksum.")
