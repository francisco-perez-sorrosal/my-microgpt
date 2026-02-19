"""Dataset loading for microgpt: a list of text documents (e.g. names)."""

import os
import random
import urllib.request

NAMES_URL = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
DEFAULT_INPUT_PATH = "input.txt"


def load_docs(path: str = DEFAULT_INPUT_PATH, url: str = NAMES_URL) -> list[str]:
    """Load a dataset of documents, downloading if necessary. Returns shuffled list of strings."""
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    docs = [line.strip() for line in open(path).read().strip().split("\n") if line.strip()]
    random.shuffle(docs)
    print(f"num docs: {len(docs)}")
    return docs


def main() -> None:
    docs = load_docs()
    
    print("first 10 docs:")
    for doc in docs[:10]:
        print(f"  {doc}")


if __name__ == "__main__":
    main()
