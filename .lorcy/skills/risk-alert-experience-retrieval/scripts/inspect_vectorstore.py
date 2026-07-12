#!/usr/bin/env python
"""Inspect document ids and metadata in a local FAISS vector store."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectorstore", required=True)
    parser.add_argument("--embedding-model", default="text-embedding-v4")
    args = parser.parse_args()

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import DashScopeEmbeddings
    except ImportError as exc:
        print(f"missing retrieval dependency: {exc}")
        return 2

    api_key = os.getenv("DASHSCOPE-API-KEY")
    if not api_key:
        print("missing DASHSCOPE-API-KEY")
        return 2

    vectorstore = Path(args.vectorstore)
    embeddings = DashScopeEmbeddings(model=args.embedding_model, dashscope_api_key=api_key)
    store = FAISS.load_local(str(vectorstore), embeddings, allow_dangerous_deserialization=True)

    for doc_id, doc in store.docstore._dict.items():
        preview = doc.page_content.replace("\n", " ")[:160]
        print(f"{doc_id}\t{doc.metadata}\t{preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
