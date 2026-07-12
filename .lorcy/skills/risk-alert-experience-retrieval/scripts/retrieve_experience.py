#!/usr/bin/env python
"""Retrieve historical PRRS risk-alert optimization experience from a FAISS vector store."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectorstore", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model-filter", default=None)
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
    if not vectorstore.exists():
        print(f"vector store not found: {vectorstore}")
        return 1

    embeddings = DashScopeEmbeddings(model=args.embedding_model, dashscope_api_key=api_key)
    store = FAISS.load_local(str(vectorstore), embeddings, allow_dangerous_deserialization=True)
    filter_arg = {"model": args.model_filter} if args.model_filter else None
    docs = store.similarity_search(args.query, k=args.top_k, filter=filter_arg)

    if not docs:
        print("no matching experience found")
        return 0

    for i, doc in enumerate(docs, 1):
        print(f"--- result {i} ---")
        print(doc.page_content)
        if doc.metadata:
            print(f"metadata: {doc.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
