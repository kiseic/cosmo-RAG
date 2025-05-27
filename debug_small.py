#!/usr/bin/env python3
"""
小規模データでのデバッグテスト
"""

import multiprocessing as mp
mp.set_start_method("fork", force=True)  # macOSでのsemaphore警告回避

import sys
from pathlib import Path
import itertools

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ext.adapters.techreport_adapter import iter_docs
from ext.agents.text_retrieval_agent import TextRetrievalAgent

def debug_small():
    """最小構成でのデバッグ"""
    print("=== 小規模デバッグテスト ===")
    
    # 最初の5ドキュメントのみ
    docs = list(itertools.islice(iter_docs(), 5))
    print(f"Loaded {len(docs)} documents")
    
    for i, doc in enumerate(docs):
        print(f"Doc {i}: {doc['doc_id']}")
        print(f"  Text: '{doc['page_text']}'")
    
    print("\n=== TextRetrievalAgent初期化 ===")
    agent = TextRetrievalAgent(
        model_name="all-MiniLM-L6-v2",
        index_path="indices/debug_test.faiss",
        device="cpu"
    )
    print("Agent initialized")
    
    print("\n=== インデックス構築 ===")
    try:
        agent.build_index(docs)
        print("Index built successfully!")
        
        # 検索テスト
        print("\n=== 検索テスト ===")
        results = agent.search("FlashSystem", top_k=3)
        print(f"Search results: {len(results)}")
        for doc_id, score in results:
            print(f"  {doc_id}: {score:.3f}")
            
    except Exception as e:
        print(f"Error during indexing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_small()