#!/usr/bin/env python3
"""
TextRetrievalAgentのエンドツーエンドテスト
"""

import sys
from pathlib import Path
import itertools

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ext.adapters.techreport_adapter import iter_docs, iter_queries
from ext.agents.text_retrieval_agent import TextRetrievalAgent

def test_text_agent():
    """TextRetrievalAgentのテスト"""
    print("=== TextRetrievalAgent テスト ===")
    
    # 小規模データでテスト
    docs = list(itertools.islice(iter_docs(), 10))
    print(f"Using {len(docs)} documents")
    
    # エージェント初期化
    agent = TextRetrievalAgent(
        model_name="all-MiniLM-L6-v2",
        index_path="indices/agent_test.faiss",
        device="cpu"
    )
    
    # インデックス構築
    print("Building index...")
    agent.build_index(docs)
    print("Index built successfully!")
    
    # クエリテスト
    queries = list(itertools.islice(iter_queries(), 3))
    
    for i, query_data in enumerate(queries):
        print(f"\n--- Query {i+1} ---")
        query = query_data['query']
        relevant_docs = query_data['relevant_docs']
        
        print(f"Query: {query[:80]}...")
        print(f"Expected: {relevant_docs}")
        
        # 検索実行
        results = agent.search(query, top_k=5)
        print("Retrieved:")
        for j, (doc_id, score) in enumerate(results):
            marker = "✓" if doc_id in relevant_docs else " "
            print(f"  {j+1}. {marker} {doc_id} (score: {score:.3f})")

def test_full_pipeline():
    """軽量設定での全パイプラインテスト"""
    print("\n=== 軽量設定での全パイプラインテスト ===")
    
    import subprocess
    import sys
    
    # 軽量設定でインデックス構築
    cmd = [
        sys.executable, 
        "bench/build_index.py", 
        "--cfg", "ext/configs/tech_min_light.yaml"
    ]
    
    print("Running: " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    if result.returncode == 0:
        print("Index building completed successfully!")
        print("STDOUT:", result.stdout[-500:])  # 最後の500文字
    else:
        print("Index building failed!")
        print("STDERR:", result.stderr[-500:])
        
    # インデックスファイルの確認
    import os
    index_path = "indices/tech_text_faiss_light"
    if os.path.exists(index_path):
        size = os.path.getsize(index_path)
        print(f"Index file created: {index_path} ({size} bytes)")
    else:
        print(f"Index file not found: {index_path}")

if __name__ == "__main__":
    test_text_agent()
    test_full_pipeline()