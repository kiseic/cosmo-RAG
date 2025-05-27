#!/usr/bin/env python3
"""
REAL-MM-RAG_TechReport データセットのロードテスト
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ext.adapters.techreport_adapter import (
    iter_docs, 
    iter_queries, 
    get_dataset_stats, 
    get_image_to_queries_mapping
)

def test_dataset_loading():
    """データセット読み込みテスト"""
    print("=== データセット統計情報 ===")
    stats = get_dataset_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== ドキュメント（画像ページ）サンプル ===")
    doc_count = 0
    for doc in iter_docs():
        print(f"Doc ID: {doc['doc_id']}")
        print(f"Image filename: {doc['image_filename']}")
        print(f"Page text length: {len(doc['page_text'])}")
        print(f"Image: {type(doc['page_img'])}")
        print("---")
        doc_count += 1
        if doc_count >= 3:  # 最初の3件のみ表示
            break
    
    print(f"\n=== クエリサンプル ===")
    query_count = 0
    for query in iter_queries():
        print(f"Query ID: {query['query_id']}")
        print(f"Query: {query['query'][:100]}...")
        print(f"Answer: {query['answer'][:100]}...")
        print(f"Relevant docs: {query['relevant_docs']}")
        print("---")
        query_count += 1
        if query_count >= 3:  # 最初の3件のみ表示
            break
    
    print(f"\n=== 画像-クエリマッピングサンプル ===")
    mapping = get_image_to_queries_mapping()
    sample_images = list(mapping.keys())[:3]
    
    for img in sample_images:
        queries = mapping[img]
        print(f"Image: {img}")
        print(f"Related queries ({len(queries)}):")
        for i, q in enumerate(queries[:2]):  # 最初の2クエリのみ
            print(f"  {i+1}. {q[:80]}...")
        if len(queries) > 2:
            print(f"  ... and {len(queries)-2} more")
        print("---")

if __name__ == "__main__":
    test_dataset_loading()