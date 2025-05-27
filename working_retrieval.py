#!/usr/bin/env python3
"""
動作確認済みの実装でTechReportデータセットのテスト
"""

import sys
from pathlib import Path
import itertools
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ext.adapters.techreport_adapter import iter_docs, iter_queries

def test_working_retrieval():
    """動作確認済みの実装でテスト"""
    print("=== 動作確認済み実装でのテスト ===")
    
    # 少量のドキュメントで検証
    docs = list(itertools.islice(iter_docs(), 20))
    print(f"Using {len(docs)} documents")
    
    # テキスト抽出
    texts = [doc['page_text'] for doc in docs]
    doc_ids = [doc['doc_id'] for doc in docs]
    
    print("Sample texts:")
    for i, text in enumerate(texts[:3]):
        print(f"  {i}: {text}")
    
    # モデル読み込み
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 埋め込み生成
    print("Generating embeddings...")
    embeddings = model.encode(texts, convert_to_numpy=True)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # FAISS インデックス構築
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # 正規化してインデックスに追加
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index.add(embeddings_norm.astype('float32'))
    
    # メタデータ保存
    metadata = [{"doc_id": doc_id, "text": text} for doc_id, text in zip(doc_ids, texts)]
    
    # インデックス保存
    os.makedirs("indices", exist_ok=True)
    index_path = "indices/working_test.faiss"
    metadata_path = "indices/working_test_metadata.pkl"
    
    faiss.write_index(index, index_path)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Index saved to {index_path}")
    
    # クエリテスト
    print("\n=== クエリテスト ===")
    queries = list(itertools.islice(iter_queries(), 5))
    
    for i, query_data in enumerate(queries):
        query = query_data['query']
        relevant_docs = query_data['relevant_docs']
        
        print(f"\nQuery {i+1}: {query[:80]}...")
        print(f"Expected: {relevant_docs}")
        
        # クエリ検索
        query_emb = model.encode([query], convert_to_numpy=True)
        query_emb_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        
        scores, indices = index.search(query_emb_norm.astype('float32'), 5)
        
        print("Retrieved:")
        for j, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(metadata):
                doc_id = metadata[idx]["doc_id"]
                marker = "✓" if doc_id in relevant_docs else " "
                print(f"  {j+1}. {marker} {doc_id} (score: {score:.3f})")

def test_recall_calculation():
    """Recall計算のテスト"""
    print("\n=== Recall計算テスト ===")
    
    # 簡単な例
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc2", "doc5", "doc6"]
    
    for k in [1, 3, 5]:
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        recall = len(retrieved_k & relevant_set) / len(relevant_set)
        print(f"Recall@{k}: {recall:.3f}")

if __name__ == "__main__":
    test_working_retrieval()
    test_recall_calculation()