#!/usr/bin/env python3
"""
最小限のテスト
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic():
    print("=== 基本テスト ===")
    
    # 1. データセットロード
    from ext.adapters.techreport_adapter import get_dataset_stats
    stats = get_dataset_stats()
    print(f"Dataset stats: {stats}")
    
    # 2. Sentence-Transformers単体テスト
    from sentence_transformers import SentenceTransformer
    print("Loading lightweight model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    test_texts = [
        "IBM FlashSystem page 15",
        "Safeguarded Copy Implementation", 
        "FlashSystem backup policy"
    ]
    
    print("Encoding texts...")
    embeddings = model.encode(test_texts, convert_to_numpy=True)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # 3. FAISS単体テスト
    import faiss
    import numpy as np
    
    print("Testing FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # 正規化
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index.add(embeddings_norm.astype('float32'))
    
    # 検索テスト
    query_emb = model.encode(["FlashSystem"], convert_to_numpy=True)
    query_emb_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    
    scores, indices = index.search(query_emb_norm.astype('float32'), 3)
    print(f"Search results: scores={scores[0]}, indices={indices[0]}")
    
    # インデックス保存テスト
    import os
    os.makedirs("indices", exist_ok=True)
    faiss.write_index(index, "indices/simple_test.faiss")
    print("Index saved to indices/simple_test.faiss")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_basic()