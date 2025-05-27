#!/usr/bin/env python3
"""
TextRetrievalAgentの__init__を段階的にテスト
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_init_manually():
    """__init__を手動で段階実行"""
    print("=== TextRetrievalAgent.__init__ 手動実行 ===")
    
    # パラメータ設定
    model_name = "all-MiniLM-L6-v2"
    index_path = "indices/test_init.faiss" 
    device = "cpu"
    
    print("1. パラメータ設定...")
    print(f"   model_name: {model_name}")
    print(f"   index_path: {index_path}")
    print(f"   device: {device}")
    
    print("2. 環境変数設定...")
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    print("3. SentenceTransformerインポート...")
    from sentence_transformers import SentenceTransformer
    
    print("4. SentenceTransformerインスタンス化...")
    # ここでハングする可能性
    model = SentenceTransformer(model_name, device=device)
    print("4. ✓ SentenceTransformer作成成功")
    
    print("5. FAISS関連インポート...")
    import faiss, pickle
    
    print("6. インデックス変数初期化...")
    index = None
    doc_metadata = None
    
    print("7. ファイル存在確認...")
    file_exists = os.path.exists(index_path)
    print(f"   {index_path} exists: {file_exists}")
    
    if file_exists:
        print("8. インデックス読み込み...")
        index = faiss.read_index(index_path)
        
        metadata_path = index_path.replace('.faiss', '_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                doc_metadata = pickle.load(f)
    else:
        print("8. 警告: インデックスファイルが見つかりません")
    
    print("9. 手動初期化完了!")
    print(f"   model type: {type(model)}")
    print(f"   index: {index}")
    print(f"   metadata: {doc_metadata}")

if __name__ == "__main__":
    test_init_manually()