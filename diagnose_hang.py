#!/usr/bin/env python3
"""
ハング原因の特定
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_step_1_imports():
    """ステップ1: インポートのテスト"""
    print("=== ステップ1: インポートテスト ===")
    
    try:
        print("1.1 基本インポート...")
        import os
        import pickle
        import numpy as np
        import faiss
        print("1.1 ✓ 基本インポート成功")
        
        print("1.2 sentence-transformersインポート...")
        from sentence_transformers import SentenceTransformer
        print("1.2 ✓ sentence-transformersインポート成功")
        
        print("1.3 adaptersインポート...")
        from ext.adapters.techreport_adapter import iter_docs
        print("1.3 ✓ adaptersインポート成功")
        
        print("1.4 agentsインポート...")
        # ここでハングする可能性をチェック
        print("1.4.1 agents moduleインポート開始...")
        from ext.agents import text_retrieval_agent
        print("1.4.1 ✓ agents moduleインポート成功")
        
        print("1.4.2 TextRetrievalAgentクラスインポート開始...")
        from ext.agents.text_retrieval_agent import TextRetrievalAgent
        print("1.4.2 ✓ TextRetrievalAgentクラスインポート成功")
        
        return True
        
    except Exception as e:
        print(f"❌ インポートエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step_2_class_definition():
    """ステップ2: クラス定義の確認"""
    print("\n=== ステップ2: クラス定義テスト ===")
    
    try:
        from ext.agents.text_retrieval_agent import TextRetrievalAgent
        print("2.1 ✓ クラス読み込み成功")
        
        print("2.2 クラス属性確認...")
        print(f"   - __init__: {hasattr(TextRetrievalAgent, '__init__')}")
        print(f"   - build_index: {hasattr(TextRetrievalAgent, 'build_index')}")
        print(f"   - search: {hasattr(TextRetrievalAgent, 'search')}")
        
        return True
        
    except Exception as e:
        print(f"❌ クラス定義エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step_3_init_params():
    """ステップ3: 初期化パラメータのテスト"""
    print("\n=== ステップ3: 初期化パラメータテスト ===")
    
    try:
        print("3.1 パラメータ準備...")
        model_name = "all-MiniLM-L6-v2"
        index_path = "indices/diagnose_test.faiss"
        device = "cpu"
        
        print(f"   - model_name: {model_name}")
        print(f"   - index_path: {index_path}")
        print(f"   - device: {device}")
        
        return True
        
    except Exception as e:
        print(f"❌ パラメータエラー: {e}")
        return False

def test_step_4_sentence_transformer_direct():
    """ステップ4: SentenceTransformerの直接テスト"""
    print("\n=== ステップ4: SentenceTransformer直接テスト ===")
    
    try:
        print("4.1 環境変数設定...")
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        print("4.2 SentenceTransformer初期化開始...")
        from sentence_transformers import SentenceTransformer
        
        # ここでハングするかもしれない
        print("4.2.1 モデル読み込み開始...")
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        print("4.2.1 ✓ モデル読み込み成功")
        
        print("4.3 エンコードテスト...")
        result = model.encode(["test"], convert_to_numpy=True)
        print(f"4.3 ✓ エンコード成功: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ SentenceTransformerエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step_5_agent_init_attempt():
    """ステップ5: エージェント初期化の試行"""
    print("\n=== ステップ5: エージェント初期化試行 ===")
    
    try:
        print("5.1 環境設定...")
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        print("5.2 TextRetrievalAgentインポート...")
        from ext.agents.text_retrieval_agent import TextRetrievalAgent
        
        print("5.3 エージェント初期化開始...")
        # この行でハングする可能性が高い
        agent = TextRetrievalAgent(
            model_name="all-MiniLM-L6-v2",
            index_path="indices/diagnose_test.faiss", 
            device="cpu"
        )
        print("5.3 ✓ エージェント初期化成功")
        
        return True
        
    except Exception as e:
        print(f"❌ エージェント初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """診断実行"""
    print("TextRetrievalAgentハング原因診断開始\n")
    
    steps = [
        test_step_1_imports,
        test_step_2_class_definition,
        test_step_3_init_params,
        test_step_4_sentence_transformer_direct,
        test_step_5_agent_init_attempt
    ]
    
    for i, test_func in enumerate(steps, 1):
        print(f"--- ステップ {i} 開始 ---")
        success = test_func()
        
        if not success:
            print(f"❌ ステップ {i} で失敗。ここが問題箇所です。")
            break
        else:
            print(f"✓ ステップ {i} 成功")
            
        print()
    
    print("診断完了")

if __name__ == "__main__":
    main()