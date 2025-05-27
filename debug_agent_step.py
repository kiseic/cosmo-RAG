#!/usr/bin/env python3
"""
TextRetrievalAgentのステップバイステップデバッグ
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_agent_init():
    """エージェント初期化のテスト"""
    print("=== TextRetrievalAgent初期化テスト ===")
    
    from ext.agents.text_retrieval_agent import TextRetrievalAgent
    
    print("1. エージェント初期化開始...")
    agent = TextRetrievalAgent(
        model_name="all-MiniLM-L6-v2",
        index_path="indices/debug_step.faiss",
        device="cpu"
    )
    print("2. エージェント初期化完了")
    
    print("3. エージェント属性確認...")
    print(f"   model: {type(agent.model)}")
    print(f"   index: {agent.index}")
    print(f"   metadata: {agent.doc_metadata}")
    
    return agent

def test_manual_build():
    """手動でインデックス構築をテスト"""
    print("\n=== 手動インデックス構築テスト ===")
    
    agent = test_agent_init()
    
    # テストドキュメント
    test_docs = [
        {
            "doc_id": "doc1",
            "page_text": "IBM FlashSystem page 15 content",
            "image_filename": "page_15.png"
        },
        {
            "doc_id": "doc2", 
            "page_text": "Safeguarded Copy Implementation",
            "image_filename": "page_32.png"
        }
    ]
    
    print("4. ドキュメント準備完了")
    print("5. build_index開始...")
    
    try:
        agent.build_index(test_docs)
        print("6. build_index完了!")
        
        # 検索テスト
        print("7. 検索テスト...")
        results = agent.search("FlashSystem", top_k=2)
        print(f"8. 検索結果: {results}")
        
    except Exception as e:
        print(f"エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_manual_build()