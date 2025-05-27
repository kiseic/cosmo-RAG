#!/usr/bin/env python3
"""
TechReport データセット用インデックス構築スクリプト

使用例:
python bench/build_index.py --cfg ext/configs/tech_min.yaml
"""

import argparse
import yaml
import os
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ext.adapters.techreport_adapter import iter_docs
from ext.agents.text_retrieval_agent import TextRetrievalAgent

def load_config(config_path: str) -> dict:
    """YAML設定ファイルを読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def chunk_documents(documents: list, chunker_config: dict) -> list:
    """
    ドキュメントをチャンク化
    現在は page method のみサポート（ページ丸ごと1チャンク）
    """
    if chunker_config.get('method') == 'page':
        # ページ丸ごと1チャンクなので、そのまま返す
        return documents
    else:
        raise ValueError(f"Unsupported chunker method: {chunker_config.get('method')}")

def build_text_index(documents: list, agent_config: dict, device: str):
    """テキスト検索用インデックスを構築"""
    print(f"Building text index with model: {agent_config['model']}")
    
    # TextRetrievalAgent を初期化
    agent = TextRetrievalAgent(
        model_name=agent_config['model'],
        index_path=agent_config['index'],
        device=device
    )
    
    # インデックス構築
    agent.build_index(documents)
    print(f"Text index saved to: {agent_config['index']}")

def main():
    parser = argparse.ArgumentParser(description="Build index for TechReport dataset")
    parser.add_argument("--cfg", required=True, help="Config YAML file path")
    args = parser.parse_args()
    
    # 設定ファイル読み込み
    config = load_config(args.cfg)
    print(f"Loaded config from: {args.cfg}")
    
    # データセット読み込み
    print(f"Loading {config['dataset']} dataset (split: {config['split']})")
    documents = list(iter_docs(split=config['split']))
    print(f"Loaded {len(documents)} documents")
    
    # チャンク化
    chunker_config = config.get('preprocess', {}).get('chunker', {})
    chunked_docs = chunk_documents(documents, chunker_config)
    print(f"Chunked into {len(chunked_docs)} chunks")
    
    # デバイス設定
    device = config.get('device', 'cpu')
    if device.startswith('cuda') and not os.system('nvidia-smi > /dev/null 2>&1') == 0:
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # テキストインデックス構築（有効な場合）
    if config.get('agents', {}).get('text', {}).get('enable', False):
        build_text_index(chunked_docs, config['agents']['text'], device)
    
    # 画像インデックス構築（将来の拡張用）
    if config.get('agents', {}).get('image', {}).get('enable', False):
        print("Image indexing not implemented yet")
        # TODO: ImageRetrievalAgent implementation
    
    print("Index building completed!")

if __name__ == "__main__":
    main()