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
import torch

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ext.adapters.techreport_adapter import iter_docs
from ext.agents.text_retrieval_agent import TextRetrievalAgent
from ext.agents.image_retrieval_agent import ImageRetrievalAgent
from ext.preprocess.vision_encoder import VisionEncoder, load_image_from_bytes

def load_config(config_path: str) -> dict:
    """YAML設定ファイルを読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def generate_image_captions(documents: list, vision_config: dict, device: str) -> list:
    """画像からキャプションを生成してドキュメントに追加"""
    if not vision_config.get('enable', False):
        return documents
    
    print("Generating image captions...")
    # Apple Silicon MPS自動検出
    vision_device = "mps" if device == "cpu" and torch.backends.mps.is_available() else device
    vision_encoder = VisionEncoder(device=vision_device)
    
    for i, doc in enumerate(documents):
        print(f"Processing image {i+1}/{len(documents)}")
        
        # 画像バイトデータからPIL Imageに変換
        image = load_image_from_bytes(doc['page_img'])
        if image is not None:
            caption = vision_encoder.generate_caption(image)
            doc['page_caption'] = caption
        else:
            doc['page_caption'] = "Failed to load image"
    
    return documents

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

def build_image_index(documents: list, agent_config: dict, device: str):
    """画像キャプション検索用インデックスを構築"""
    print(f"Building image index with model: {agent_config['model']}")
    
    # キャプション付きドキュメントを準備
    caption_docs = []
    for doc in documents:
        if 'page_caption' in doc:
            caption_docs.append({
                'text': doc['page_caption'],
                'doc_id': doc['doc_id']
            })
    
    if not caption_docs:
        print("Warning: No captions found. Make sure vision preprocessing is enabled.")
        return
    
    # ImageRetrievalAgent を初期化
    agent = ImageRetrievalAgent(
        model_name=agent_config['model'],
        index_path=agent_config['index'],
        device=device
    )
    
    # インデックス構築
    agent.build_index(caption_docs)
    print(f"Image index saved to: {agent_config['index']}")

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
    
    # デバイス設定
    device = config.get('device', 'cpu')
    if device.startswith('cuda') and not os.system('nvidia-smi > /dev/null 2>&1') == 0:
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # 画像キャプション生成（有効な場合）
    vision_config = config.get('preprocess', {}).get('vision', {})
    documents = generate_image_captions(documents, vision_config, device)
    
    # チャンク化
    chunker_config = config.get('preprocess', {}).get('chunker', {})
    chunked_docs = chunk_documents(documents, chunker_config)
    print(f"Chunked into {len(chunked_docs)} chunks")
    
    # テキストインデックス構築（有効な場合）
    if config.get('agents', {}).get('text', {}).get('enable', False):
        build_text_index(chunked_docs, config['agents']['text'], device)
    
    # 画像インデックス構築（有効な場合）
    if config.get('agents', {}).get('image', {}).get('enable', False):
        build_image_index(chunked_docs, config['agents']['image'], device)
    
    print("Index building completed!")

if __name__ == "__main__":
    main()