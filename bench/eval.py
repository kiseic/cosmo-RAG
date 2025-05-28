#!/usr/bin/env python3
"""
TechReport データセット評価スクリプト

使用例:
python bench/eval.py --cfg ext/configs/tech_min.yaml --stage retrieval
"""

import argparse
import yaml
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ext.adapters.techreport_adapter import iter_docs, iter_queries
from ext.agents.text_retrieval_agent import TextRetrievalAgent
from ext.agents.image_retrieval_agent import ImageRetrievalAgent
from ext.agents.vlm_generation_agent import VLMGenerationAgent
from ext.agents.mlx_vlm_generation_agent import MLXVLMGenerationAgent
from ext.decision.voting_agent import VotingDecisionAgent

def load_config(config_path: str) -> dict:
    """YAML設定ファイルを読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def calculate_recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
    """Recall@K を計算"""
    if not relevant_docs:
        return 0.0
    
    retrieved_set = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)
    
    return len(retrieved_set & relevant_set) / len(relevant_set)

def calculate_ndcg_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
    """NDCG@K を計算"""
    if not relevant_docs:
        return 0.0
    
    # 簡略化されたNDCG実装（バイナリ関連性）
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs[:k]):
        if doc_id in relevant_docs:
            dcg += 1.0 / (i + 2)  # log2(i+2)の代わりに(i+2)で近似
    
    # 理想的なDCG
    idcg = sum(1.0 / (i + 2) for i in range(min(len(relevant_docs), k)))
    
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_retrieval(config: dict) -> Dict[str, Any]:
    """検索段階の評価を実行"""
    print("Starting retrieval evaluation...")
    
    # エージェントを初期化
    agents = {}
    device = config.get('device', 'cpu')
    
    if config.get('agents', {}).get('text', {}).get('enable', False):
        text_config = config['agents']['text']
        agents['text'] = TextRetrievalAgent(
            model_name=text_config['model'],
            index_path=text_config['index'],
            device=device
        )
        print(f"Loaded text agent: {text_config['model']}")
    
    if config.get('agents', {}).get('image', {}).get('enable', False):
        image_config = config['agents']['image']
        agents['image'] = ImageRetrievalAgent(
            model_name=image_config['model'],
            index_path=image_config['index'],
            device=device
        )
        print(f"Loaded image agent: {image_config['model']}")
    
    # VLM生成エージェントを初期化
    generation_agent = None
    if config.get('generation', {}).get('enable', False):
        gen_config = config['generation']
        
        # MLXフレームワーク使用の場合
        if gen_config.get('framework') == 'mlx':
            generation_agent = MLXVLMGenerationAgent(
                model_type=gen_config.get('model_type', 'qwen2-vl-4b')
            )
            print(f"Loaded MLX VLM generation agent: {gen_config.get('model_type', 'qwen2-vl-4b')}")
        else:
            # 通常のHuggingFace VLM
            generation_agent = VLMGenerationAgent(
                model_type=gen_config.get('model_type', 'lightweight'),
                device=device
            )
            print(f"Loaded VLM generation agent: {gen_config.get('model_type', 'lightweight')}")
    
    # 決定エージェントを初期化
    decision_config = config.get('decision', {})
    if decision_config.get('strategy') == 'voting':
        voting_config = decision_config.get('voting', {})
        decision_agent = VotingDecisionAgent(
            method=voting_config.get('method', 'rrf'),
            rrf_lambda=voting_config.get('lambda', 60.0)
        )
        print("Loaded voting decision agent")
    
    # クエリデータを読み込み
    queries = list(iter_queries(split=config['split']))
    print(f"Loaded {len(queries)} queries")
    
    # 評価結果を格納
    results = []
    
    for i, query_data in enumerate(queries):
        if i % 10 == 0:
            print(f"Processing query {i+1}/{len(queries)}")
        
        query = query_data['query']
        # 新しいデータ形式では relevant_docs が直接提供される
        relevant_docs = query_data.get('relevant_docs', [])
        
        # 各エージェントで検索実行
        hits_dict = {}
        for agent_name, agent in agents.items():
            agent_config = config['agents'][agent_name]
            hits = agent.search(query, top_k=agent_config['top_k'])
            hits_dict[agent_name] = hits
        
        # 決定エージェントで結果を統合
        if len(agents) > 1 and 'decision_agent' in locals():
            final_hits = decision_agent(hits_dict, top_k=50)  # 評価用に多めに取得
        elif len(agents) == 1:
            final_hits = list(hits_dict.values())[0]  # 単一エージェントの場合
        else:
            final_hits = []
        
        # 検索結果のdoc_idリストを作成
        retrieved_doc_ids = [doc_id for doc_id, score in final_hits]
        
        # VLM生成（有効な場合）
        generated_answer = None
        if generation_agent:
            # 検索結果をドキュメント形式に変換
            retrieved_docs_for_gen = [
                {'doc_id': doc_id, 'score': score} 
                for doc_id, score in final_hits[:3]
            ]
            
            generated_answer = generation_agent.generate_answer(
                query=query,
                retrieved_docs=retrieved_docs_for_gen,
                max_length=config.get('generation', {}).get('max_length', 150)
            )
        
        # メトリクス計算
        query_result = {
            'query': query,
            'relevant_docs': relevant_docs,
            'retrieved_docs': retrieved_doc_ids,
            'generated_answer': generated_answer,
            'metrics': {}
        }
        
        # 設定されたメトリクスを計算
        for metric in config.get('metrics', []):
            if metric.startswith('recall@'):
                k = int(metric.split('@')[1])
                query_result['metrics'][metric] = calculate_recall_at_k(retrieved_doc_ids, relevant_docs, k)
            elif metric.startswith('ndcg@'):
                k = int(metric.split('@')[1])
                query_result['metrics'][metric] = calculate_ndcg_at_k(retrieved_doc_ids, relevant_docs, k)
        
        results.append(query_result)
    
    # 平均メトリクスを計算
    avg_metrics = {}
    if results:
        for metric in config.get('metrics', []):
            scores = [r['metrics'].get(metric, 0.0) for r in results]
            avg_metrics[metric] = sum(scores) / len(scores)
    
    return {
        'config': config,
        'results': results,
        'average_metrics': avg_metrics,
        'num_queries': len(queries),
        'timestamp': datetime.now().isoformat()
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate TechReport dataset")
    parser.add_argument("--cfg", required=True, help="Config YAML file path")
    parser.add_argument("--stage", default="retrieval", choices=["retrieval", "decision"], 
                        help="Evaluation stage")
    args = parser.parse_args()
    
    # 設定ファイル読み込み
    config = load_config(args.cfg)
    print(f"Loaded config from: {args.cfg}")
    
    # 評価実行
    if args.stage == "retrieval":
        eval_results = evaluate_retrieval(config)
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")
    
    # 結果を表示
    print("\n=== Evaluation Results ===")
    for metric, score in eval_results['average_metrics'].items():
        print(f"{metric}: {score:.4f}")
    
    # 結果をJSONファイルに保存
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"logs/eval_{args.stage}_{timestamp}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()