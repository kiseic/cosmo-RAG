from typing import Dict, List, Tuple
from collections import defaultdict

class VotingDecisionAgent:
    """
    投票ベースの決定エージェント
    複数の検索エージェントの結果をReciprocal Rank Fusion (RRF)で統合
    """
    
    def __init__(self, method: str = "rrf", rrf_lambda: float = 60.0):
        self.method = method
        self.rrf_lambda = rrf_lambda
    
    def __call__(self, hits_dict: Dict[str, List[Tuple[str, float]]], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        複数エージェントの検索結果を統合
        
        Args:
            hits_dict: {"text": [(doc_id, score), ...], "image": [(doc_id, score), ...]}
            top_k: 最終的に返す上位k件
            
        Returns:
            統合された結果 [(doc_id, combined_score), ...]
        """
        if self.method == "rrf":
            return self._reciprocal_rank_fusion(hits_dict, top_k)
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")
    
    def _reciprocal_rank_fusion(self, hits_dict: Dict[str, List[Tuple[str, float]]], top_k: int) -> List[Tuple[str, float]]:
        """
        Reciprocal Rank Fusion による結果統合
        
        RRF Score = Σ(1 / (λ + rank_i))
        """
        doc_scores = defaultdict(float)
        
        for agent_name, hits in hits_dict.items():
            if not hits:  # 空の結果をスキップ
                continue
                
            for rank, (doc_id, score) in enumerate(hits, 1):
                # RRF スコア計算
                rrf_score = 1.0 / (self.rrf_lambda + rank)
                doc_scores[doc_id] += rrf_score
        
        # スコア順にソート
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 上位k件を返す
        return sorted_results[:top_k]
    
    def _weighted_fusion(self, hits_dict: Dict[str, List[Tuple[str, float]]], weights: Dict[str, float], top_k: int) -> List[Tuple[str, float]]:
        """
        重み付き融合（将来の拡張用）
        """
        doc_scores = defaultdict(float)
        
        for agent_name, hits in hits_dict.items():
            weight = weights.get(agent_name, 1.0)
            
            for doc_id, score in hits:
                doc_scores[doc_id] += weight * score
        
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]