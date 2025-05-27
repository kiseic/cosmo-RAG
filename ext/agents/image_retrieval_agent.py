"""
Image retrieval agent that searches through image captions using vector similarity.
"""
import os
import json
from typing import List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class ImageRetrievalAgent:
    """Retrieval agent for image caption-based search using FAISS."""
    
    def __init__(self, model_name: str, index_path: str, device: str = "cpu"):
        """Initialize image retrieval agent.
        
        Args:
            model_name: SentenceTransformer model name for embedding
            index_path: Path to FAISS index file
            device: Device to run embedding model on
        """
        # Apple Silicon対応: multithreading問題を回避
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # torch threadsも制限
        try:
            import torch
            torch.set_num_threads(1)
        except:
            pass
            
        self.model = SentenceTransformer(model_name, device=device)
        self.index_path = index_path
        self.index = None
        self.metadata = []
        
        # インデックスとメタデータを読み込み
        self._load_index()
    
    def _load_index(self):
        """Load FAISS index and metadata."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        
        # FAISSインデックス読み込み
        self.index = faiss.read_index(self.index_path)
        
        # メタデータファイルパス生成（柔軟対応）
        if self.index_path.endswith('.faiss'):
            metadata_path = self.index_path.replace('.faiss', '_metadata.json')
        else:
            metadata_path = f"{self.index_path}_metadata.json"
        
        # メタデータ読み込み
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            print(f"Warning: Metadata file not found: {metadata_path}")
            self.metadata = []
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for relevant image captions.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of (doc_id, score) tuples sorted by relevance
        """
        if self.index is None:
            raise RuntimeError("Index not loaded")
        
        # クエリをベクトル化
        query_vector = self.model.encode([query], convert_to_tensor=False)
        query_vector = np.array(query_vector).astype('float32')
        
        # L2正規化（コサイン類似度のため）
        faiss.normalize_L2(query_vector)
        
        # 検索実行
        scores, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.metadata):
                doc_id = self.metadata[idx]['doc_id']
                # FAISS IndexFlatIPはコサイン類似度を返す（-1～1）
                results.append((doc_id, float(score)))
        
        return results
    
    def build_index(self, documents: List[dict]):
        """Build FAISS index from documents.
        
        Args:
            documents: List of documents with 'text' and 'doc_id' fields
        """
        if not documents:
            raise ValueError("No documents provided for indexing")
        
        print(f"Building FAISS index for {len(documents)} captions...")
        
        # テキストを抽出してエンベディング生成
        texts = [doc['text'] for doc in documents]
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # L2正規化（コサイン類似度のため）
        faiss.normalize_L2(embeddings)
        
        # FAISSインデックス作成
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        self.index.add(embeddings)
        
        # メタデータ準備
        self.metadata = [{'doc_id': doc['doc_id']} for doc in documents]
        
        # インデックスフォルダを作成
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # インデックス保存
        faiss.write_index(self.index, self.index_path)
        
        # メタデータ保存
        if self.index_path.endswith('.faiss'):
            metadata_path = self.index_path.replace('.faiss', '_metadata.json')
        else:
            metadata_path = f"{self.index_path}_metadata.json"
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Index built: {len(embeddings)} vectors, dimension={dimension}")
        print(f"Saved to: {self.index_path}")
        print(f"Metadata saved to: {metadata_path}")

    def get_stats(self) -> dict:
        """Get index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'vector_dimension': self.index.d if self.index else 0,
            'metadata_count': len(self.metadata),
            'index_path': self.index_path
        }