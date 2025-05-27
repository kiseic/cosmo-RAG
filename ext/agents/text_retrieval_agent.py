import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Any

class TextRetrievalAgent:
    """
    テキスト検索エージェント
    Sentence-Transformers + FAISS を使用してテキスト検索を行う
    """
    
    def __init__(self, model_name: str, index_path: str, device: str = "cpu"):
        self.model_name = model_name
        self.index_path = index_path
        self.device = device
        
        # Sentence-Transformers モデルを読み込み
        self.model = SentenceTransformer(model_name, device=device)
        
        # FAISS インデックスとメタデータを読み込み
        self.index = None
        self.doc_metadata = None
        self._load_index()
    
    def _load_index(self):
        """FAISS インデックスとメタデータを読み込む"""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            
            # メタデータファイルも読み込み
            metadata_path = self.index_path.replace('.faiss', '_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.doc_metadata = pickle.load(f)
        else:
            print(f"Warning: Index file {self.index_path} not found")
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """
        ドキュメントからFAISSインデックスを構築
        
        Args:
            documents: [{"page_text": str, "doc_id": str, ...}, ...]
        """
        texts = []
        metadata = []
        
        for doc in documents:
            if doc.get("page_text"):
                texts.append(doc["page_text"])
                metadata.append({
                    "doc_id": doc.get("doc_id", ""),
                    "text": doc["page_text"]
                })
        
        if not texts:
            print("No texts to embed")
            return
        
        # テキストを埋め込みベクトルに変換
        print(f"Embedding {len(texts)} documents...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # FAISS インデックスを構築
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
        
        # ベクトルを正規化（cosine similarity用）
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(embeddings_normalized.astype('float32'))
        
        # メタデータを保存
        self.doc_metadata = metadata
        
        # インデックスとメタデータをディスクに保存
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        
        metadata_path = self.index_path.replace('.faiss', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.doc_metadata, f)
        
        print(f"Index saved to {self.index_path}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        クエリに対してテキスト検索を実行
        
        Args:
            query: 検索クエリ
            top_k: 取得する上位k件
            
        Returns:
            [(doc_id, score), ...] のリスト
        """
        if self.index is None or self.doc_metadata is None:
            print("Index not loaded")
            return []
        
        # クエリを埋め込みベクトルに変換
        query_embedding = self.model.encode([query])
        query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # 検索実行
        scores, indices = self.index.search(query_embedding_normalized.astype('float32'), top_k)
        
        # 結果を整形
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.doc_metadata):
                doc_id = self.doc_metadata[idx]["doc_id"]
                results.append((doc_id, float(score)))
        
        return results