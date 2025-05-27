from datasets import load_dataset
from collections import defaultdict
from typing import Dict, List, Iterator, Set

def iter_docs(split="test"):
    """
    REAL-MM-RAG_TechReport から画像ページをドキュメントコーパスとして読み込む
    
    実際のデータ構造:
    - 各行は (クエリ, 回答, 関連画像) のペア
    - 同じ画像が複数のクエリで参照される
    - 画像ページが検索対象のドキュメントコーパス
    
    Returns:
        doc_id: image_filename (重複除去済み)
        page_img: PIL Image
        page_text: OCRテキスト（将来の拡張用、現在は空）
        image_filename: 元ファイル名
    """
    ds = load_dataset("ibm-research/REAL-MM-RAG_TechReport", split=split)
    
    # 画像ファイル名で重複除去（同じ画像が複数クエリで使われるため）
    seen_images: Set[str] = set()
    
    for row in ds:
        image_filename = row["image_filename"]
        
        if image_filename in seen_images:
            continue
            
        seen_images.add(image_filename)
        
        yield {
            "doc_id": image_filename,  # ファイル名をdoc_idとして使用
            "page_img": row["image"],  # PIL Image
            "page_text": image_filename,  # プレースホルダー：ファイル名をテキストとして使用
            "image_filename": image_filename
        }

def iter_queries(split="test"):
    """
    REAL-MM-RAG_TechReport からクエリと正解を読み込む
    
    Returns:
        query: 自然言語クエリ
        answer: 回答テキスト
        relevant_docs: 正解となる画像ファイル名リスト（評価用）
    """
    ds = load_dataset("ibm-research/REAL-MM-RAG_TechReport", split=split)
    
    for row in ds:
        # Noneクエリをスキップ
        if row["query"] is None or not row["query"].strip():
            continue
            
        yield {
            "query_id": str(row["id"]),
            "query": row["query"],
            "answer": row["answer"],
            "relevant_docs": [row["image_filename"]],  # 評価用正解画像
            "image_filename": row["image_filename"]
        }

def iter_query_variants(split="test"):
    """
    リフレーズクエリも含めた全バリエーションを返す（評価拡張用）
    
    Returns:
        各クエリのオリジナル + rephrase_level_1/2/3 バリエーション
    """
    ds = load_dataset("ibm-research/REAL-MM-RAG_TechReport", split=split)
    
    for row in ds:
        base_data = {
            "query_id": str(row["id"]),
            "answer": row["answer"],
            "relevant_docs": [row["image_filename"]],
            "image_filename": row["image_filename"]
        }
        
        # オリジナルクエリ
        if row["query"]:
            yield {**base_data, "query": row["query"], "variant": "original"}
        
        # リフレーズクエリ（レベル1-3）
        for level in [1, 2, 3]:
            rephrase_key = f"rephrase_level_{level}"
            if row.get(rephrase_key):
                yield {
                    **base_data, 
                    "query": row[rephrase_key], 
                    "variant": f"rephrase_level_{level}"
                }

def get_dataset_stats(split="test") -> Dict:
    """
    データセットの統計情報を取得
    """
    ds = load_dataset("ibm-research/REAL-MM-RAG_TechReport", split=split)
    
    # 重複除去した画像数
    unique_images = set()
    total_queries = 0
    
    for row in ds:
        unique_images.add(row["image_filename"])
        if row["query"]:
            total_queries += 1
    
    return {
        "total_records": len(ds),
        "unique_images": len(unique_images),
        "total_queries": total_queries,
        "queries_per_image": total_queries / len(unique_images) if unique_images else 0
    }

def get_image_to_queries_mapping(split="test") -> Dict[str, List[str]]:
    """
    画像ファイル名から関連クエリのマッピングを取得（デバッグ用）
    """
    ds = load_dataset("ibm-research/REAL-MM-RAG_TechReport", split=split)
    
    image_to_queries = defaultdict(list)
    for row in ds:
        if row["query"]:
            image_to_queries[row["image_filename"]].append(row["query"])
    
    return dict(image_to_queries)