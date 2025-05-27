from datasets import load_dataset

def iter_docs(split="test"):
    """
    HuggingFace REAL-MM-RAG_TechReport データセットからドキュメントを読み込む
    """
    ds = load_dataset("ibm-research/REAL-MM-RAG_TechReport", split=split)
    for row in ds:
        yield {
            "page_img": row["pdf_page"], 
            "page_text": row["text"], 
            "doc_id": row["doc_id"]
        }

def iter_queries(split="test"):
    """
    HuggingFace REAL-MM-RAG_TechReport データセットからクエリを読み込む
    """
    ds = load_dataset("ibm-research/REAL-MM-RAG_TechReport", split=split)
    for row in ds:
        yield {
            "query": row["query"], 
            "answers": row["answers"]
        }