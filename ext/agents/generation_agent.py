"""
Generation agent for producing final answers using retrieved documents and VLMs.
"""
import os
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image


class VLMGenerationAgent:
    """Generate final answers using retrieved documents and Vision-Language Models."""
    
    def __init__(self, model_name: str = "microsoft/Phi-4-multimodal", device: str = "cpu"):
        """Initialize generation agent.
        
        Args:
            model_name: HuggingFace model name for text generation
            device: Device to run the model on ("cpu", "cuda", or "mps")
        """
        # Apple Silicon対応
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Apple Silicon MPS自動検出
        if device == "cpu" and torch.backends.mps.is_available():
            device = "mps"
            print("Apple Silicon detected: Using MPS for generation")
        elif device == "cpu":
            torch.set_num_threads(2)
            
        self.device = device
        self.model_name = model_name
        
        # 軽量モデル優先
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                low_cpu_mem_usage=True
            ).to(device)
            
            # PAD token設定
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"Generation model loaded: {model_name}")
            
        except Exception as e:
            print(f"Failed to load {model_name}, falling back to simple template")
            self.model = None
            self.tokenizer = None
    
    def generate_answer(self, query: str, retrieved_docs: List[Dict], max_length: int = 200) -> str:
        """Generate answer from query and retrieved documents.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved documents with doc_id and score
            max_length: Maximum length of generated answer
            
        Returns:
            Generated answer text
        """
        if self.model is None:
            # フォールバック：テンプレートベース回答
            return self._template_answer(query, retrieved_docs)
        
        # コンテキスト構築
        context = self._build_context(retrieved_docs)
        
        # プロンプト作成
        prompt = self._create_prompt(query, context)
        
        try:
            # トークン化
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # デコード
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # プロンプト部分を除去
            answer = generated_text[len(prompt):].strip()
            
            return answer if answer else "Unable to generate answer from retrieved documents."
            
        except Exception as e:
            print(f"Generation failed: {e}")
            return self._template_answer(query, retrieved_docs)
    
    def _build_context(self, retrieved_docs: List[Dict], max_docs: int = 3) -> str:
        """Build context from retrieved documents."""
        if not retrieved_docs:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:max_docs]):
            doc_id = doc.get('doc_id', f'Document {i+1}')
            # ドキュメントテキストがある場合は使用、なければdoc_idを使用
            doc_text = doc.get('text', doc_id)
            context_parts.append(f"Document {i+1}: {doc_text}")
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for generation."""
        prompt = f"""Context:
{context}

Question: {query}

Answer: Based on the provided documents, """
        
        return prompt
    
    def _template_answer(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Generate template-based answer when LLM is not available."""
        if not retrieved_docs:
            return "I couldn't find relevant information to answer your question."
        
        # 最も関連性の高いドキュメント
        top_doc = retrieved_docs[0]
        doc_id = top_doc.get('doc_id', 'a relevant document')
        score = top_doc.get('score', 0.0)
        
        return (f"Based on the retrieved documents (confidence: {score:.3f}), "
                f"the most relevant information can be found in {doc_id}. "
                f"I found {len(retrieved_docs)} potentially relevant documents for your query about: {query}")


class LightweightGenerationAgent:
    """Lightweight generation agent using templates."""
    
    def __init__(self):
        """Initialize lightweight generation agent."""
        pass
    
    def generate_answer(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Generate template-based answer."""
        if not retrieved_docs:
            return "No relevant documents were found for your query."
        
        # 最も関連性の高いドキュメント分析
        top_doc = retrieved_docs[0]
        doc_id = top_doc.get('doc_id', 'Unknown Document')
        score = top_doc.get('score', 0.0)
        
        # クエリの種類を簡単に分析
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'what is', 'what are']):
            answer_prefix = "Based on the retrieved documents,"
        elif any(word in query_lower for word in ['how', 'how to']):
            answer_prefix = "According to the documentation,"
        elif any(word in query_lower for word in ['why', 'why does']):
            answer_prefix = "The documents indicate that"
        else:
            answer_prefix = "The search results suggest that"
        
        # 信頼度に応じたトーン調整
        if score > 0.5:
            confidence = "with high confidence"
        elif score > 0.3:
            confidence = "with moderate confidence"
        else:
            confidence = "with some uncertainty"
        
        answer = (f"{answer_prefix} {confidence}, the answer can be found in "
                 f"{doc_id} (relevance score: {score:.3f}). "
                 f"Total {len(retrieved_docs)} relevant documents were retrieved.")
        
        return answer