"""
VLM Generation agent for producing final answers using multimodal models.
"""
import os
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, LlavaNextForConditionalGeneration, LlavaNextProcessor
import torch
from PIL import Image


class VLMGenerationAgent:
    """Generate final answers using Vision-Language Models with retrieved documents and images."""
    
    # サポートするVLMモデルの設定
    SUPPORTED_MODELS = {
        'phi4-multimodal': {
            'model_name': 'microsoft/Phi-4-multimodal',
            'processor_class': 'AutoProcessor',
            'model_class': 'AutoModelForCausalLM'
        },
        'gemma3-4b': {
            'model_name': 'google/gemma-2-4b-it',
            'processor_class': 'AutoTokenizer',
            'model_class': 'AutoModelForCausalLM'
        },
        'qwen-vl-4b': {
            'model_name': 'Qwen/Qwen2-VL-4B-Instruct',
            'processor_class': 'AutoProcessor', 
            'model_class': 'AutoModelForCausalLM'
        },
        'llava-7b': {
            'model_name': 'llava-hf/llava-v1.6-mistral-7b-hf',
            'processor_class': 'LlavaNextProcessor',
            'model_class': 'LlavaNextForConditionalGeneration'
        },
        'lightweight': {
            'model_name': 'lightweight',
            'processor_class': 'template',
            'model_class': 'template'
        }
    }
    
    def __init__(self, model_type: str = "phi4-multimodal", device: str = "cpu"):
        """Initialize VLM generation agent.
        
        Args:
            model_type: Type of VLM to use (phi4-multimodal, gemma3-4b, qwen-vl-4b, llava-7b, lightweight)
            device: Device to run the model on ("cpu", "cuda", or "mps")
        """
        # Apple Silicon対応
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Apple Silicon MPS自動検出
        if device == "cpu" and torch.backends.mps.is_available():
            device = "mps"
            print("Apple Silicon detected: Using MPS for VLM generation")
        elif device == "cpu":
            torch.set_num_threads(2)
            
        self.device = device
        self.model_type = model_type
        
        if model_type not in self.SUPPORTED_MODELS:
            print(f"Warning: {model_type} not supported, falling back to lightweight")
            model_type = 'lightweight'
            
        self.model_config = self.SUPPORTED_MODELS[model_type]
        
        # モデル初期化
        if model_type == 'lightweight':
            self.model = None
            self.processor = None
            self.is_multimodal = False
            print("Using lightweight template-based generation")
        else:
            self._load_model()
    
    def _load_model(self):
        """Load the specified VLM model."""
        model_name = self.model_config['model_name']
        
        try:
            print(f"Loading VLM: {model_name}")
            
            # プロセッサ読み込み
            if self.model_config['processor_class'] == 'AutoProcessor':
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.is_multimodal = True
            elif self.model_config['processor_class'] == 'LlavaNextProcessor':
                self.processor = LlavaNextProcessor.from_pretrained(model_name)
                self.is_multimodal = True
            else:  # AutoTokenizer
                self.processor = AutoTokenizer.from_pretrained(model_name)
                self.is_multimodal = False
            
            # モデル読み込み
            if self.model_config['model_class'] == 'LlavaNextForConditionalGeneration':
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device)
            else:  # AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device)
            
            # PAD token設定
            if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            elif hasattr(self.processor, 'pad_token') and self.processor.pad_token is None:
                self.processor.pad_token = self.processor.eos_token
                
            print(f"VLM loaded successfully: {model_name}")
            
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            print("Falling back to lightweight template")
            self.model = None
            self.processor = None
            self.is_multimodal = False
    
    def generate_answer(self, 
                       query: str, 
                       retrieved_docs: List[Dict], 
                       retrieved_images: Optional[List[Image.Image]] = None,
                       max_length: int = 200) -> str:
        """Generate answer from query, retrieved documents, and images.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved documents with doc_id and score
            retrieved_images: List of PIL Images (for multimodal models)
            max_length: Maximum length of generated answer
            
        Returns:
            Generated answer text
        """
        if self.model is None:
            return self._template_answer(query, retrieved_docs)
        
        # コンテキスト構築
        context = self._build_context(retrieved_docs)
        
        # プロンプト作成
        if self.is_multimodal and retrieved_images:
            return self._generate_multimodal_answer(query, context, retrieved_images, max_length)
        else:
            return self._generate_text_answer(query, context, max_length)
    
    def _generate_multimodal_answer(self, query: str, context: str, images: List[Image.Image], max_length: int) -> str:
        """Generate answer using both text and images."""
        try:
            # マルチモーダルプロンプト作成
            prompt = self._create_multimodal_prompt(query, context)
            
            # 最初の画像を使用（複数画像対応は後日拡張）
            image = images[0] if images else None
            
            if self.model_type in ['phi4-multimodal', 'qwen-vl-4b']:
                # Phi-4/Qwen形式
                inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            elif self.model_type == 'llava-7b':
                # LLaVA形式
                inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
            else:
                # フォールバック
                return self._generate_text_answer(query, context, max_length)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None
                )
            
            # デコード
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # プロンプト部分を除去
            answer = generated_text[len(prompt):].strip()
            
            return answer if answer else "Unable to generate multimodal answer."
            
        except Exception as e:
            print(f"Multimodal generation failed: {e}")
            return self._generate_text_answer(query, context, max_length)
    
    def _generate_text_answer(self, query: str, context: str, max_length: int) -> str:
        """Generate text-only answer."""
        try:
            prompt = self._create_text_prompt(query, context)
            
            # トークン化
            if hasattr(self.processor, 'encode'):
                inputs = self.processor.encode(prompt, return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(prompt, return_tensors="pt").to(self.device)
                inputs = inputs['input_ids']
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.processor.eos_token_id if hasattr(self.processor, 'eos_token_id') else None
                )
            
            # デコード
            if hasattr(self.processor, 'decode'):
                generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            else:
                generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # プロンプト部分を除去
            answer = generated_text[len(prompt):].strip()
            
            return answer if answer else "Unable to generate answer."
            
        except Exception as e:
            print(f"Text generation failed: {e}")
            return self._template_answer(query, [])
    
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
    
    def _create_multimodal_prompt(self, query: str, context: str) -> str:
        """Create prompt for multimodal generation."""
        if self.model_type == 'phi4-multimodal':
            prompt = f"""<|image|>
Context from retrieved documents:
{context}

User Question: {query}

Please analyze the image and the provided context to answer the question comprehensively.

Answer:"""
        elif self.model_type == 'qwen-vl-4b':
            prompt = f"""<|vision_start|><|image_pad|><|vision_end|>
Context: {context}

Question: {query}

Based on the image and context, provide a detailed answer:"""
        else:  # llava-7b
            prompt = f"""USER: <image>
Context: {context}

Question: {query}
ASSISTANT:"""
        
        return prompt
    
    def _create_text_prompt(self, query: str, context: str) -> str:
        """Create prompt for text-only generation."""
        if self.model_type == 'gemma3-4b':
            prompt = f"""<bos><start_of_turn>user
Context: {context}

Question: {query}
<end_of_turn>
<start_of_turn>model
"""
        else:
            prompt = f"""Context:
{context}

Question: {query}

Answer: Based on the provided documents, """
        
        return prompt
    
    def _template_answer(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Generate template-based answer when VLM is not available."""
        if not retrieved_docs:
            return "I couldn't find relevant information to answer your question."
        
        # 最も関連性の高いドキュメント
        top_doc = retrieved_docs[0]
        doc_id = top_doc.get('doc_id', 'a relevant document')
        score = top_doc.get('score', 0.0)
        
        return (f"Based on the retrieved documents (confidence: {score:.3f}), "
                f"the most relevant information can be found in {doc_id}. "
                f"I found {len(retrieved_docs)} potentially relevant documents for your query: {query}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_type': self.model_type,
            'model_name': self.model_config['model_name'] if self.model_config else 'lightweight',
            'is_multimodal': self.is_multimodal,
            'device': self.device,
            'loaded': self.model is not None
        }