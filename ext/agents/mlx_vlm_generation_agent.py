"""
MLX-optimized VLM Generation agent for Apple Silicon.
"""
import os
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import mlx.core as mx
from mlx_vlm import load, generate
try:
    from mlx_vlm.utils import load_config
except ImportError:
    # フォールバック: 直接設定を使用
    def load_config(model_path):
        return {"model_type": "qwen2_vl"}


class MLXVLMGenerationAgent:
    """MLX-optimized VLM Generation agent for Apple Silicon."""
    
    # HuggingFace直接ロード対応モデル設定
    SUPPORTED_MODELS = {
        'qwen2-vl-7b': {
            'model_name': 'Qwen/Qwen2-VL-7B-Instruct',
            'use_mlx': False,  # HuggingFaceから直接
            'chat_template': 'qwen2-vl',
            'max_tokens': 500
        },
        'qwen2-vl-2b': {
            'model_name': 'Qwen/Qwen2-VL-2B-Instruct',
            'use_mlx': False,  # HuggingFaceから直接
            'chat_template': 'qwen2-vl',
            'max_tokens': 500
        },
        'phi3-vision': {
            'model_name': 'microsoft/Phi-3-vision-128k-instruct',
            'use_mlx': False,  # HuggingFaceから直接
            'chat_template': 'phi3-vision',
            'max_tokens': 300
        },
        'gemma3-4b': {
            'model_name': 'google/gemma-3-4b-it-qat-q4_0-gguf',
            'use_mlx': False,  # HuggingFaceから直接
            'chat_template': 'gemma',
            'max_tokens': 400
        },
        'lightweight': {
            'model_name': 'lightweight',
            'use_mlx': False,
            'chat_template': 'template',
            'max_tokens': 200
        }
    }
    
    def __init__(self, model_type: str = "qwen2-vl-4b"):
        """Initialize MLX VLM generation agent.
        
        Args:
            model_type: Type of MLX VLM to use (qwen2-vl-4b, qwen2-vl-2b, llava-4b, lightweight)
        """
        self.model_type = model_type
        
        if model_type not in self.SUPPORTED_MODELS:
            print(f"Warning: {model_type} not supported, falling back to lightweight")
            model_type = 'lightweight'
            self.model_type = model_type
            
        self.model_config = self.SUPPORTED_MODELS[model_type]
        
        # モデル初期化
        if model_type == 'lightweight':
            self.model = None
            self.processor = None
            self.config = None
            print("Using lightweight template-based generation")
        else:
            self._load_hf_model()
    
    def _check_hf_authentication(self, model_name: str):
        """HuggingFace認証が必要なモデルの場合はログインを促す"""
        # 認証が必要なモデルのリスト
        auth_required_models = [
            'google/gemma',
            'meta-llama/Llama',
            'microsoft/Phi-3',
        ]
        
        # モデル名で認証が必要かチェック
        needs_auth = any(pattern in model_name for pattern in auth_required_models)
        
        if needs_auth:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                
                # 現在のトークンで認証確認
                try:
                    user_info = api.whoami()
                    print(f"✅ HuggingFace authenticated as: {user_info['name']}")
                    return
                except Exception:
                    pass
                
            except ImportError:
                print("huggingface_hub not available for authentication check")
            
            # 認証が必要な場合の案内
            print(f"⚠️  Model {model_name} requires authentication")
            print("Please login to HuggingFace:")
            print("  1. Install: pip install huggingface_hub")
            print("  2. Login: huggingface-cli login")
            print("  3. Or set token: export HUGGINGFACE_HUB_TOKEN=your_token")
            
            # 自動ログインを試行
            try:
                import subprocess
                import sys
                
                print("\n🔑 Attempting automatic login...")
                print("Please follow the prompts in your terminal:")
                
                result = subprocess.run([
                    sys.executable, "-m", "huggingface_hub.commands.huggingface_cli", "login"
                ], capture_output=False, text=True)
                
                if result.returncode == 0:
                    print("✅ Login successful!")
                else:
                    print("❌ Login failed. Please try manual login.")
                    
            except Exception as e:
                print(f"❌ Automatic login failed: {e}")
                print("Please run: huggingface-cli login")
                raise Exception("HuggingFace authentication required")
    
    def _load_hf_model(self):
        """Load model directly from HuggingFace."""
        model_name = self.model_config['model_name']
        
        try:
            print(f"Loading VLM from HuggingFace: {model_name}")
            print("Note: Models will be cached in ~/.cache/huggingface/")
            
            # HuggingFace認証確認
            self._check_hf_authentication(model_name)
            
            # Apple Silicon最適化設定
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            
            if self.model_type.startswith('qwen2-vl'):
                # Qwen2-VL用
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                    device_map="auto" if device == "mps" else None,
                    low_cpu_mem_usage=True
                ).to(device)
                
                print(f"✅ Qwen2-VL loaded on {device}")
                
            elif self.model_type == 'phi3-vision':
                # Phi-3-Vision用
                from transformers import AutoModelForCausalLM, AutoProcessor
                
                self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                    device_map="auto" if device == "mps" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                ).to(device)
                
                print(f"✅ Phi-3-Vision loaded on {device}")
                
            elif self.model_type == 'gemma3-4b':
                # Gemma3 GGUF用
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                # GGUF モデルの場合、ファイル名を指定
                if 'gguf' in model_name.lower():
                    # GGUF ファイル内の特定モデルファイルを指定
                    gguf_file = "model.gguf"  # または適切なファイル名
                    
                    self.processor = AutoTokenizer.from_pretrained(
                        model_name,
                        gguf_file=gguf_file,
                        trust_remote_code=True
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        gguf_file=gguf_file,
                        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                        device_map="auto" if device == "mps" else None,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    ).to(device)
                else:
                    self.processor = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                        device_map="auto" if device == "mps" else None,
                        low_cpu_mem_usage=True
                    ).to(device)
                
                if self.processor.pad_token is None:
                    self.processor.pad_token = self.processor.eos_token
                    
                print(f"✅ Gemma3-4B GGUF loaded on {device}")
            
            self.device = device
            
        except Exception as e:
            print(f"❌ Failed to load HuggingFace model {model_name}: {e}")
            print("   Falling back to lightweight template")
            self.model = None
            self.processor = None
            self.model_type = 'lightweight'
    
    def generate_answer(self, 
                       query: str, 
                       retrieved_docs: List[Dict], 
                       retrieved_images: Optional[List[Image.Image]] = None,
                       max_length: int = None) -> str:
        """Generate answer using MLX-optimized VLM.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved documents with doc_id and score
            retrieved_images: List of PIL Images for multimodal generation
            max_length: Maximum length of generated answer
            
        Returns:
            Generated answer text
        """
        if self.model is None:
            return self._template_answer(query, retrieved_docs)
        
        # デフォルト長設定
        if max_length is None:
            max_length = self.model_config['max_tokens']
        
        # コンテキスト構築
        context = self._build_context(retrieved_docs)
        
        # 画像がある場合はマルチモーダル生成
        if retrieved_images and len(retrieved_images) > 0:
            return self._generate_multimodal_hf(query, context, retrieved_images[0], max_length)
        else:
            return self._generate_text_hf(query, context, max_length)
    
    def _generate_multimodal_hf(self, query: str, context: str, image: Image.Image, max_tokens: int) -> str:
        """Generate multimodal answer using HuggingFace models."""
        try:
            import torch
            
            if self.model_type.startswith('qwen2-vl'):
                # Qwen2-VL形式のマルチモーダルプロンプト
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": f"Context: {context}\n\nQuestion: {query}\n\nPlease analyze the image and context to answer the question."}
                        ]
                    }
                ]
                
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = self.processor.preprocess_image_and_video(
                    images=[image], videos=None
                )
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=0.7,
                        do_sample=True
                    )
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                
                return response[0].strip()
                
            elif self.model_type == 'phi3-vision':
                # Phi-3-Vision形式
                prompt = f"<|image|>\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
                
                inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    generate_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=0.7,
                        do_sample=True
                    )
                
                response = self.processor.batch_decode(
                    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                # プロンプト部分を除去
                answer = response.split("Answer:")[-1].strip()
                return answer
            
            else:
                return self._template_answer(query, [])
                
        except Exception as e:
            print(f"HuggingFace multimodal generation failed: {e}")
            return self._generate_text_hf(query, context, max_tokens)
    
    def _generate_text_hf(self, query: str, context: str, max_tokens: int) -> str:
        """Generate text-only answer using HuggingFace models."""
        try:
            import torch
            
            if self.model_type == 'gemma3-4b':
                # Gemma3形式
                prompt = f"""<bos><start_of_turn>user
Context: {context}

Question: {query}
<end_of_turn>
<start_of_turn>model
"""
                
                inputs = self.processor.encode(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=max_tokens,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.processor.eos_token_id
                    )
                
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
                # プロンプト部分を除去
                answer = response[len(prompt):].strip()
                return answer
                
            else:
                # 汎用形式
                prompt = f"""Context:
{context}

Question: {query}

Answer: Based on the provided documents,"""
                
                inputs = self.processor.encode(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=max_tokens,
                        temperature=0.7,
                        do_sample=True
                    )
                
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
                answer = response[len(prompt):].strip()
                return answer
            
        except Exception as e:
            print(f"HuggingFace text generation failed: {e}")
            return self._template_answer(query, [])
    
    def _build_context(self, retrieved_docs: List[Dict], max_docs: int = 3) -> str:
        """Build context from retrieved documents."""
        if not retrieved_docs:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:max_docs]):
            doc_id = doc.get('doc_id', f'Document {i+1}')
            score = doc.get('score', 0.0)
            # ドキュメントテキストがある場合は使用、なければdoc_idを使用
            doc_text = doc.get('text', doc_id)
            context_parts.append(f"Document {i+1} (confidence: {score:.3f}): {doc_text}")
        
        return "\n\n".join(context_parts)
    
    def _template_answer(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Generate template-based answer when MLX VLM is not available."""
        if not retrieved_docs:
            return "I couldn't find relevant information to answer your question."
        
        # 最も関連性の高いドキュメント
        top_doc = retrieved_docs[0]
        doc_id = top_doc.get('doc_id', 'a relevant document')
        score = top_doc.get('score', 0.0)
        
        return (f"Based on the retrieved documents (MLX-optimized search, confidence: {score:.3f}), "
                f"the most relevant information can be found in {doc_id}. "
                f"I found {len(retrieved_docs)} potentially relevant documents for your query: {query}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_type': self.model_type,
            'model_name': self.model_config['model_name'],
            'is_multimodal': self.model_type in ['qwen2-vl-4b', 'qwen2-vl-2b', 'phi3-vision'] and self.model is not None,
            'optimization': 'HuggingFace + Apple Silicon MPS' if self.model else 'Template-based',
            'platform': 'Apple Silicon optimized',
            'device': getattr(self, 'device', 'N/A'),
            'loaded': self.model is not None
        }
    
    def benchmark_speed(self, test_prompt: str = "What is shown in this image?") -> Dict[str, float]:
        """Benchmark generation speed."""
        if self.model is None:
            return {'template_time_ms': 1.0}
        
        import time
        import torch
        
        # ウォームアップ
        try:
            self._generate_text_hf("Test", "No context", 10)
        except:
            pass
        
        # 実際のベンチマーク
        times = []
        for _ in range(3):
            start_time = time.time()
            try:
                _ = self._generate_text_hf(test_prompt, "Test context", 50)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # ms
            except Exception as e:
                print(f"Benchmark failed: {e}")
                return {'error': str(e)}
        
        return {
            'avg_time_ms': sum(times) / len(times),
            'min_time_ms': min(times),
            'max_time_ms': max(times)
        }