# -*- coding: utf-8 -*-
"""
模型加载与大模型客户端封装

支持三种调用方式：
1）本地 HuggingFace 模型（local）
2）内部 OpenAI 兼容 API（internal_api / openai_compatible）
3）云端 OpenAI 兼容 API（Qwen/DeepSeek 等）
"""

import os
import json
from typing import List, Optional, Literal, Dict, Any

import requests

try:
    # 对于本地模型后端需要这些依赖；API 模式可以在未安装的情况下正常工作
    import torch  # type: ignore
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # type: ignore
except ImportError:  # pragma: no cover - 仅在缺少本地依赖时触发
    torch = None  # type: ignore
    AutoTokenizer = AutoModelForCausalLM = pipeline = None  # type: ignore


BackendType = Literal["local", "internal_api", "qwen", "deepseek", "openai_compatible"]


class ModelLoader:
    """统一的大模型调用封装"""

    def __init__(
        self,
        backend: BackendType = "local",
        model_path: Optional[str] = None,
        device: str = "cuda",
        # 多GPU相关参数
        device_map: Optional[Any] = None,  # "auto"、None 或字典，用于自动模型并行
        gpu_ids: Optional[str] = None,  # 指定使用的GPU，如 "0,1,2,3" 或 "0-7"
        max_memory: Optional[Dict[int, str]] = None,  # 每个GPU的最大内存，如 {0: "20GiB", 1: "20GiB"}
        # OpenAI 兼容 API 参数
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_context: int = 8192,
        max_tokens: int = 1024,
    ):
        """
        初始化模型加载器 / API 客户端

        Args:
            backend: 调用后端类型：
                     - 'local'：本地 HuggingFace 模型
                     - 'internal_api'：内部 OpenAI 兼容 API（如截图中的内部模型）
                     - 'qwen'：阿里 Qwen（DashScope 兼容模式）
                     - 'deepseek'：DeepSeek API
                     - 'openai_compatible'：任意 OpenAI 兼容 API
            model_path: 本地模型路径或 HuggingFace 模型名称（仅 backend='local' 时使用）
            device: 设备类型。支持：'cuda'（默认）或 'cpu'
            device_map: 模型并行策略。支持：
                       - None：单GPU模式，模型加载到指定device
                       - "auto"：自动将模型切分到多个GPU（推荐用于大模型）
                       - 字典：手动指定每层在哪个GPU，如 {"": 0, "layers.0": 0, "layers.1": 1}
            gpu_ids: 指定使用的GPU ID，格式如 "0,1,2,3" 或 "0-7"。当device_map="auto"时，
                    可以通过设置环境变量 CUDA_VISIBLE_DEVICES 或使用此参数来限制使用的GPU
            max_memory: 每个GPU的最大内存限制，如 {0: "20GiB", 1: "20GiB"}。
                       当device_map="auto"时，用于控制内存分配
            api_key: OpenAI 兼容 API 的 key（可为空，某些后端会从环境变量读取）
            base_url: OpenAI 兼容 API 的 base url，例如 'http://10.10.0.102:7865/v1'
            model_name: 模型名称，例如 'Fusion2-chat-v2.0' / 'qwen-plus' / 'deepseek-chat'
            temperature: 采样温度
            max_context: 模型最大上下文长度（MODEL_MAX_CONTEXT）
            max_tokens: 默认最大生成 tokens（DEFAULT_MAX_TOKENS）
        """
        self.backend: BackendType = backend
        self.model_path = model_path
        
        # 解析设备参数：只支持 "cuda" 和 "cpu"，默认 "cuda"
        if device.lower() == "cpu":
            self.device = "cpu"
        else:
            # 默认使用 cuda
            print(f"警告: 不支持的设备参数 '{device}'，将使用默认值 'cuda'")
            self.device = "cuda"
        
        # 多GPU相关配置
        self.device_map = device_map
        self.gpu_ids = gpu_ids
        self.max_memory = max_memory
        
        # 如果指定了gpu_ids，设置CUDA_VISIBLE_DEVICES环境变量
        if gpu_ids and self.device != "cpu":
            gpu_id_list = self._parse_gpu_spec(gpu_ids)
            if gpu_id_list:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_id_list))
                print(f"已设置 CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
                # 注意：设置CUDA_VISIBLE_DEVICES后，GPU编号会重新映射，device_map中的编号需要相应调整

        # API 相关配置
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_context = max_context
        self.default_max_tokens = max_tokens

        # 本地模型句柄
        self.model = None
        self.tokenizer = None
        self.pipeline = None

        # 根据 backend 初始化
        if self.backend == "local":
            if not self.model_path:
                raise ValueError("本地模型模式下必须提供 model_path")
            self._load_local_model()
        else:
            self._init_api_client()

    def _parse_gpu_spec(self, gpu_spec: str) -> List[int]:
        """
        解析GPU规格字符串，支持多种格式：
        - "0" -> [0]
        - "0,1,2,3" -> [0, 1, 2, 3]
        - "0-7" -> [0, 1, 2, 3, 4, 5, 6, 7]
        - "0,2,4-6" -> [0, 2, 4, 5, 6]
        """
        gpu_ids = []
        parts = gpu_spec.split(",")
        
        for part in parts:
            part = part.strip()
            if "-" in part:
                # 范围格式：0-7
                try:
                    start, end = map(int, part.split("-"))
                    gpu_ids.extend(range(start, end + 1))
                except (ValueError, IndexError):
                    continue
            else:
                # 单个编号
                try:
                    gpu_id = int(part)
                    if gpu_id not in gpu_ids:
                        gpu_ids.append(gpu_id)
                except ValueError:
                    continue
        
        # 去重并排序
        gpu_ids = sorted(list(set(gpu_ids)))
        return gpu_ids

    # === 本地模型相关 ===

    def _load_local_model(self):
        """加载本地 HuggingFace 模型，支持多GPU并行/模型切分"""
        print(f"正在加载本地模型: {self.model_path}")
        print(f"设备: {self.device}")

        if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError(
                "本地模型模式需要安装 'torch' 和 'transformers'，"
                "但当前环境中未检测到这些依赖。请先安装相关依赖，"
                "或改用 deepseek / qwen / internal_api / openai_compatible 等 API 模式。"
            )

        # 检查GPU是否可用
        is_cuda = self.device != "cpu"
        if is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA不可用，但指定了CUDA设备。请使用 'cpu' 或确保CUDA已正确安装。")
            num_gpus = torch.cuda.device_count()
            print(f"检测到 {num_gpus} 个GPU设备")
            if num_gpus > 1:
                print(f"GPU设备列表: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")

        try:
            # 尝试加载 HuggingFace 格式的模型
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            
            # 确定device_map策略
            if is_cuda:
                if self.device_map == "auto":
                    # 自动模型并行：让transformers自动切分模型到多个GPU
                    device_map_value = "auto"
                    print("使用 device_map='auto' 进行自动模型并行（多GPU切分）")
                elif isinstance(self.device_map, dict):
                    # 手动指定device_map
                    device_map_value = self.device_map
                    print(f"使用手动指定的 device_map: {device_map_value}")
                else:
                    # 单GPU模式：device_map=None，加载后手动移动到指定GPU
                    device_map_value = None
                    print(f"单GPU模式：将模型加载到 {self.device}")
            else:
                device_map_value = None
            
            # 准备加载参数
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if is_cuda else torch.float32,
                "low_cpu_mem_usage": True,  # 优化内存使用，对大模型很重要
            }
            
            # 设置device_map
            if device_map_value is not None:
                load_kwargs["device_map"] = device_map_value
                # 如果指定了max_memory，也传入
                if self.max_memory is not None:
                    load_kwargs["max_memory"] = self.max_memory
                    print(f"设置GPU内存限制: {self.max_memory}")
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **load_kwargs
            )

            # 只有在不使用device_map时才手动移动模型
            if device_map_value is None:
                if is_cuda:
                    print(f"正在将模型移动到 {self.device}...")
                    self.model = self.model.to(self.device)
                    actual_device = next(self.model.parameters()).device
                    print(f"模型已移动到设备: {actual_device}")
                else:
                    self.model = self.model.to(self.device)
            else:
                # 使用device_map时，模型已经自动分布在多个GPU上
                # 打印模型分布信息
                if hasattr(self.model, "hf_device_map"):
                    print(f"模型已自动切分到多个GPU:")
                    for layer_name, device in self.model.hf_device_map.items():
                        print(f"  {layer_name}: {device}")
                else:
                    # 尝试从模型参数推断设备分布
                    device_set = set()
                    for name, param in self.model.named_parameters():
                        device_set.add(str(param.device))
                    if len(device_set) > 1:
                        print(f"模型分布在多个设备上: {device_set}")
                    else:
                        print(f"模型设备: {list(device_set)[0] if device_set else 'unknown'}")

            self.model.eval()
            print("本地模型加载成功")

        except Exception as e:
            print(f"警告: 使用 transformers 加载失败: {e}")
            
            # 如果使用了device_map，pipeline方式不支持多GPU，直接报错
            if self.device_map == "auto" or isinstance(self.device_map, dict):
                raise RuntimeError(
                    f"使用 device_map 时无法回退到 pipeline 方式（pipeline不支持多GPU）。"
                    f"原始错误: {e}"
                )
            
            print("尝试使用 pipeline 方式加载（单GPU模式）...")

            try:
                # pipeline方式（仅支持单GPU）
                if is_cuda:
                    pipeline_device = 0  # 使用第一个GPU
                else:
                    pipeline_device = -1
                
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_path,
                    device=pipeline_device,
                    trust_remote_code=True,
                )
                print("使用 pipeline 加载成功（单GPU模式）")
            except Exception as e2:
                raise RuntimeError(f"无法加载本地模型: {e2}")

    # === OpenAI 兼容 API 相关 ===

    def _init_api_client(self):
        """初始化 OpenAI 兼容 API 客户端配置"""
        # 按后端设置默认的 base_url / model_name / api_key
        if self.backend == "qwen":
            # DashScope 兼容模式
            self.base_url = self.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            self.model_name = self.model_name or "qwen-plus"
            self.api_key = self.api_key or os.getenv("DASHSCOPE_API_KEY")
            if not self.api_key:
                raise ValueError("Qwen 模式需要环境变量 DASHSCOPE_API_KEY 或通过 --api_key 显式传入")

        elif self.backend == "deepseek":
            self.base_url = self.base_url or "https://api.deepseek.com/v1"
            self.model_name = self.model_name or "deepseek-chat"
            self.api_key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
            if not self.api_key:
                raise ValueError("DeepSeek 模式需要环境变量 DEEPSEEK_API_KEY 或通过 --api_key 显式传入")

        elif self.backend in ("internal_api", "openai_compatible"):
            if not self.base_url:
                raise ValueError("internal_api / openai_compatible 模式需要提供 --api_base_url")
            if not self.model_name:
                raise ValueError("internal_api / openai_compatible 模式需要提供 --api_model_name")
            if not self.api_key:
                # 内部接口有时不校验 key，但推荐仍然使用统一字段
                print("警告: 未提供 api_key，将以空 key 调用内部接口（如果服务端要求认证可能失败）")

        else:
            raise ValueError(f"不支持的 backend 类型: {self.backend}")

        # 统一处理 base_url 末尾的 '/'
        self.base_url = self.base_url.rstrip("/")

    # === 通用生成接口 ===

    def generate(self, prompt: str, max_length: int = 512, **kwargs) -> str:
        """
        生成文本（统一入口）

        Args:
            prompt: 输入提示
            max_length: 最大长度（仅对本地模型有效；API 模式使用 default_max_tokens）
        """
        if self.backend == "local":
            return self._generate_local(prompt, max_length=max_length, **kwargs)
        else:
            return self._generate_via_api(prompt, **kwargs)

    def _generate_local(self, prompt: str, max_length: int = 512, max_new_tokens: int = None, **kwargs) -> str:
        """使用本地模型生成"""
        # 清理无效的生成参数（pipeline 不支持某些参数）
        clean_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['temperature', 'top_p', 'top_k'] or self.pipeline is None}
        
        if self.pipeline is not None:
            # pipeline 模式：只使用 max_new_tokens 或 max_length（不能同时使用）
            pipeline_kwargs = {'do_sample': False}
            if max_new_tokens is not None:
                pipeline_kwargs['max_new_tokens'] = max_new_tokens
            else:
                pipeline_kwargs['max_length'] = max_length
            pipeline_kwargs.update(clean_kwargs)
            
            result = self.pipeline(prompt, **pipeline_kwargs)
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt) :].strip()
                return generated_text
            return ""

        # tokenizer 调用时明确设置 truncation
        # 为生成留出空间：如果使用max_length，tokenizer truncation应该小于max_length
        # 如果使用max_new_tokens，tokenizer可以truncate到max_context
        if max_new_tokens is not None:
            # 使用max_new_tokens时，tokenizer可以truncate到max_context
            tokenizer_max_length = self.max_context if hasattr(self, 'max_context') else 2048
        else:
            # 使用max_length时，tokenizer truncation应该保留至少50个token的空间给生成
            tokenizer_max_length = min(max_length - 50, self.max_context if hasattr(self, 'max_context') else max_length - 50)
            if tokenizer_max_length < 50:
                tokenizer_max_length = 50  # 至少保留50个token
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer_max_length
        ).to(self.device)

        with torch.no_grad():
            # 使用 max_new_tokens 或 max_length（不能同时使用）
            generate_kwargs = {'do_sample': False}
            # 设置pad_token_id，如果tokenizer没有pad_token，使用eos_token_id
            if self.tokenizer.pad_token_id is None:
                generate_kwargs['pad_token_id'] = self.tokenizer.eos_token_id
            else:
                generate_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
            
            if max_new_tokens is not None:
                generate_kwargs['max_new_tokens'] = max_new_tokens
            else:
                # 使用max_length时，确保它大于输入长度
                input_length = inputs.input_ids.shape[1]
                actual_max_length = max(max_length, input_length + 50)  # 至少比输入长50
                generate_kwargs['max_length'] = actual_max_length
            generate_kwargs.update(clean_kwargs)
            
            outputs = self.model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, **generate_kwargs)

        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        return generated_text.strip()

    def _generate_via_api(self, prompt: str, **kwargs) -> str:
        """通过 OpenAI 兼容 API 生成文本"""
        url = f"{self.base_url}/chat/completions"

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.default_max_tokens),
        }

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
            resp.raise_for_status()
            data = resp.json()

            # 兼容 OpenAI / 各种兼容实现
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                # chat.completions 格式
                if "message" in choice and "content" in choice["message"]:
                    return str(choice["message"]["content"]).strip()
                # completions 格式
                if "text" in choice:
                    return str(choice["text"]).strip()

            print(f"警告: API 返回格式异常: {data}")
            return ""
        except Exception as e:
            print(f"错误: 调用 OpenAI 兼容 API 失败: {e}")
            return ""

    def generate_batch(self, prompts: List[str], max_length: int = 512, **kwargs) -> List[str]:
        """
        批量生成文本

        - 本地模型：逐条或批量生成
        - API：按顺序多次请求（避免复杂的批处理逻辑）
        """
        results: List[str] = []
        for prompt in prompts:
            results.append(self.generate(prompt, max_length=max_length, **kwargs))
        return results

    def generate_first_token(self, prompt: str) -> str:
        """
        生成第一个 token（用于测量首 token 延迟）
        """
        if self.backend == "local":
            # 使用 max_new_tokens=1
            if self.pipeline is not None:
                result = self.pipeline(prompt, max_new_tokens=1, do_sample=False)
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", "")
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt) :].strip()
                    return generated_text
                return ""

            # tokenizer truncation时保留空间
            tokenizer_max_length = (self.max_context - 10) if hasattr(self, 'max_context') else 2038
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=tokenizer_max_length
            ).to(self.device)
            with torch.no_grad():
                # 设置pad_token_id
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            return generated_text.strip()

        # API 模式：请求 max_tokens=1
        return self._generate_via_api(prompt, max_tokens=1)

    def generate_with_length(self, prompt: str, max_length: int = 100) -> str:
        """
        生成指定长度的文本

        - 本地模型：使用 max_new_tokens=max_length（更准确）
        - API：使用 max_tokens=max_length
        """
        if self.backend == "local":
            # 使用 max_new_tokens 而不是 max_length，避免与输入长度冲突
            return self._generate_local(prompt, max_new_tokens=max_length)
        else:
            return self._generate_via_api(prompt, max_tokens=max_length)


