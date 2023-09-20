import torch

from transformers import BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, GenerationConfig
from accelerate import infer_auto_device_map, init_empty_weights
import transformers

# 语言模型文件夹，以及下面的几个模型名称
modelsPath = "e:/oobabooga_windows/text-generation-webui/models/"
ChineseAlpaca2_7b_hf = modelsPath + "chinese-alpaca-2-7b-16k-hf"
Llama2_chat_7b = modelsPath + "llama-7b-chat-hf"
vicuna_13b_GPTQ_bit = modelsPath + "vicuna-13b-GPTQ-4bit-128g"
baichuan_7B_Chat_4bits = modelsPath + "baichuan-inc_Baichuan2-7B-Chat-4bits"
baichuan_13B_Chat_4bits = modelsPath + "baichuan-inc_Baichuan2-13B-Chat-4bits"

models = {
    "chinese-alpaca-2-7b-16k-hf": ChineseAlpaca2_7b_hf,
    "llama-7b-chat-hf": Llama2_chat_7b,
    "vicuna-13b-GPTQ-4bit-128g": vicuna_13b_GPTQ_bit,
    "baichuan-inc_Baichuan2-7B-Chat-4bits": baichuan_7B_Chat_4bits,
    "baichuan-inc_Baichuan2-13B-Chat-4bits": baichuan_13B_Chat_4bits
}

# 这里是选择模型
defaultCheckpoint = ChineseAlpaca2_7b_hf

stop_everything = False


class _StopEverythingStoppingCriteria(transformers.StoppingCriteria):
    def __init__(self):
        transformers.StoppingCriteria.__init__(self)

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor) -> bool:
        return stop_everything


class HuggingFaceLaoder():
    checkpoint = None
    CPU_MEMORY = 16
    model = None
    tokenizer = None

    def __init__(self, checkpoint="chinese-alpaca-2-7b-16k-hf", **params):
        modelList = models.keys()
        self.checkpoint = models[checkpoint] if checkpoint in modelList else defaultCheckpoint
        self._loadModel(**params)

    def get_max_memory_dict(self):
        max_memory = {}
        # 读取第一个GPU的显存
        total_mem = (torch.cuda.get_device_properties(
            0).total_memory / (1024 * 1024))
        # 给显存只留下1000M，其它全部占用
        suggestion = round((total_mem - 1000) / 1000) * 1000
        if total_mem - suggestion < 800:
            suggestion -= 1000

        suggestion = int(round(suggestion / 1000))
        print(
            f"Auto-assiging --gpu-memory {suggestion} for your GPU to try to prevent out-of-memory errors. You can manually set other values.")
        max_memory = {0: f'{suggestion}GiB', 'cpu': f'{self.CPU_MEMORY}GiB'}

        return max_memory if len(max_memory) > 0 else None

    def generateLoaderParams(
        self,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        load_in_4bit=False,
        load_in_8bit=True,
        compute_dtype="float16",
        quant_type="fp4",
        use_double_quant=False,
        auto_devices=True,
        use_bf16=False,
        use_disk=False,
    ) -> dict:
        """
        生成加载模型的参数:
        以下两个参数不能同时为True
        load_in_4bit: 是否使用4bit加载
        load_in_8bit: 是否使用8bit加载

        以下参数只有在load_in_4bit为True时才会生效
        compute_dtype: 4bit的计算精度，可以是"bfloat16", "float16", "float32"
        quant_type: 4bit的量化类型，可以是"fp4", "nf4"
        use_double_quant: 是否使用双量化

        以下参数只有在load_in_8bit为True时才会生效
        auto_devices: 是否自动分配设备
        use_bf16: 是否使用bf16格式的权重
        use_disk: 是否使用磁盘缓存
        """
        # 下面是自动检测并生成配置文件
        params = {
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "trust_remote_code": trust_remote_code
        }

        use_cpu = False
        # 下面是自动检测是否使用cpu
        if not any((torch.cuda.is_available(), torch.backends.mps.is_available())):
            use_cpu = True

        if use_cpu:
            params["torch_dtype"] = torch.float32
        else:
            params["device_map"] = 'auto'
            if load_in_4bit:
                quantization_config_params = {
                    'load_in_4bit': True,
                    'bnb_4bit_compute_dtype': eval("torch.{}".format(compute_dtype)) if compute_dtype in ["bfloat16", "float16", "float32"] else None,
                    'bnb_4bit_quant_type': quant_type,
                    'bnb_4bit_use_double_quant': use_double_quant,
                }
                params['quantization_config'] = BitsAndBytesConfig(
                    **quantization_config_params)
            elif load_in_8bit:
                # 这里是使用8bit的配置
                if auto_devices:
                    params['quantization_config'] = BitsAndBytesConfig(
                        load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
                else:
                    params['quantization_config'] = BitsAndBytesConfig(
                        load_in_8bit=True)
            elif use_bf16:
                params["torch_dtype"] = torch.bfloat16
            else:
                params["torch_dtype"] = torch.float16

            params['max_memory'] = self.get_max_memory_dict()
            if use_disk:
                params["offload_folder"] = "cache"

        if load_in_8bit and params.get('max_memory', None) is not None and params['device_map'] == 'auto':
            config = AutoConfig.from_pretrained(
                self.checkpoint, trust_remote_code=False)
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(
                    config, trust_remote_code=False)

            model.tie_weights()
            params['device_map'] = infer_auto_device_map(
                model,
                dtype=torch.int8,
                max_memory=params['max_memory'],
                no_split_module_classes=model._no_split_modules
            )

        return params

    def _loadModel(self, **params):
        if self.checkpoint == defaultCheckpoint:
            config = AutoConfig.from_pretrained(self.checkpoint)
            loadParams = self.generateLoaderParams(**params)

            model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint, config=config, **loadParams)
            tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint, use_fast=True)
        elif self.checkpoint == baichuan_7B_Chat_4bits or self.checkpoint == baichuan_13B_Chat_4bits:
            tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint, use_fast=False, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
            model.generation_config = GenerationConfig.from_pretrained(
                self.checkpoint)
        else:
            config = AutoConfig.from_pretrained(self.checkpoint)
            model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint, config=config)
            tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint, use_fast=True)

        self.model = model
        self.tokenizer = tokenizer

    def getGenerateParams(self) -> dict:
        max_new_tokens = 16384 if self.checkpoint == defaultCheckpoint else 4096
        generate_params = {
            'max_new_tokens': max_new_tokens,
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.9,
            'typical_p': 1,
            'repetition_penalty': 1.15,
            'guidance_scale': 1,
            'encoder_repetition_penalty': 1,
            'top_k': 20, 'min_length': 0,
            'no_repeat_ngram_size': 0,
            'num_beams': 1,
            'penalty_alpha': 0,
            'length_penalty': 1,
            'early_stopping': False,
            'use_cache': True,
            'pad_token_id': 2,
            'stopping_criteria': [StoppingCriteriaList(), _StopEverythingStoppingCriteria()],
            'logits_processor': []
            # 'tfs': 1,
            # 'top_a': 0,
            # 'mirostat_mode': 0,
            # 'mirostat_tau': 5,
            # 'mirostat_eta': 0.1,
            # 'repetition_penalty_range': 0,
        }
        return generate_params
