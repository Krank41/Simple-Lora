import torch
import torch.nn as nn
import os
from typing import Optional
from transformers import Trainer
from transformers import LlamaModel,LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaSdpaAttention
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.llama.configuration_llama import LlamaConfig

def unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model

class LoraLlamaConfig(LlamaConfig):
    def __init__(self, num_of_components=320, **kwargs):
        super().__init__(**kwargs)
        self.num_of_components = num_of_components
        self.loading_adapter = False

    def to_dict(self):
        base_dict = super().to_dict()
        config_dict = {
            "num_of_components": self.num_of_components,
        }
        base_dict.update(config_dict)
        return base_dict

class LoraLinear(nn.Module):
    def __init__(self,config,linear, in_features: int, out_features: int, n_component: int,layer_idx:int,neme:str,bias: bool = True):
        super(LoraLinear, self).__init__()
        self.lora_A = nn.Linear(in_features, n_component, bias=False,dtype=torch.bfloat16)
        self.lora_B = nn.Linear(n_component, out_features, bias=bias,dtype=torch.bfloat16)
        self.linear = linear
        #variable passing --> self, linear, rank, alpha
        if config.loading_adapter == True:
            print("loading adapater weights ...................")
            new_weights_path = "LORA_tinyllama_custom_k_q_/q_k_v_o_projector.bin"
            new_weights = torch.load(new_weights_path, map_location='cpu')
            for k,v in new_weights.items():
                if layer_idx == int(k.split(".")[2]):
                    if k.split(".")[4] == neme+"_proj":
                        if k.split(".")[-2] == "lora_A":        
                            self.lora_A.load_state_dict({k.split('.')[-1]: v })
                        if k.split(".")[-2] == "lora_B":
                            self.lora_B.load_state_dict({k.split('.')[-1]: v })

    def forward(self, input):
        prev_input = self.linear(input)
        x = self.lora_A(input)
        return prev_input+self.lora_B(x)


class LoraLlamaAttention(LlamaSdpaAttention):
    def __init__(self, config: LoraLlamaConfig, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        config.num_of_components = 320
        self.q_proj = LoraLinear(config,self.q_proj,self.hidden_size, self.num_heads * self.head_dim, config.num_of_components,layer_idx,"q",bias=False)
        self.k_proj = LoraLinear(config,self.k_proj,self.hidden_size, self.num_key_value_heads * self.head_dim, config.num_of_components,layer_idx,"k" ,bias=False)
        self.v_proj = LoraLinear(config,self.v_proj,self.hidden_size, self.num_key_value_heads * self.head_dim, config.num_of_components, layer_idx,"v",bias=False)
        self.o_proj = LoraLinear(config,self.o_proj,self.hidden_size, self.hidden_size, config.num_of_components,layer_idx,"o" ,bias=False)
       
class LoraLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LoraLlamaConfig, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.self_attn = LoraLlamaAttention(config=config, layer_idx=layer_idx)
        
class LoraLlamaModel(LlamaModel):
    def __init__(self, config: LoraLlamaConfig,loading_adapter):
        super().__init__(config)
        config.loading_adapter = loading_adapter
        self.layers = nn.ModuleList([LoraLlamaDecoderLayer(config,layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.post_init()

class LoraLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LoraLlamaConfig,loading_adapter=False):
        super().__init__(config)
        print("loading adapater >>",loading_adapter)
        self.model = LoraLlamaModel(config,loading_adapter)
        self.post_init()
    def save_checkpoint(self, dir):
        # to bypass the code line 2291 in transformers.trainer
        pass

class LoraSFTTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        _state_dict = state_dict
        if _state_dict is None:
            model_to_save = unwrap_model(self.model)
            _state_dict = model_to_save.state_dict()
        weight_to_save = {}
        keys_to_match = ['lora']
        for k, v in _state_dict.items():
            if any(key_match in k for key_match in keys_to_match):
                weight_to_save[k] = v
        torch.save(weight_to_save, os.path.join(output_dir, f'q_k_v_o_projector.bin'))
        super(LoraSFTTrainer, self)._save(output_dir, state_dict)