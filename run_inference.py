import torch
import transformers
from transformers_lora_model import  LoraLlamaForCausalLM
from alpaca_data_prep import data_prep_inference
########################################## loading configurations 
model_path =  "LORA_tinyllama_custom_k_q_"
tokenizer_path ="LORA_tinyllama_custom_k_q_"
test_data_path = "Data/test_data.json"
##########################################  STEP 1 : loading models and tokenizer
model = LoraLlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        loading_adapter = True,
    )
tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
#########################################  STEP 2 : Data Pre-Processing 
question = "hello how are you."
tokenizer.chat_template="{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
messages=[
    {"role": "user", "content": question}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")
########################################## STEP 3 : Model output generation
output_tokens = model.generate(input_ids,max_new_tokens=256,do_sample=True,temperature=0.7, top_k=50, top_p=0.95)
print (output_tokens)
decoded_output = tokenizer.decode(output_tokens[0])
print("Decoded output >",decoded_output)



# #########################################  STEP 2 : Data Pre-Processing for alpaca data format
# input_ids = data_prep_inference(test_data_path,tokenizer).to("cuda")