# Simple-Lora

Simple-Lora is a fine-tuning code developed from scratch for implementing LORA (Low Rank Adapters) on small language models, typically those with less than 4 billion parameters. It addresses the challenge of poor performance encountered when applying LORA due to quantization issues (such as fp16, fp8). This implementation focuses on applying LORA using fp32 training, aiming to improve the efficiency and effectiveness of small language models.

## Prerequisites

Python 3.10.12 is required to run Simple-Lora. Install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
```

Additionally, download the TinyLlama-1.1B-Chat-v1.0 model from the Hugging Face model repository.

### Setting up the Training Configuration

Before training, configure the training settings:

- **Model and Tokenizer and  Data Path**: Set the path for the model,tokenizer,data. Simple-Lora supports two formats:
  1. AlphaCA Format: `"instruction: , input: , output: "`
  2. Chat ML Format:
     ```
     "assistant\n{msg}",
     "user\n{msg}"
     ```
- **Training Parameters**: Set the training parameters such as optimizer, number of training epochs, learning rate, and learning rate scheduler type. We have found the following parameters to perform well across different models like Phi-2, TinyLlama, and Gemma:
  - Optimizer: `adamw_torch`
  - Number of Train Epochs: `5`
  - Learning Rate: `2e-5`
  - Learning Rate Scheduler Type: `linear`

### Training Command

Execute the following command to start training:

```bash
torchrun --nproc_per_node=1 --master_port=8000 run_trainer.py
```

## Inference Command

Once trained, Simple-Lora can be used for inference. Run the following command:

```bash
python run_inference.py
```

Feel free to modify the configurations and parameters according to your specific requirements and datasets.
