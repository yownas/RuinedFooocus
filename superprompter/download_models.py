from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import os


def download_models(modelDir):
    model_name = "roborovski/superprompt-v1"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16
    )

    os.makedirs(modelDir, exist_ok=True)
    tokenizer.save_pretrained(modelDir)
    model.save_pretrained(modelDir)
    print("Downloaded SuperPrompt-v1 model files to", modelDir)
    return modelDir


if __name__ == "__main__":
    download_models()
