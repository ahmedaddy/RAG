import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import time

class RAGGenerator:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # This moves the model's parameters to the selected device.
        self.model.to(self.device)

    
    def generate_answer(self, question, chunks, max_tokens = 128):
        context = []
        total_len = 0
        for c in chunks:
            chunk_str = f"Source: {c.file_path}\n{c.text}\n"
            context.append(chunk_str)
            total_len += len(c.text)
            if total_len > 10000:
                break
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful coding assistant. "
                    "Answer the user's question based ONLY on the provided"
                    " context. "
                    "If the context does not contain enough information to"
                    " answer the question, respond with exactly: "
                    "'I don't have enough information in the provided context"
                    " to answer this question.' "
                    "Do NOT use your training knowledge to fill gaps. "
                    "Do NOT guess or infer beyond what is explicitly stated"
                    " in the context. "
                    "Keep your answer clear, self-contained, and faithful"
                    " to the source."
                    "Give only the answer dont use post sentence"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context information is below.\n"
                    f"---------------------\n"
                    f"{context}\n"
                    f"---------------------\n"
                    f"Given the context information, answer the"
                    f" following question: {question}"
                ),
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokenized = self.tokenizer(
            inputs,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        model_inputs = {
            name: tensor.to(self.device)
            for name, tensor in tokenized.items()
        }
        # print(model_inputs)
        start = time.perf_counter()
        generated = self.model.generate(
            input_ids=model_inputs['input_ids'],
            attention_mask=model_inputs['attention_mask'],
            max_new_tokens=42,
            do_sample=False,
            use_cache=True,
        )
        elapsed = time.perf_counter() - start

        
        # Trim the prompt tokens from the outputs
        input_len = model_inputs["input_ids"].shape[1]
        generated_ids = generated[0][input_len:].tolist()

        response: str = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        print(f"Input tokens: {model_inputs['input_ids'].shape[1]}")
        print(f"Generated tokens: {len(generated_ids)}")
        print(f"Generation time: {elapsed:.2f}s")
        # print(f"Tokens/sec: {len(generated_ids) / elapsed:.2f}")
        return response.strip()