import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
import re

class ModelQuestionGenerator:
    def __init__(self, model_name: str = "google/gemma-3-1b-it", logger: logging=None):
        dtype = torch.bfloat16
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True, 
        #     bnb_4bit_compute_dtype=torch.float16
        # )
        # quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # quantization_config=quantization_config,
            # torch_dtype=dtype,
            device_map="auto",
        )
        self.logging=logger

    def _format_response(self, response: str) -> list:
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse model response: {response[:500]}...")
            raise ValueError(f"Failed to parse model response: {response[:500]}...")

    def generate_questions(self, prompt: str) -> list:
        """Generate questions from a fully prepared prompt"""
        chat = [{"role": "user", "content": prompt}]
        template = self.tokenizer.apply_chat_template(
            chat, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = self.tokenizer(template, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True
        )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._format_response(generated_text)