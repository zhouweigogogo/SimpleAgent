from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM


class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path
    
    def load_model(self, prompt:str, history: List[dict]) -> None:
        pass
    def chat(self) -> None:
        pass

class QwenAgent(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def load_model(self) -> None:
        print('================ Loading model ================')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.path)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.path, 
                                                          torch_dtype="auto",
                                                          device_map="auto")
        print('================ Model loaded ================')
    
    def chat(self, messages:List[dict]) -> str:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors='pt').to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
if __name__ == "__main__":
    model = QwenAgent('/home/shared/class/zhouw/models/Qwen2.5-3B-Instruct/Qwen/Qwen2___5-3B-Instruct')
    messages = [{'role':"system",'content': '你是一个聊天机器人'}]
    while True:
        query = input('user query:')
        messages.append({'role':'user','content': query})
        response = model.chat(messages=messages)
        print(response)
        messages.append({'role':'assistant', 'content':response})

