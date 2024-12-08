from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers.generation.utils import GenerationConfig

class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path
    
    def load_model(self, prompt:str, history: List[dict]) -> None:
        pass
    def chat(self) -> None:
        pass

class Qwen(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def load_model(self) -> None:
        print('================ Loading model ================')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.path)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.path, 
                                                          torch_dtype="auto",
                                                          device_map="auto").eval()
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
    
class Sunsimiao(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def load_model(self) -> None:
        print('================ Loading model ================')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.path)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.path, 
                                                          torch_dtype="auto",
                                                          device_map="auto").eval()
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

# 推理需要的transformers版本太旧，暂时搁置
class BianQue2(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def load_model(self) -> None:
        print('================ Loading model ================')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=self.path, 
                                                          torch_dtype="auto",
                                                          device_map="auto",
                                                          trust_remote_code=True).half()
        print('================ Model loaded ================')
    
    def chat(self, prompt: str) -> str:
        response, history = self.model.chat(self.tokenizer, query=prompt, history=None, max_length=2048, num_beams=1, do_sample=True, top_p=0.75, temperature=0.95, logits_processor=None)
        return response, history
    
class HuatuoGPT2(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def load_model(self) -> None:
        print('================ Loading model ================')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.path, 
                                                          torch_dtype="auto",
                                                          device_map="auto",
                                                          trust_remote_code=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(pretrained_model_name=self.path)
        print('================ Model loaded ================')
    
    def chat(self, messages:List[dict]) -> str:
        response = self.model.HuatuoChat(self.tokenizer, messages)
        return response

class Yi(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def load_model(self) -> None:
        print('================ Loading model ================')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.path, 
                                                          torch_dtype="auto",
                                                          device_map="auto",
                                                          trust_remote_code=True).eval()
        print('================ Model loaded ================')

    def chat(self, messages:List[dict]) -> str:
        model_inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, return_tensors='pt')
        generated_ids = self.model.generate(model_inputs.to(self.model.device), eos_token_id=self.tokenizer.eos_token_id, max_new_tokens=512)
        response = self.tokenizer.decode(generated_ids[0][model_inputs.shape[1]:], skip_special_tokens=True)
        return response

# transformers==4.41.2  
class ChatGLM3(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def load_model(self) -> None:
        print('================ Loading model ================')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=self.path, 
                                                          torch_dtype="auto",
                                                          device_map="auto",
                                                          trust_remote_code=True).eval()
        print('================ Model loaded ================') 

    def chat(self, query:str, history:List[dict]) -> str:
        response, history = self.model.chat(self.tokenizer, query, history=history)
        return response, history

if __name__ == "__main__":
    # model = Qwen('/home/shared/class/zhouw/models/Qwen2.5-3B-Instruct/Qwen/Qwen2___5-3B-Instruct')
    # messages = [{'role':"system",'content': '你是一个聊天机器人'}]
    # while True:
    #     query = input('user query:')
    #     messages.append({'role':'user','content': query})
    #     response = model.chat(messages=messages)
    #     print(response)
    #     messages.append({'role':'assistant', 'content':response})
    
    # model = Sunsimiao('/home/shared/class/zhouw/models/Sunsimiao-Qwen2-7B/X-D-Lab/Sunsimiao-Qwen2-7B')
    # messages = [{'role':"system",'content': '你是一个聊天机器人'}]
    # while True:
    #     query = input('user query:')
    #     messages.append({'role':'user','content': query})
    #     response = model.chat(messages=messages)
    #     print(response)
    #     messages.append({'role':'assistant', 'content':response})

    # model = BianQue2('/home/shared/class/zhouw/models/BianQue-2')
    # query = input('user query:')
    # input_text = "病人：" + query + "\n医生："
    # response, history = model.chat(prompt=input_text)
    # print(response)

    model = HuatuoGPT2('/home/shared/class/zhouw/models/HuatuoGPT2-7B')
    messages = [{'role':"system",'content': '你是一个聊天机器人'}]
    while True:
        query = input('user query:')
        messages.append({'role':'user','content': query})
        response = model.chat(messages=messages)
        print(response)
        messages.append({'role':'assistant', 'content':response})

    # model = Yi('/home/shared/class/zhouw/models/Yi-1.5-6B-Chat/01ai/Yi-1___5-6B-Chat')
    # messages = [{'role':"system",'content': '你是一个聊天机器人'}]
    # while True:
    #     query = input('user query:')
    #     messages.append({'role':'user','content': query})
    #     response = model.chat(messages=messages)
    #     print(response)
    #     messages.append({'role':'assistant', 'content':response})

    # model = ChatGLM3('/home/shared/class/zhouw/models/chatglm3-6b/ZhipuAI/chatglm3-6b')
    # history = []
    # while True:
    #     query = input('user query:')
    #     response = model.chat(query=query, history=history)
    #     print(response)
    #     history.append(query)