import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

torch.random.manual_seed(0) 
model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3-mini-4k-instruct",  
    # device_map="mps",  
    torch_dtype="auto",  
    trust_remote_code=True,  
) 

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct") 

messages = [ 
            {"role": "system", "content": "From user commands extract object (noun) of the action, the name of the action (verb), and the target of the action (noun). Write it out using this template: object|action|target. If you can't find object, action or target then leave its place blank. Be as concise as possible. Example: mug|put|there"}, 
    {"role": "user", "content": "Take this red object and put it on top of the blue object"}, 
] 

pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 

generation_args = { 
    "max_new_tokens": 500, 
    "return_full_text": False, 
#    "temperature": 0.0, 
    "do_sample": False, 
} 

output = pipe(messages, **generation_args) 
print(output[0]['generated_text'])

