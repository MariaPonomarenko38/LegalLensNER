from peft import PeftConfig, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Predictor:
    def __init__(self, model_load_path: str):
        self.model = AutoPeftModelForCausalLM.from_pretrained(
                            model_load_path,
                            low_cpu_mem_usage=True,
                            torch_dtype=torch.float16,
                            device_map='auto',
                        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_load_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def get_input_ids(self, prompt: str):
        
        input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            ).input_ids.cuda()
        return input_ids

    @torch.inference_mode()
    def predict(self, prompt: str, max_target_length: int = 150, temperature: float = 0.01) -> str:
        input_ids = self.get_input_ids(prompt)
        outputs = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_p=0.9,
            max_new_tokens=max_target_length,
            temperature=temperature,
        )
        prediction = self.tokenizer.batch_decode(outputs.cpu().numpy(), skip_special_tokens=True)[0]

        return prediction
    
if __name__ == '__main__':
    
    path = '/root/LegalLensNER/models/v1/assets'
    predictor = Predictor(model_load_path=path)
    prediction = predictor.predict(prompt="USER: warning to all sellers out there ! this e-commerce giant has been pulling the wool over our eyes. they launched a new business initiative, promising us growth and success. but all theyve done is inflate their earnings guidance and mislead us about the initiatives success. theyve been reducing their financial guidance, blaming it on softer sales and unplanned expenses. but the truth is, they cant handle the increased inventory levels and are facing margin pressures. its all a scheme to inflate their stock price. were the ones paying the price for their lies ! ASSISTANT: ")
    print(prediction)