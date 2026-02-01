import os
import pandas as pd
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./models/fietje" 
DATA_DIR = "../data/"
letters_path = os.path.join(DATA_DIR, "Dataset Advice letters on objections towing of bicycles.xlsx")
INPUT_COLUMN = "geanonimiseerd_doc_inhoud"

device = "mps" if torch.backends.mps.is_available() else "cpu"

class FietjeStableEngine:
    def __init__(self):
        print(f"[*] Loading Fietje-2 (2.7B) on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        # We MUST ensure the tokenizer knows the model limit
        self.tokenizer.model_max_length = 2048
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(device)

    @torch.inference_mode()
    def generate_query(self, text):
        # We use a direct prompt. 
        # Tokenizer will handle the 2048 limit AUTOMATICALLY now to prevent crashes.
        prompt = f"Gebruiker: Vat de volgende brief kort samen.\n\nBrief: {text}\n\nAssistent: De kern van dit bezwaar is:"

        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, # This is the safety switch
            max_length=2048  # This prevents the 4.9GB crash
        ).to(device)
        
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=100, 
            do_sample=False, # Faster and more stable
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

if __name__ == "__main__":
    # Prevent the 'Leaked Semaphore' warning by cleaning environment
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    df = pd.read_excel(letters_path).head(10)
    engine = FietjeStableEngine()

    print("\n" + "🎓" * 15 + " STABLE LEGAL INFERENCE " + "🎓" * 15)

    for i, (index, row) in enumerate(df.iterrows(), 1):
        raw_text = str(row[INPUT_COLUMN])
        print(f"[{i}/10] Case {index}: ({len(raw_text)} chars)...", end=" ", flush=True)
        
        start = time.time()
        try:
            result = engine.generate_query(raw_text)
            print(f"DONE ✅ ({time.time() - start:.2f}s)")
            print(f"   RESULT: {result}\n" + "-"*50)
        except Exception as e:
            print(f"FAILED ❌ - {e}")