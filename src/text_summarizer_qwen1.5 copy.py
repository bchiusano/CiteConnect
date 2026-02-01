import os
import pandas as pd
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./models/qwen-1.5b" 
DATA_DIR = "../data/"
letters_path = os.path.join(DATA_DIR, "Dataset Advice letters on objections towing of bicycles.xlsx")
INPUT_COLUMN = "geanonimiseerd_doc_inhoud"

# Detect Mac GPU
device = "mps" if torch.backends.mps.is_available() else "cpu"

class FullContextEngine:
    def __init__(self):
        print(f"[*] Loading Qwen-1.5B for FULL DATA inference on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            # This is the key: we tell the model it might see very long text
            trust_remote_code=True 
        ).to(device)

    @torch.inference_mode()
    def generate_query(self, text):
        # --- ZERO REDUCTION ---
        # We take the raw input from your Excel exactly as it is.
        full_text = str(text)
        
        prompt = f"Vat deze juridische brief samen in 2-3 zinnen.\n\nBrief: {full_text}\n\nSamenvatting:"

        # We tokenize without a max_length cap. 
        # This will send all 15,000+ characters to the model.
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        
        input_length = inputs["input_ids"].shape[1]

        # Generate using the full context window (Qwen 2.5 supports up to 128k tokens)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Only decode the newly generated summary
        return self.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True).strip()

if __name__ == "__main__":
    # Crucial for 16GB Mac: don't let PyTorch "reserve" memory it doesn't need
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    df = pd.read_excel(letters_path).head(10)
    engine = FullContextEngine()

    print("\n" + "📂" * 15 + " FULL CONTEXT PROCESSING " + "📂" * 15)

    for i, (index, row) in enumerate(df.iterrows(), 1):
        raw_text = str(row[INPUT_COLUMN])
        char_count = len(raw_text)
        
        print(f"[{i}/10] Case {index}: Processing {char_count} characters...", end=" ", flush=True)
        
        start = time.time()
        try:
            result = engine.generate_query(raw_text)
            duration = time.time() - start
            print(f"DONE ✅ ({duration:.2f}s)")
            print(f"   SUMMARY: {result}\n" + "-"*60)
        except Exception as e:
            print(f"FAILED ❌ - Your Mac likely ran out of RAM for this length. Error: {e}")