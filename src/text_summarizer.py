import os
import pandas as pd
import torch
import time
import gc
from groq import Groq
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./models/qwen-1.5b" 
DATA_DIR = "../data/"
letters_path = os.path.join(DATA_DIR, "Dataset Advice letters on objections towing of bicycles.xlsx")
OUTPUT_PATH = os.path.join(DATA_DIR, "evaluation", "Summaries_Results.csv")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "evaluation", "checkpoint_results.csv")
INPUT_COLUMN = "geanonimiseerd_doc_inhoud"

device = "mps" if torch.backends.mps.is_available() else "cpu"

class BulkInferenceEngine:
    def __init__(self):
        print(f"[*] Initializing Bulk Engine on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)

    @torch.inference_mode()
    def generate_query(self, text):
        # Simplified prompt in Dutch - less structured to avoid placeholders
        prompt = f"""Vat deze juridische brief samen met focus op deze sleuteldetails:

- Specifieke locatie, adres of gebied
- Type zaak (bijv. fiets wegslepen, parkeren, bezwaar tegen bestuur)
- Het kernarguement van de indiener
- Relevante wettelijke bepalingen of artikelen
- Korte conclusie (gegrond of ongegrond)

BELANGRIJK: Gebruik SPECIFIEKE details uit de brief, geen generieke taal. Zet eigennamen, locaties, data en technische termen op.

Brief:
{text}

Samenvatting:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[1]

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True).strip()

class GroqSummarizer:
    """Fast summarizer using Groq API (free tier). Drop-in replacement for BulkInferenceEngine."""

    def __init__(self, model="llama-3.3-70b-versatile"):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Set GROQ_API_KEY environment variable. Get a free key at https://console.groq.com")
        self.client = Groq(api_key=api_key)
        self.model = model
        print(f"[*] GroqSummarizer initialized (model: {model})")

    def generate_query(self, text):
        # Truncate very long texts to stay within context limits
        max_chars = 12000
        if len(text) > max_chars:
            text = text[:max_chars]

        prompt = """Vat deze juridische brief samen met focus op deze sleuteldetails:

- Specifieke locatie, adres of gebied
- Type zaak (bijv. fiets wegslepen, parkeren, bezwaar tegen bestuur)
- Het kernarguement van de indiener
- Relevante wettelijke bepalingen of artikelen
- Korte conclusie (gegrond of ongegrond)

BELANGRIJK: Gebruik SPECIFIEKE details uit de brief, geen generieke taal. Zet eigennamen, locaties, data en technische termen op.

Geef de samenvatting in het Nederlands, maximaal 3-4 zinnen."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Je bent een juridisch assistent die Nederlandse juridische brieven samenvat."},
                {"role": "user", "content": f"{prompt}\n\nBrief:\n{text}"}
            ],
            max_tokens=300,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    # Create evaluation directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Load data
    df = pd.read_excel(letters_path)
    engine = BulkInferenceEngine()
    results = []

    print("\n" + "🚀" * 15 + f" BULK RUN: 1 TO {len(df)} " + "🚀" * 15)

    # Process all documents
    for i, (index, row) in enumerate(df.iterrows(), 1):
        raw_text = str(row[INPUT_COLUMN])
        print(f"[{i}/{len(df)}] Case {index}: {len(raw_text)} chars...", end=" ", flush=True)
        
        start = time.time()
        summary = engine.generate_query(raw_text)
        duration = time.time() - start
        
        print(f"DONE ✅ ({duration:.1f}s)")
        
        results.append({
            "Original_Index": index,
            "Char_Count": len(raw_text),
            "Time_Taken": round(duration, 2),
            "Generated_Summary": summary
        })

        # Memory management: every 5 cases, clear cache and save checkpoint
        if i % 5 == 0:
            torch.mps.empty_cache()
            gc.collect()
            pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
            print(f"   [CHECKPOINT] Progress saved at case {i}")

    # Save final results to CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[SUCCESS] All results saved to: {OUTPUT_PATH}")