import logging
import traceback

from datasets import load_dataset

from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator
from sentence_transformers.cross_encoder.losses import CachedMultipleNegativesRankingLoss

# TODO: not sure if this model is inline with the rest of the trainingsetup

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model_name = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
train_batch_size = 64
num_epochs = 1
num_rand_negatives = 5  # How many random negatives should be used for each question-answer pair

# loading a cross encoder model to finetune
model = CrossEncoder(model_name)

logging.info("Read the training dataset")

full_dataset = load_dataset("csv", data_files="question_answer.csv", split="train")
dataset_dict = full_dataset.train_test_split(test_size=180, seed=12)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]
logging.info(train_dataset)
logging.info(eval_dataset)


# 3. Define our training loss.
loss = CachedMultipleNegativesRankingLoss(
    model=model,
    num_negatives=num_rand_negatives,
    mini_batch_size=32,  # Informs the memory usage
)

# 4. Use an evaluator
# Create a Dutch-specific evaluator from your eval set
eval_df = eval_dataset.to_pandas()

# Assuming your CSV has 'question' and 'answer' columns
eval_sentences1 = eval_df['question'].tolist()
eval_sentences2 = eval_df['answer'].tolist()
eval_labels = [1] * len(eval_df)  # All are positive pairs

# Add some negative examples for better evaluation
# (randomly shuffle answers to create mismatches)
shuffled_answers = eval_df['answer'].sample(frac=1, random_state=42).tolist()
eval_sentences1.extend(eval_df['question'].tolist())
eval_sentences2.extend(shuffled_answers)
eval_labels.extend([0] * len(eval_df))

evaluator = CrossEncoderClassificationEvaluator(
    sentence_pairs=list(zip(eval_sentences1, eval_sentences2)),
    labels=eval_labels,
    name="dutch_qa_eval"
)

print("STARTING TRAINING")

# 5. Define the training arguments
short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
print(short_model_name)
run_name = f"reranker-{short_model_name}-ecli-trained"
args = CrossEncoderTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=50,
    logging_first_step=True,
    seed=12,
)

# 6. Create the trainer & start training
trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=evaluator,
)
trainer.train()

# 7. Evaluate the final model, useful to include these in the model card
evaluator(model)

# 8. Save the final model
final_output_dir = f"models/{run_name}/final"
model.save_pretrained(final_output_dir)
