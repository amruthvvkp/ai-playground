from datasets import load_dataset
from langchain import Chain, Step
import mlflow
import mlflow.pytorch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Load the dataset
dataset = load_dataset("csv", data_files={"train": "path/to/your/train.csv", "test": "path/to/your/test.csv"})

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Fine-tune the model
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()


# Define a step for preprocessing the feedback
class PreprocessStep(Step):
    def run(self, input_data):
        # Preprocess the input data (e.g., cleaning, tokenization)
        return tokenizer(input_data, return_tensors="pt")


# Define a step for categorizing the feedback
class CategorizeStep(Step):
    def run(self, input_data):
        # Generate category using the fine-tuned model
        outputs = model.generate(input_data["input_ids"], max_length=50)
        category = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return category


# Create the chain
chain = Chain(steps=[PreprocessStep(), CategorizeStep()])

# Example usage
user_feedback = "The app crashes when I try to upload a photo."
category = chain.run(user_feedback)
print(f"Categorized as: {category}")

# Start an MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 2e-5)
    mlflow.log_param("batch_size", 4)
    mlflow.log_param("epochs", 3)

    # Train the model
    trainer.train()

    # Log the model
    mlflow.pytorch.log_model(model, "model")

    # Log metrics
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        mlflow.log_metric(key, value)
