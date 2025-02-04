import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

# Load the dataset
with open('data/train_data.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open('data/test_data.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

def prepare_dataset(data):
    contexts, questions, answers = [], [], []

    # Iterate through nested structure to extract context, question, and answer
    for item in data["data"]:
        for paragraph in item["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                if len(qa["answers"]) > 0:  # Ensure the question has at least one answer
                    contexts.append(context)
                    questions.append(qa["question"])
                    answers.append({
                        "text": qa["answers"][0]["text"],
                        "answer_start": qa["answers"][0]["answer_start"]
                    })

    return Dataset.from_dict({
        "context": contexts,
        "question": questions,
        "answers": answers
    })

# Prepare the train and test datasets
train_dataset = prepare_dataset(train_data)
test_dataset = prepare_dataset(test_data)

# Split the dataset into training and validation sets
train_val_split = train_dataset.train_test_split(test_size=0.1)
train_dataset = train_val_split['train']
val_dataset = train_val_split['test']

# Load pre-trained model and tokenizer
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Preprocessing function to tokenize inputs
def preprocess_function(examples):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    offset_mapping = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        answer = examples["answers"][i]
        start_char = answer["answer_start"]
        end_char = start_char + len(answer["text"])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        context_start = 0
        while sequence_ids[context_start] != 1:
            context_start += 1
        context_end = context_start
        while context_end < len(sequence_ids) and sequence_ids[context_end] == 1:
            context_end += 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offsets[context_start][0] > end_char or offsets[context_end - 1][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Find start token position
            idx = context_start
            while idx < context_end and offsets[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            # Find end token position
            idx = context_end - 1
            while idx >= context_start and offsets[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Tokenize the train and validation datasets
tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Tokenize the test dataset
tokenized_test = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

# Evaluate the model
results = trainer.evaluate(tokenized_test)
print(results)

# Save the model and tokenizer after training
model.save_pretrained("./amharic_qa_model")
tokenizer.save_pretrained("./amharic_qa_model")