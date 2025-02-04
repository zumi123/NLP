# Amharic Question-Answering System

This project implements an **Amharic Question-Answering System** using **transformers** (Hugging Face library). The system can extract answers from a given context in Amharic using a pre-trained or locally trained model.

---

## **Contents**
- `train.py`: Script to train the question-answering model.
- `app.py`: Flask/Gradio app for interacting with the trained model.
---

## **System Requirements**
Ensure you have the following installed:
- **Python** (>= 3.7)
- **Pip** (Python package manager)
- **Git** (optional for cloning the repository)

---

## **Step 1: Install Dependencies**
Create a virtual environment and install the required libraries.

1. Open a terminal and navigate to the project directory.
2. Run the following commands:

```bash
# Create virtual environment
python -m venv env

# Activate the virtual environment
# On Windows
env\Scripts\activate

# On macOS/Linux
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

If the `requirements.txt` file is not provided, manually install the following:
```bash
pip install torch transformers flask gradio
```

---

## **Step 2: Train the Model (Optional)**
If you need to train the model locally, use the provided `train.py`:

```bash
python train.py
```

Ensure you have the required dataset (Amharic question-answering dataset). The `train.py` file should be configured to:
- Load the training data
- Train the model
- Save the model to the `./amharic_qa_model` directory

> **Note:** If you already have the trained model, you can skip this step and proceed to deployment.

---

## **Step 3: Running the Application**
You can run the system either as a **Flask API** or as a **Gradio UI**. Here's how:

### **Option 1: Flask API**
The Flask app exposes a REST API endpoint at `http://localhost:5000/predict`. You can test it using any REST client like Postman or curl.

1. Run the app:
```bash
python app.py
```

2. Send a POST request to the endpoint with the context and question:

Example using `curl`:
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "context": "አማርኛ የኢትዮጵያ መንግስት ቋንቋ ናት።",
           "question": "የኢትዮጵያ መንግስት ቋንቋ ማን ናት?"
         }'
```

Expected Output:
```json
{
    "answer": "አማርኛ"
}
```

---

### **Option 2: Gradio Web Interface**
The Gradio interface provides a simple user-friendly interface for testing the system.

1. Run the app:
```bash
python app.py
```

2. A link will be displayed in the terminal, similar to:
```
Running on http://127.0.0.1:7860/
```

3. Open the link in a browser, enter the **context** and **question**, and get the answer interactively.

---


## **Possible Enhancements**
- Host the model on the **Hugging Face Hub** for easier sharing.
- Deploy the application on **Hugging Face Spaces** or **Heroku** for online access.

---
