# ProfiliTable: Profiling-Driven Tabular Data Processing via Agentic Workflows

This repository contains the implementation of **ProfiliTable**

---

## 📦 Environment Setup & Reproduction

Follow the steps below to set up the environment and run the code.

### 1. Create Conda Environment
```bash
conda create --name ProfiliTable python=3.10 -y
conda activate ProfiliTable
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e . 
```

> 💡 Ensure `requirements.txt` is in the root directory of this repository.

### 3. Prepare Data
- Unzip `data.zip` in the **current (root) directory** so that data files are placed alongside `main/`:
  ```bash
  unzip data.zip
  ```
- Navigate to `table_agent/utils/` and unzip `operators.zip` there:
  ```bash
  cd table_agent/utils/
  unzip operators.zip
  cd ../..
  ```

### 4. Set API Credentials
Export your LLM provider’s API key and endpoint URL:
```bash
export API_KEY='Your API KEY'
export API_URL='Your API URL'
```
> Replace `'Your API KEY'` and `'Your API URL'` with actual credentials from your model provider (e.g., OpenAI, Anthropic, DeepSeek, etc.).

### 5. Run the Main Script
```bash
cd main
python ProfiliTable.py [arguments]
```

---

## ⚙️ Command-Line Arguments

| Argument | Type | Default | Description |
|--------|------|--------|-------------|
| `--score_threshold` | `float` | `0.1` | Minimum similarity score threshold for terminating multi-turn feedback loops. Lower values allow more iterations; higher values exit earlier. |
| `--task` | `str` | — | Task ID to evaluate (e.g., `T0001`, `T0011`, ...). Required for specific evaluation. |
| `--use_rag` | flag | `False` | Enable Retrieval-Augmented Generation strategy for operator selection. |
| `--model` | `str` | `"gpt-4o"` | LLM backend. |
| `--input_path` | `str` | `"NL2Op"` | Input modality: `"NL2Op"` (natural language to single-step tasks) or `"NL2Dag"` (to multi-step tasks). |

### Example Usage
```bash
# Run task T0001 with RAG using GPT-4o
python ProfiliTable.py --task T0001 --use_rag --model gpt-4o

# Run NL2Dag mode without RAG
python ProfiliTable.py --task T0001 --input_path NL2Dag 
```

---

## 🔧 Additional Configuration for RAG

If you enable `--use_rag`, the system uses semantic similarity to retrieve relevant operators from a local registry.

- The embedding model path is specified in:
  ```
  table_agent/utils/utils.py → retrieve_operators()
  ```
- By default, it expects a sentence-transformers model. We recommend using:
  ```
  all-MiniLM-L12-v1  # slightly more accurate
  ```

### To use `all-MiniLM-L12-v1`:
1. Download it from Hugging Face:
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L12-v1')
   model.save('/path/to/local/all-MiniLM-L12-v1')
   ```
2. Update the model path in `retrieve_operators()`:

> 📌 Make sure the model path is accessible and matches your deployment environment.

---

## 📁 Directory Structure (After Setup)
```
ProfiliTable/
├── data/                   # Unzipped from data.zip
├── main/
│   └── ProfiliTable.py     # Main entry point
├── table_agent/
│   └── utils/
│       ├── operators/      # Unzipped from operators.zip
│       └── utils.py        # Contains retrieve_operators()
├── requirements.txt
└── README.md
```

---





