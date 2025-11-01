# 🤖 ChatGroq RAG Assistant  

### 🔗 Live Demo  
👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/your-username/chatgroq-rag-assistant)

---

### 🧠 Overview  
**ChatGroq RAG Assistant** is an AI-powered chatbot that uses **Retrieval-Augmented Generation (RAG)** to answer questions based on **Hugging Face documentation**.  
Built using **LangChain**, **Streamlit**, and **Groq API**, it performs semantic search with **FAISS** and **Hugging Face embeddings** to deliver accurate, context-aware responses.  

---

### 🚀 Features  
- 📚 Fetches and indexes Hugging Face documentation automatically  
- 🧩 Splits text into smaller chunks for efficient retrieval  
- 🧠 Uses **FAISS** for vector similarity search  
- ⚡ Integrates **Groq LLM** for fast and intelligent responses  
- 🖥️ Clean and modern **Streamlit** web interface  
- 🔍 Displays which parts of the docs were used for each answer  
- ⏱️ Shows response generation time  

---

### 🧰 Tech Stack  
| Component | Purpose |
|------------|----------|
| **LangChain** | Building the RAG pipeline |
| **Groq API** | LLM inference for response generation |
| **FAISS** | Vector similarity search |
| **Hugging Face Embeddings** | Text vectorization |
| **Streamlit** | Interactive frontend |
| **WebBaseLoader** | Loads Hugging Face documentation |

---

### ⚙️ Installation  

**1️⃣ Clone the repository:**  
```bash
git clone https://github.com/your-username/chatgroq-rag-assistant.git
cd chatgroq-rag-assistant
```

**2️⃣ Install dependencies:**
```bash
pip install -r requirements.txt
```

**3️⃣ Set up environment variables:**
Create a .env file in the project root and add:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

**▶️ Run Locally**
```bash
streamlit run app.py
```

🧑‍💻 Author
Khwaish Khandelwal
- AI/ML Enthusiast | Computer Vision | NLP | Data Science | 
[LinkedIn](https://www.linkedin.com/in/khwaish-khandelwal-543b9725a/) | [GitHub](https://github.com/KhwaishK)
