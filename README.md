# 📄 **ChatGroq Document Q&A Chatbot**  

An **AI-powered chatbot** that allows users to upload **PDF** documents and ask questions based on their content. The chatbot uses **Retrieval-Augmented Generation (RAG)** powered by **ChatGroq’s Deepseek model**, **ChromaDB** for vector search, and **Streamlit** for an interactive UI.

---

## 🚀 **Features**  
- 📤 **Upload Documents:** Supports PDF files.  
- 🔍 **Context-Aware Q&A:** Retrieves the most relevant document chunks for accurate answers.  
- 🤖 **Powered by ChatGroq:** Generates human-like, context-driven responses using advanced LLMs.  
- 📝 **Chat History:** Maintains an ongoing conversation with users.  
- 📚 **Source Transparency:** Displays the document sources for each answer.  

---

## 🧩 **Tech Stack**  
- **Frontend:** [Streamlit](https://streamlit.io/) for building the interactive chatbot UI  
- **LLM:** [ChatGroq (Deepseek)](https://groq.com/) for generating responses  
- **Vector Database:** [ChromaDB](https://www.trychroma.com/) for fast similarity searches  
- **Embeddings:** [Hugging Face Sentence Transformers](https://www.sbert.net/) for document chunk vectorization  
- **Orchestration:** [LangChain](https://www.langchain.com/) for managing document processing, retrieval, and LLM interactions  

---

## ⚙️ **Installation**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/mohamedbilal1800/Doc_QA_Chatbot.git
cd Doc_QA_Chatbot
```

### **2️⃣ Create a Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **4️⃣ Set Up Environment Variables**  
Create a `.env` file in the project root:  
```bash
touch .env
```
Add your **Groq API key**:  
```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 💡 **Usage**  

### **1️⃣ Run the Chatbot**  
```bash
streamlit run main.py
```

### **2️⃣ Interact with the Chatbot**  
1. 📄 Upload a PDF or TXT document.  
2. 💬 Ask questions about the document’s content.  
3. 🤖 Get accurate, AI-generated answers.  
4. 📚 View document sources for transparency.

---

## 📦 **Project Structure**  

```
Doc_QA_Chatbot/
├── main.py               # Main Streamlit app
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (API keys)
└── chroma_db/            # ChromaDB vector database (auto-created)
```

---

## ⚡ **Example Workflow**  
1. **Upload:** A PDF containing a research paper.  
2. **Question:** “What is the main conclusion of this paper?”  
3. **Processing:**  
   - The document is split into chunks.  
   - Chunks are embedded and stored in ChromaDB.  
   - The most relevant chunks are retrieved based on the question.  
4. **Response:** ChatGroq generates an accurate, context-aware answer.  
5. **Sources:** You can see which parts of the document were used to generate the answer.  

---

## 🛠️ **Key Functionalities Explained**  
- **RAG (Retrieval-Augmented Generation):** Enhances LLM responses by providing real document context.  
- **Caching:** Optimizes performance by caching models, retrieved data, and chat history.  
- **Session State:** Maintains conversation continuity across multiple user inputs.  

---

## 🚧 **Future Improvements**  
- 🌍 Multi-document support  
- 🔑 User authentication and session management  
- 🎯 Advanced search filters for better retrieval  
- 📊 Analytics dashboard for document insights  

---

## 🙌 **Acknowledgments**  
- [Streamlit](https://streamlit.io/)  
- [LangChain](https://www.langchain.com/)  
- [ChatGroq](https://groq.com/)  
- [ChromaDB](https://www.trychroma.com/)  
- [Hugging Face](https://huggingface.co/)  

---
