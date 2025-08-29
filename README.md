# 🚀 Advanced RAG System with Streamlit

A comprehensive Retrieval-Augmented Generation (RAG) system built with Streamlit, featuring multiple search methods, intelligent answer generation, and interactive data visualization.

## ✨ Features

### 🔍 **Multi-Modal Search**
- **Dense Search**: Semantic embeddings using SentenceTransformers
- **Sparse Search**: SPLADE++ for keyword matching with BM25 fallback
- **Hybrid Search**: RRF (Reciprocal Rank Fusion) combining both methods
- **Alpha Mode**: Weighted combination of dense and sparse results

### 📄 **Document Processing**
- **Formats**: PDF, DOCX, TXT, MD, Excel (.xlsx, .xls), CSV
- **Smart Chunking**: 12+ chunking strategies including recursive, semantic, and hybrid
- **Metadata Extraction**: Automatic source tracking and deduplication

### 🤖 **Intelligent Answering**
- **Extractive QA**: Direct answers from document context
- **GPT Integration**: Optional OpenAI GPT enhancement
- **Policy Formatting**: Smart formatting for HR policies, dress codes, eligibility criteria
- **Structured Answers**: Bullet points, numbered lists, and clean formatting

### 📊 **Grounded Charts & Tables**
- **Data Extraction**: Extract numeric data from documents
- **Visualizations**: Bar, line, pie, donut, area, scatter charts using Vega-Lite
- **Table Generation**: Markdown tables with proper formatting
- **CSV Export**: Machine-readable data download
- **Dual Upload**: Use indexed documents or upload files specifically for charts

### ⚙️ **Advanced Configuration**
- **Reranking**: Cohere v3, Qwen3-Reranker-4B, or local cross-encoder
- **Persistence**: ChromaDB, FAISS, or in-memory storage
- **Custom Settings**: Comprehensive configuration for all RAG parameters
- **System Status**: Real-time status of all components

## 🛠️ **Installation**

### **Required Dependencies**
```bash
pip install streamlit sentence-transformers chromadb faiss-cpu rank-bm25 nltk spacy python-docx PyMuPDF PyPDF2 scikit-learn numpy pandas langchain-text-splitters
```

### **Optional Dependencies**
```bash
# For GPT integration
pip install openai

# For Cohere reranking
pip install cohere

# For Qwen reranking
pip install transformers torch

# For Excel/CSV processing
pip install pandas openpyxl

# For advanced chunking
pip install spacy
python -m spacy download en_core_web_sm
```

## 🚀 **Quick Start**

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd recursive-chunking
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Access the app**
   - Open your browser to `http://localhost:8501`

## 📚 **Usage**

### **Index Documents**
1. Go to the "📁 Index" tab
2. Upload your documents (PDF, DOCX, TXT, MD, Excel, CSV)
3. Choose chunking strategy (recommended: hybrid_chunking)
4. Configure chunk settings
5. Click "Process Documents"

### **Ask Questions**
1. Go to the "🔍 Ask" tab  
2. Enter your question
3. Choose retrieval mode (semantic, sparse, hybrid, or alpha)
4. Adjust parameters (top-k, reranking, etc.)
5. Get structured answers with references

### **Generate Charts**
1. Go to the "📊 Charts" tab
2. Enter your data request
3. Choose document source (indexed or upload new)
4. Select chart type and settings
5. View interactive visualizations

### **Configure System**
1. **Config Tab**: Set up GPT integration and API keys
2. **Settings Tab**: Advanced RAG parameters and system configuration

## 🎯 **Key Components**

### **Document Processing**
- **PyMuPDF/PyPDF2**: PDF text extraction
- **python-docx**: Word document processing
- **pandas**: Excel and CSV handling
- **Custom parsers**: Markdown and text processing

### **Text Chunking**
- **Recursive Character**: LangChain recursive splitter
- **Semantic Boundary**: Similarity-based chunking
- **Markdown Structure**: Header-aware chunking
- **Hybrid Strategy**: Combination of multiple methods

### **Search & Retrieval**
- **SentenceTransformers**: Dense embeddings (all-MiniLM-L6-v2)
- **SPLADE++**: Sparse neural retrieval
- **ChromaDB**: Vector database with persistence
- **FAISS**: High-performance similarity search

### **Answer Generation**
- **Smart Formatting**: Context-aware answer structuring
- **Policy Extraction**: Specialized handling for HR policies
- **Eligibility Criteria**: Structured bullet-point formatting
- **Reference Generation**: Automatic source citations

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Optional API keys
OPENAI_API_KEY=your_openai_key
COHERE_API_KEY=your_cohere_key
```

### **Storage Structure**
```
rag_store/
├── chroma/          # ChromaDB persistence
├── faiss/           # FAISS index files
├── bm25/            # BM25 index
├── splade/          # SPLADE++ index
└── meta.jsonl       # Document metadata
```

## 📈 **Performance Tips**

1. **For large document sets**: Use FAISS instead of ChromaDB
2. **For precise answers**: Use hybrid retrieval with reranking
3. **For speed**: Use semantic search only
4. **For comprehensive policies**: Enable GPT enhancement

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 **License**

This project is open source and available under the MIT License.

## 🆘 **Support**

For issues and questions:
1. Check the system status in the sidebar
2. Review the troubleshooting tips in each tab
3. Ensure all dependencies are installed
4. Check the console for error messages

## 🙏 **Acknowledgments**

- **Streamlit**: Amazing framework for data apps
- **SentenceTransformers**: Excellent embedding models
- **ChromaDB**: Powerful vector database
- **LangChain**: Text splitting utilities
- **Vega-Lite**: Beautiful interactive visualizations
