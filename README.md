# SDP-LLM: Acousto-Electric Effect RAG Chatbot
A specialized Retrieval-Augmented Generation (RAG) chatbot for answering questions about the acousto-electric effect** (AE effect) using your provided research papers.

### Features
- **Automatic Hybrid Mode**: Fast-path first → falls back to full RAG only when no match is found.
- **Safe Fallback**: Replies *"Sorry, I do not have the answer at the moment"* when no relevant information is available.
- **Sentence-aware chunking**: PDFs are split by full stops so no sentence is ever broken.
- **Two sets of sample questions** with **🔀 Shuffle** buttons for easy exploration.
- **Real-time terminal logging** showing whether Fast-path or RAG was used.
- **Optimized for MacBook M2** (MPS + hybrid CPU/GPU).

File Structure
textRAG/
├── pdfs/
│   ├── 1999_jossinet_Impedance Modulation by Pulsed Ultrasound.pdf
│   ├── 2000_lavandier_Experimental measurement of the acousto-electric interaction signal in saline solution.pdf
│   ├── 2007_Russel_witte_Imaging current flow in lobster nerve cord using the acoustoelectric effect.pdf
│   ├── 2008_Russel_Olafsson- Ultrasound Current Source Density Imaging.pdf
│   ├── 2017_Jean_An integrated and highly sensitive ultrafast Phys._Med._Biol._62_5808.pdf
│   ├── 2019_Jean_berthonMapping biological current densities with…coustoelectric Imaging_application...rat heart.pdf
│   ├── 2020_Russel_In vivo acoustoelectric imaging for high-resolution.... dynamics.pdf
│   ├── Frontiers in Neuroscience_AE review paper.pdf
│   └── VantageSequenceProgrammingManual.pdf
├── pdf_ingest.py
├── build_index.py
├── rag_chat.py
├── chunks.txt
├── faiss.index
├── chunks.pkl
├── dictionary.json
├── index_metadata.json
├── requirement.txt
└── README.md

### Architecture
PDFs
├── Text extraction (PyMuPDF)
├── Sentence-aware chunking (full-stop boundary)
├── Embedding (sentence-transformers/all-MiniLM-L6-v2)
├── FAISS vector index
└── Retrieval
↓
LLaMA-7B (hybrid CPU + MPS)
↓
Gradio Web UI Chatbot



## Prerequisites (MacBook M2 / Apple Silicon)
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
# 2. Upgrade pip
python -m pip install --upgrade pip
# 3. Install PyTorch with MPS support (critical for M2)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# 4. Install project dependencies
pip install -r requirement.txt
# setup and run:
mkdir -p pdfs
python pdf_ingest.py          
python build_index.py
python rag_chat.py