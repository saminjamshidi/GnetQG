# GnetQG — Graph-Enhanced Question Generation (GAT + BART)

This repo implements a hybrid **Graph Attention Network (GAT)** over entity nodes combined with a **Seq2Seq BART** generator for question generation. The GAT selects salient entities; selected entities are appended to the context fed to BART, improving QG quality.

> Paper: If you cite or describe this work, please reference your paper and/or include the ACL Anthology entry.  
> (Example) Jamshidi, S., et al. (2025). *[Your paper title]*. ACL GenAI Workshop.  

---

## Features
- **Entity-graph reasoning** via a custom `AttentionLayer` (see `GAT.py`)
- **BART-based decoding** with encoder freezing option for faster training
- **End-to-end pipeline:** dataset → training → validation → metrics (BLEU, METEOR, ROUGE-L, BERTScore) → predictions dump
- Clean, modular codebase:
