# GNET-QG: Graph Network for Multi-hop Question Generation

[![Paper](https://img.shields.io/badge/Paper-ACL%20Anthology-blue)](https://aclanthology.org/2025.genaik-1.3/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](#)

**GNET-QG** integrates a **Graph Attention Network (GAT)** with a transformer **seq2seq generator** (e.g., BART/T5) to produce coherent, **multi-hop** questions. The GAT focuses on salient **entities** and relationships; selected entity texts are concatenated to the original context and answer to form an **enriched input** that improves question generation.

> **Reference:** Samin Jamshidi & Yllias Chali (2025). *GNET-QG: Graph Network for Multi-hop Question Generation.* GenAIK Workshop. ACL Anthology ID: 2025.genaik-1.3. :contentReference[oaicite:0]{index=0}

---

## ✨ Highlights

- **Entity-graph reasoning:** Build an entity graph (NER + co-occurrence/structure edges) and run **GAT** to score/select key entities.
- **Model-agnostic enrichment:** Append selected entities to `(context; answer)` and feed any text-encoder/decoder (BART, T5).
- **End-to-end training:** Backprop through GAT + transformer jointly. (As used in the paper on HotpotQA.) :contentReference[oaicite:1]{index=1}
- **Stronger semantic quality:** Outperforms prior work on **METEOR** and competitive on BLEU/ROUGE-L. :contentReference[oaicite:2]{index=2}
- **Human eval gains:** Higher multi-hop relevance rate and improved completeness/answerability vs a strong BART baseline. :contentReference[oaicite:3]{index=3}

---

## 🧠 Method (Paper Summary)

1. **Entity Graph:** Nodes = entities (BERT-NER). Edges = same-sentence co-occurrence, paragraph title links, and cross-paragraph consistency. :contentReference[oaicite:4]{index=4}  
2. **Graph Attention:** Multi-head GAT computes attention over neighbors and updates node features; a small MLP + sigmoid selects entities. :contentReference[oaicite:5]{index=5}  
3. **Enriched Input:** Concatenate selected entity texts with the **context** and **answer** → feed into BART/T5 encoder-decoder. :contentReference[oaicite:6]{index=6}  
4. **Training:** End-to-end on **HotpotQA** (filtered, MulQG preprocessing). Pretrained BART/T5 variants are used as backbones. :contentReference[oaicite:7]{index=7}

---

## 📦 Repository Structure

