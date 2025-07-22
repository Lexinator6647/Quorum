
# Quorum

> _Truth by convergence, not supervision._

**Quorum** is an open-source framework for evaluating factual consistency and semantic agreement across language model outputs — without relying on human-labeled data or LLM-as-a-judge prompts.

Instead of supervised truth detection, Quorum adopts a **consensus-based approach**: it repeatedly samples responses to the same query from different models (or the same model multiple times), and computes agreement using **semantic similarity** and **Semantic Role Labeling (SRL)**.

---

## 🔍 Why Quorum?

LLMs are prone to hallucinations, but evaluating their accuracy is tricky. Human labels are costly and language models used as judges tend to agree with users (helpfulness bias).  
Quorum circumvents this by asking: _"What do other models say?"_

It assumes that truthful, consistent information will **emerge through convergence** — without any ground truth labels or explicit verification.

---

## ✨ Features

- 🤖 **Blind Sampling** – Models are queried independently without prior examples, targets, or priming.
- 🧠 **Semantic Role Labeling (SRL)** – Measures agreement beyond surface-level similarity.
- 📊 **Semantic Similarity Matrix** – Uses `sentence-transformers` to compare embedding distances.
- 🔁 **Two Modes**:
  - **Model Consistency**: Multiple samples from a single model
  - **Model Consensus**: One sample each from multiple different models
- 📦 Lightweight and easily extendable

---

## 🧪 Example

```python
# Prompt example
"When did Einstein propose the theory of relativity?"

# Sampled Responses (Consensus Mode)
1. "Albert Einstein proposed the theory of relativity in 1905."
2. "Einstein introduced his relativity theory in 1905."
3. "Einstein's theory was published in the early 20th century."

# Result: High semantic agreement and SRL alignment → High Quorum score

