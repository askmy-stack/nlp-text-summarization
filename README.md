# nlp-text-summarization

> Extractive and abstractive text summarization using transformer models.

Compares extractive (sentence selection) and abstractive (generation) summarization using pretrained HuggingFace models, evaluated with ROUGE metrics.

## What it does

- Applies extractive summarization (sentence scoring + ranking)
- Runs pretrained abstractive models (T5, BART variants)
- Evaluates output with ROUGE-1, ROUGE-2, ROUGE-L

## Stack

- **Languages:** Python
- **ML/NLP:** HuggingFace Transformers, PyTorch
- **Tooling:** Jupyter Notebook

## Setup

```bash
git clone https://github.com/askmy-stack/nlp-text-summarization.git
cd nlp-text-summarization
pip install -r requirements.txt
jupyter notebook
```

## What I learned

Abstractive models produce more fluent summaries but hallucinate on factual content. ROUGE alone doesn't capture fluency — human eval still matters for precision-critical domains.

## License

MIT

---

Built by [Abhinaysai Kamineni](https://github.com/askmy-stack)
