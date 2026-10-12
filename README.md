# LLMs from Scratch

Two self-contained PyTorch notebooks build small language models, train them on
toy data, and generate text. Read each from top to bottom in a fresh kernel.

| Notebook | Focus |
| --- | --- |
| [00 Build an LLM End to End](notebooks/00_build_an_llm_end_to_end.ipynb) | A vanilla decoder-only Transformer, built from basic operations. |
| [01 Kimi K3 From Scratch](notebooks/01_kimi_k3_from_scratch.ipynb) | A small educational model inspired by Kimi K3, including recurrent attention, compressed global attention, attention across layers, and sparse experts. |

## Vanilla Transformer notebook

Every core building block is implemented by hand (no `nn.Linear`, no
`nn.MultiheadAttention`, no `F.scaled_dot_product_attention` in the final model):

1. **Primitives** — linear, embedding, SiLU, softmax, RMSNorm, SwiGLU, cross entropy
2. **The Transformer** — scaled dot product attention, RoPE, multi head attention, a pre norm block, the full LM
3. **Tokenization** — train a byte level BPE tokenizer and encode/decode text
4. **Training systems** — cosine LR schedule, gradient clipping, AdamW, batching, checkpoints
5. **Train a tiny model** — wire it all together and run a real training loop
6. **Inference** — KV cache, sampling (temperature / top k / top p), the generation loop, then sample from the trained model
7. **Alignment (bonus)** — SFT loss with prompt masking and the DPO preference loss

### How each section reads

For every component you get:

- **What & why** — what the block does and where it sits in the model
- **Math / algorithm** — the equations, written so you can translate them to code
- **Design choices & alternatives** — what else people use and the trade offs
- **Pitfalls** — the bugs that actually bite when you implement this
- **Interview angles** — what an interviewer tends to probe on this topic
- **Your turn** — a stub to implement
- **Solution, run to continue** — a correct reference so the pipeline keeps working

Read it top to bottom in one kernel. Each block you build stays in memory and is
reused by every block that follows, so by the end you have assembled the whole
model out of your own pieces. You can implement each "Your turn" stub yourself, or
just run the solution cells to keep the end to end pipeline flowing.

## Setup

```bash
cd llm-from-scratch-notebook
python -m venv .venv
source .venv/bin/activate
pip install -e ".[notebook]"
jupyter lab
```

Open either notebook from JupyterLab. Both training runs are intentionally small
enough for a CPU. The Kimi K3 notebook includes 31 explained code cells, uses
variable-length examples with left padding, saves executed outputs, and checks
causality, cached decoding, and padding behavior. It is a teaching
implementation, not a compatible replacement for the official Kimi K3 model or
weights. It uses standard PyTorch layers where useful to focus on K3's larger
architectural ideas.
