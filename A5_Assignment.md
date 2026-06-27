# Exercises

1. **Over-smoothing — how deep is too deep?**

Each GCN layer aggregates 1-hop neighbors. With `k` layers, a node sees its `k`-hop neighborhood. Too many layers → **over-smoothing** (all node representations collapse to the same vector).

a) Modify `GCN` to support variable depth. Train with `n_layers ∈ {1, 2, 3, 4, 5}` on MovieLens, then fill in:

| # Layers | Test Accuracy | Avg cosine similarity |
|---|---|---|
| 1 | ? | ? |
| 2 | ? | ? |
| 3 | ? | ? |
| 4 | ? | ? |
| 5 | ? | ? |

b) After each model, compute **pairwise cosine similarity** of test embeddings and plot it against depth. At what depth does accuracy drop noticeably? Explain mechanically why over-smoothing happens.

2. **GCN vs GAT vs GraphSAGE — when does each win?**

a) Compare the three models on the full test set:

| Model | Test Accuracy | Avg epoch time |
|---|---|---|
| GCN | ? | ? |
| GAT (8 heads) | ? | ? |
| GraphSAGE (k=10) | ? | ? |

b) Visualize attention weights for 3 sample nodes in GAT. Do the top-attended neighbors share the same genre label as the query node?

```python
attn_0 = attn[0].cpu().detach().numpy()
top_neighbors = np.argsort(attn_0)[-5:]
```

c) In what type of graph would you expect GAT to outperform GCN by the largest margin? What about GraphSAGE vs GCN?

3. **MLP baseline — does the graph actually help?**

a) Implement and train a 2-layer MLP that ignores graph structure entirely:

```python
class MLP(nn.Module):
    def __init__(self, in_features, hidden, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden, n_classes))
    def forward(self, X):
        return self.net(X)
```

b) Fill in the comparison:

| Model | Test Accuracy |
|---|---|
| MLP (no graph) | ? |
| GCN | ? |
| GAT | ? |
| GraphSAGE | ? |

c) By how much does graph structure improve accuracy? What does this tell you about the value of relational information in this dataset?

4. **LightGCN — when less is more** (challenge)

**LightGCN** (He et al., SIGIR 2020) removes the weight matrix `W` and the non-linear activation — and gets *better* recommendation results.

- GCN layer: `H' = σ( D^{-1/2} A D^{-1/2} · H · W )` — has W and σ
- LightGCN layer: `H' = D^{-1/2} A D^{-1/2} · H` — pure propagation
- Final embedding: weighted average across all `K` layers

a) Implement `LightGCN` and train on MovieLens-100k with `embed_dim=32`, `n_layers=3`, `lr=1e-2`, 30 epochs.

b) Fill in:

| Model | # Params | AUC | Recall@10 |
|---|---|---|---|
| RecGCN (with W) | ? | ? | ? |
| LightGCN (no W) | ? | ? | ? |

c) Train LightGCN with `n_layers ∈ {1, 2, 3, 4}` and plot Recall@10 vs depth. Does over-smoothing appear? Why or why not (compare to node classification)?

d) What are the *only* trainable parameters left in LightGCN? Write 2–3 sentences explaining what the model is actually learning.

---
## Submission

Push your completed notebook to GitHub.

Your repository must include:

1. The completed `A5-Graph-Neural-Networks.ipynb` with all cells executed and output visible.

2. A `README.md` with:
- Results table (all 4 exercises)
- t-SNE visualizations (GCN / GAT / GraphSAGE)
- Over-smoothing plot (Exercise 1b)
- Attention visualization (Exercise 2b)
- Discussion (2–3 sentences): when would you use a GNN instead of an MLP? Give a concrete example from biology, traffic routing, or social networks.

3. Commit message: `A5: Graph Neural Networks — [your student ID]`