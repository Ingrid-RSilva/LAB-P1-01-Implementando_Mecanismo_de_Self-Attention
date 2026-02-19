"""
heatmap.py - Visualização dos pesos de atenção via heatmap.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from attention import ScaledDotProductAttention


def plot_attention_heatmap(attention_weights: np.ndarray, tokens: list, title: str, ax):
    """Plota um heatmap dos pesos de atenção."""
    im = ax.imshow(attention_weights, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, fontsize=11)
    ax.set_yticklabels(tokens, fontsize=11)
    ax.set_xlabel("Key (Origem)", fontsize=12)
    ax.set_ylabel("Query (Destino)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    # Anotar valores nas células
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            val = attention_weights[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')
    return im


def main():
    # Tokens de exemplo (frase simples)
    tokens = ["O", "gato", "sentou", "no", "tapete"]
    seq_len = len(tokens)
    d_model = 16
    d_k = 8
    d_v = 8

    # Embeddings aleatórios mas reproduzíveis
    rng = np.random.default_rng(seed=7)
    X = rng.standard_normal((seq_len, d_model))

    model = ScaledDotProductAttention(d_model, d_k, d_v)
    output, attention_weights = model.forward(X)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0f0f1a')
    fig.suptitle(
        "LAB P1-01 · Scaled Dot-Product Attention — Heatmap de Pesos",
        fontsize=15, fontweight='bold', color='white', y=1.02
    )

    for ax in axes:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')

    # Heatmap principal
    im1 = plot_attention_heatmap(attention_weights, tokens,
                                  "Attention Weights\nsoftmax(QKᵀ / √d_k)", axes[0])
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Heatmap de scores brutos (antes do softmax) para comparação
    Q = X @ model.W_q
    K = X @ model.W_k
    raw_scores = (Q @ K.T) / np.sqrt(d_k)

    im2 = axes[1].imshow(raw_scores, cmap='coolwarm', aspect='auto')
    axes[1].set_xticks(range(seq_len))
    axes[1].set_yticks(range(seq_len))
    axes[1].set_xticklabels(tokens, fontsize=11, color='white')
    axes[1].set_yticklabels(tokens, fontsize=11, color='white')
    axes[1].set_xlabel("Key (Origem)", fontsize=12, color='white')
    axes[1].set_ylabel("Query (Destino)", fontsize=12, color='white')
    axes[1].set_title("Scaled Scores\n(QKᵀ / √d_k) antes do Softmax",
                       fontsize=13, fontweight='bold', color='white', pad=12)
    for i in range(seq_len):
        for j in range(seq_len):
            val = raw_scores[i, j]
            axes[1].text(j, i, f"{val:.2f}", ha='center', va='center',
                         fontsize=10, color='white', fontweight='bold')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("attention_heatmap.png", dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("✅ Heatmap salvo em: attention_heatmap.png")
    plt.show()


if __name__ == "__main__":
    main()
