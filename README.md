# LAB P1-01 — Scaled Dot-Product Attention (From Scratch)

Implementação manual do mecanismo de **Self-Attention** conforme o paper  
_"Attention Is All You Need"_ (Vaswani et al., 2017).

## Equação Implementada

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```

---

## Como Rodar

```bash
# 1. Instalar dependências
pip install numpy matplotlib

# 2. Rodar os testes
python test_attention.py

# 3. Gerar o heatmap de pesos de atenção
python heatmap.py
```

---

## Como a Normalização √d_k foi Aplicada

O produto escalar `QKᵀ` cresce em magnitude à medida que `d_k` aumenta,  
empurrando o softmax para regiões de gradiente muito pequeno (vanishing gradients).

Para estabilizar o treinamento, **dividimos os scores por `√d_k`** antes do softmax:

```python
scaled_scores = (Q @ K.T) / np.sqrt(self.d_k)
attention_weights = softmax(scaled_scores)
```

Isso mantém a variância dos scores próxima de 1, independente da dimensão escolhida.

---

## Exemplo de Input e Output Esperado

```python
# Input: 3 tokens, d_model=3 (matriz identidade)
X = np.eye(3)

# Output esperado (aproximado, depende dos pesos W_q, W_k, W_v)
# Attention Weights ≈ distribuição uniforme (~0.333 por célula)
# porque tokens da identidade são ortogonais entre si
```

| Token   | Peso p/ Token 1 | Peso p/ Token 2 | Peso p/ Token 3 |
| ------- | --------------- | --------------- | --------------- |
| Token 1 | 0.334           | 0.334           | 0.333           |
| Token 2 | 0.331           | 0.332           | 0.337           |
| Token 3 | 0.333           | 0.333           | 0.334           |

_Soma de cada linha = 1.0 (propriedade do softmax)_

---

## Testes de Validação

O arquivo `test_attention.py` contém **6 casos de teste**:

1. **Shapes de saída** — verifica se `output.shape == (seq_len, d_v)` e `attention_weights.shape == (seq_len, seq_len)`
2. **Pesos somam 1 por linha** — valida a propriedade fundamental do softmax
3. **Pesos em [0, 1]** — garante que nenhum peso está fora do intervalo válido
4. **Exemplo numérico determinístico** — força `W_q = W_k = W_v = I` e verifica o cálculo passo a passo contra resultado manual
5. **Scaling evita colapso do softmax** — compara a entropia das distribuições com e sem `√d_k`, confirmando que o scaling suaviza os pesos
6. **Softmax numericamente estável** — testa com scores da ordem de 1000 para garantir ausência de `NaN` ou `Inf`

---

## Visualização (heatmap.py)

O script `heatmap.py` gera dois painéis lado a lado para uma sequência de 5 tokens (`"O", "gato", "sentou", "no", "tapete"`) com `d_model=16`, `d_k=8`, `d_v=8`:

- **Painel esquerdo** — Attention Weights após o softmax (colormap `viridis`, escala [0, 1])
- **Painel direito** — Scaled Scores brutos `QKᵀ / √d_k` antes do softmax (colormap `coolwarm`)

O heatmap é salvo automaticamente como `attention_heatmap.png`.

---

## Estrutura do Repositório

```
.
├── attention.py          # Implementação da classe ScaledDotProductAttention
├── test_attention.py     # Scripts de teste com 6 casos de validação
├── heatmap.py            # Visualização dos pesos via heatmap (matplotlib)
├── attention_heatmap.png # Heatmap gerado
├── requirements.txt      # Dependências: numpy>=1.24, matplotlib>=3.7
└── README.md             # Esta documentação

```
---

## Dependências

```
numpy>=1.24
matplotlib>=3.7

```
---


> Este README foi gerado com auxílio de Inteligência Artificial (Claude, Anthropic) e revisado por Ingrid*
```
