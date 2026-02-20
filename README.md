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
## Normalização por √d_k — Explicação Detalhada

---

### O Problema: Por que os scores explodem?

No mecanismo de atenção, calculamos o produto escalar entre queries e keys:

```
scores = Q @ Kᵀ
```

Cada elemento desse resultado é um produto escalar da forma:

```
score(q, k) = q₁k₁ + q₂k₂ + ... + q_{d_k} · k_{d_k}
```

Ou seja, é uma **soma de `d_k` termos**. Agora a parte estatística crucial.

---

### A Matemática por Trás

Assumindo que os elementos de Q e K são variáveis aleatórias independentes com:
- **Média = 0**
- **Variância = 1**

A variância de cada produto `qᵢ · kᵢ` é:

```
Var(qᵢ · kᵢ) = Var(qᵢ) · Var(kᵢ) = 1 · 1 = 1
```

Como o score é a **soma de `d_k` termos independentes**, pela linearidade da variância:

```
Var(score) = Var(q₁k₁) + Var(q₂k₂) + ... + Var(q_{d_k}k_{d_k})
           = 1 + 1 + ... + 1   (d_k vezes)
           = d_k
```

Logo, o desvio padrão do score é **`√d_k`**. Isso significa que quanto maior a dimensão `d_k`, maior a magnitude dos scores — eles crescem na raiz de `d_k`.

---

### Por que isso é um problema para o Softmax?

O softmax é definido como:

```
softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)
```

Quando os scores têm magnitude muito alta, a função exponencial **satura**: o maior valor domina completamente o resultado e os demais ficam próximos de zero. O softmax colapsa para algo parecido com uma função one-hot:

```
# d_k pequeno → scores moderados → distribuição suave
[0.3, 0.4, 0.3] 

# d_k grande → scores enormes → colapso
[0.001, 0.998, 0.001]  
```

Isso é conhecido como **colapso do softmax**, e tem duas consequências graves:

1. **Gradientes próximos de zero** — a derivada do softmax saturado é quase nula, impedindo que o modelo aprenda (vanishing gradient).
2. **Atenção "rígida"** — o modelo sempre foca em apenas um token, perdendo a capacidade de distribuir atenção entre múltiplas posições relevantes.

O **Teste 5** do `test_attention.py` demonstra exatamente isso, comparando a entropia das distribuições com e sem scaling:

```python
entropy = lambda w: -np.sum(w * np.log(w + 1e-12), axis=-1).mean()
h_no = entropy(weights_no_scale)  # entropia baixa → distribuição concentrada
h_sc = entropy(weights_scaled)    # entropia alta  → distribuição mais suave
assert h_sc > h_no  # scaling deve aumentar a entropia
```

---

### A Solução: Dividir por √d_k

Se a variância dos scores é `d_k`, basta dividir por `√d_k` para normalizá-la de volta a 1:

```
Var(score / √d_k) = Var(score) / d_k = d_k / d_k = 1
```

Isso é exatamente o que o código faz em `attention.py`:

```python
scaling_factor = np.sqrt(self.d_k)
scaled_scores = scores / scaling_factor       # variância ≈ 1 agora
attention_weights = self._softmax(scaled_scores)
```

O resultado é que os scores ficam em uma magnitude controlada, independente do valor de `d_k` escolhido. O softmax recebe entradas numa faixa razoável e produz uma distribuição de probabilidade equilibrada e estável.

---

### Visualização do Efeito

O `heatmap.py` mostra esse contraste visualmente nos dois painéis:

| Painel | O que mostra | Colormap |
|--------|-------------|----------|
| **Esquerdo** | Attention Weights após softmax com scaling | `viridis` (escala 0–1) |
| **Direito** | Scaled Scores brutos `QKᵀ / √d_k` antes do softmax | `coolwarm` (valores positivos e negativos) |

O painel da esquerda é o resultado "saudável": uma distribuição de probabilidade contínua onde é possível ver diferentes tons, indicando que a atenção está distribuída — não colapsada.

---

### Resumo

> Dividir por `√d_k` mantém a **variância dos scores igual a 1** independente da dimensão, evitando que o softmax sature e garantindo gradientes úteis durante o aprendizado.

É uma operação simples — uma única divisão — mas matematicamente motivada e essencial para que o mecanismo de atenção funcione corretamente em qualquer escala.

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

> Este README foi gerado com auxílio de Inteligência Artificial (Claude, Anthropic) e revisado por Ingrid
