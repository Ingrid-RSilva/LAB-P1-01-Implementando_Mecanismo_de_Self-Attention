"""
test_attention.py
Testes de validação para o mecanismo Scaled Dot-Product Attention.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from attention import ScaledDotProductAttention


# Utilitário
def assert_close(actual, expected, tol=1e-6, msg=""):
    diff = np.max(np.abs(actual - expected))
    status = "PASSOU" if diff <= tol else "FALHOU"
    print(f"  {status} — {msg} (diferença máx: {diff:.2e})")
    assert diff <= tol, f"Falha em '{msg}': diff={diff}"


def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)

# 1. Shapes de saída
def test_output_shapes():
    section("1. Shapes de saída")
    seq_len, d_model, d_k, d_v = 5, 8, 4, 6
    attn = ScaledDotProductAttention(d_model, d_k, d_v)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((seq_len, d_model))

    output, weights = attn.forward(X)

    assert output.shape == (seq_len, d_v), \
        f"Shape do output esperado ({seq_len},{d_v}), obtido {output.shape}"
    assert weights.shape == (seq_len, seq_len), \
        f"Shape dos pesos esperado ({seq_len},{seq_len}), obtido {weights.shape}"

    print(f"PASSOU — output.shape == ({seq_len}, {d_v})")
    print(f"PASSOU — attention_weights.shape == ({seq_len}, {seq_len})")


# 2. Pesos de atenção somam 1 por linha
def test_attention_weights_sum_to_one():
    section("2. Pesos de atenção somam 1 (por linha)")
    attn = ScaledDotProductAttention(d_model=8, d_k=4, d_v=4)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((6, 8))
    _, weights = attn.forward(X)

    row_sums = weights.sum(axis=-1)
    assert_close(row_sums, np.ones(6), tol=1e-6,
                 msg="Cada linha de attention_weights soma 1.0")

# 3. Pesos de atenção estão em [0, 1]
def test_attention_weights_range():
    section("3. Pesos de atenção estão em [0, 1]")
    attn = ScaledDotProductAttention(d_model=8, d_k=4, d_v=4)
    rng = np.random.default_rng(2)
    X = rng.standard_normal((6, 8))
    _, weights = attn.forward(X)

    assert np.all(weights >= 0) and np.all(weights <= 1), \
        "Pesos fora do intervalo [0, 1]"
    print(f"PASSOU — Todos os pesos em [0, 1]"
          f"(min={weights.min():.4f}, max={weights.max():.4f})")


# 4. Exemplo numérico determinístico
def test_deterministic_numeric_example():
    """
    Valida o resultado passo a passo com valores controlados.
    Usa d_model=d_k=d_v=2, seq_len=2, pesos identidade para
    Q/K/V (para que Q=K=V=X) e verifica o cálculo manual.
    """
    section("4. Exemplo numérico determinístico (passo a passo)")

    d_model, d_k, d_v = 2, 2, 2
    attn = ScaledDotProductAttention(d_model, d_k, d_v)

    # Forçar pesos como identidade → Q = K = V = X
    attn.W_q = np.eye(d_model, d_k)
    attn.W_k = np.eye(d_model, d_k)
    attn.W_v = np.eye(d_model, d_v)

    X = np.array([[1.0, 0.0],
                  [0.0, 1.0]])

    output, weights = attn.forward(X)

    # ── Cálculo manual ──────────────────────────────
    Q = K = V = X                           # pesos identidade
    scores = Q @ K.T                        # [[1,0],[0,1]]
    scaled = scores / np.sqrt(d_k)

    # softmax linha 0: [1/sqrt2, 0] → exp → normaliza
    def softmax_row(row):
        r = row - row.max()
        e = np.exp(r)
        return e / e.sum()

    w0 = softmax_row(scaled[0])
    w1 = softmax_row(scaled[1])
    expected_weights = np.array([w0, w1])
    expected_output  = expected_weights @ V

    print(f"  X              =\n{X}")
    print(f"  scores (QKᵀ)  =\n{Q @ K.T}")
    print(f"  scaled_scores  =\n{scaled}")
    print(f"  attention_weights (esperado) =\n{expected_weights}")
    print(f"  attention_weights (obtido)   =\n{weights}")
    print(f"  output (esperado) =\n{expected_output}")
    print(f"  output (obtido)   =\n{output}")

    assert_close(weights, expected_weights, tol=1e-8,
                 msg="attention_weights bate com cálculo manual")
    assert_close(output,  expected_output,  tol=1e-8,
                 msg="output bate com cálculo manual")


# 5. Invariância ao scaling — não deve inflar norma
def test_scaling_reduces_large_scores():
    """
    Com d_k grande, sem scaling os scores explodem e o softmax
    colapsa para one-hot. Com scaling, a distribuição deve ser
    mais suave (entropia maior).
    """
    section("5. Scaling evita colapso do softmax")

    d_model, d_k, d_v = 16, 64, 16
    attn = ScaledDotProductAttention(d_model, d_k, d_v)
    rng = np.random.default_rng(7)
    X = rng.standard_normal((4, d_model))

    Q, K, V = attn._project(X)
    scores = Q @ K.T

    # Softmax SEM scaling
    weights_no_scale = attn._softmax(scores)
    # Softmax COM scaling
    weights_scaled   = attn._softmax(scores / np.sqrt(d_k))

    entropy = lambda w: -np.sum(w * np.log(w + 1e-12), axis=-1).mean()
    h_no = entropy(weights_no_scale)
    h_sc = entropy(weights_scaled)

    print(f"  Entropia média SEM scaling: {h_no:.4f}")
    print(f"  Entropia média COM scaling: {h_sc:.4f}")
    assert h_sc > h_no, "Scaling deve aumentar a entropia (distribuição mais suave)"
    print("PASSOU — Scaling produz distribuição mais suave")


# 6. Softmax numericamente estável
def test_softmax_numerical_stability():
    section("6. Softmax numericamente estável (valores grandes)")
    attn = ScaledDotProductAttention(4, 4, 4)

    # Scores muito grandes que causariam overflow sem o shift
    scores = np.array([[1000.0, 1001.0, 999.0]])
    result = attn._softmax(scores)

    assert not np.any(np.isnan(result)), "NaN detectado no softmax"
    assert not np.any(np.isinf(result)), "Inf detectado no softmax"
    assert_close(result.sum(axis=-1), np.array([1.0]), tol=1e-6,
                 msg="Softmax estável soma 1 mesmo com inputs grandes")


# Runner

if __name__ == "__main__":
    tests = [
        test_output_shapes,
        test_attention_weights_sum_to_one,
        test_attention_weights_range,
        test_deterministic_numeric_example,
        test_scaling_reduces_large_scores,
        test_softmax_numerical_stability,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"ERRO: {e}")
            failed += 1

    section(f"RESULTADO FINAL: {passed}/{len(tests)} testes passaram")
    if failed == 0:
        print("Todos os testes passaram!\n")
    else:
        print(f"{failed} teste(s) falharam.\n")
