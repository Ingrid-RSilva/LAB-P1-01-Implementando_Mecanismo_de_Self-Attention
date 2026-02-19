import numpy as np


class ScaledDotProductAttention:
    """
    Implementação manual do mecanismo de Scaled Dot-Product Attention.
    
    Fórmula:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Parâmetros:
        d_model (int): Dimensão do modelo (embedding size)
        d_k (int): Dimensão das matrizes Query e Key
        d_v (int): Dimensão da matriz Value
    """

    def __init__(self, d_model: int, d_k: int, d_v: int):
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        # Pesos de projeção linear para Q, K, V
        rng = np.random.default_rng(seed=42)
        self.W_q = rng.standard_normal((d_model, d_k)) * 0.1
        self.W_k = rng.standard_normal((d_model, d_k)) * 0.1
        self.W_v = rng.standard_normal((d_model, d_v)) * 0.1

    def _project(self, X: np.ndarray):
        """Projeta a entrada X nas matrizes Q, K, V via multiplicação linear."""
        Q = X @ self.W_q  # (seq_len, d_k)
        K = X @ self.W_k  # (seq_len, d_k)
        V = X @ self.W_v  # (seq_len, d_v)
        return Q, K, V

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        """
        Softmax numericamente estável aplicado em cada linha (axis=-1).
        Subtrai o máximo para evitar overflow.
        """
        shifted = scores - scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    def forward(self, X: np.ndarray):
        """
        Executa o forward pass do mecanismo de atenção.

        Parâmetros:
            X (np.ndarray): Tensor de entrada com shape (seq_len, d_model)

        Retorna:
            output (np.ndarray): Saída da atenção com shape (seq_len, d_v)
            attention_weights (np.ndarray): Pesos de atenção (seq_len, seq_len)
        """
        Q, K, V = self._project(X)

        # Produto escalar QK^T
        scores = Q @ K.T  # (seq_len, seq_len)

        # Scaling: dividir por sqrt(d_k)
        scaling_factor = np.sqrt(self.d_k)
        scaled_scores = scores / scaling_factor

        # Softmax por linha para obter os pesos de atenção
        attention_weights = self._softmax(scaled_scores)

        # Saída: média ponderada dos Values
        output = attention_weights @ V  # (seq_len, d_v)

        return output, attention_weights
