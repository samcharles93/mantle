package toy

import "infer/internal/tensor"

// ToyLM is a minimal language model used for testing and benchmarking the
// kernels in this package.  It consists of an embedding matrix, a weight
// matrix for projecting hidden activations back to vocab logits, and a bias
// vector.  It is deliberately simplistic: each call to Forward operates on
// a single token and returns a new slice of logits.
type ToyLM struct {
    Vocab  int
    Hidden int

    Emb  tensor.Mat   // [Vocab x Hidden] embedding matrix
    W    tensor.Mat   // [Hidden x Vocab] projection weights
    Bias []float32    // [Vocab] bias added to logits
    h    []float32    // scratch space [Hidden]
}

// NewToyLM constructs a model with the given vocabulary and hidden size.  It
// initialises the embedding and weight matrices with random values derived
// from the provided seed.  Biases are zeroed.
func NewToyLM(vocab, hidden int, seed int64) *ToyLM {
    m := &ToyLM{
        Vocab:  vocab,
        Hidden: hidden,
        Emb:    tensor.NewMat(vocab, hidden),
        W:      tensor.NewMat(hidden, vocab),
        Bias:   make([]float32, vocab),
        h:      make([]float32, hidden),
    }
    // Deterministically fill embeddings and weights
    tensor.FillRand(&m.Emb, seed+11)
    tensor.FillRand(&m.W, seed+23)
    for i := range m.Bias {
        m.Bias[i] = 0
    }
    return m
}

// Forward computes the logits over the vocabulary for a single input token.
// If the token index is outside [0, Vocab), it is reduced modulo Vocab.
// The embedding vector is copied into scratch space, then a GEMV
// (matrix‑vector multiply) is performed with W and added bias.  A newly
// allocated slice is returned to hold the logits.
func (m *ToyLM) Forward(tok int) []float32 {
    // Wrap token index into the valid range.
    if tok < 0 || tok >= m.Vocab {
        tok = tok % m.Vocab
        if tok < 0 {
            tok += m.Vocab
        }
    }
    // h = Emb[tok]
    copy(m.h, m.Emb.Row(tok))
    // logits = h * W + bias
    logits := make([]float32, m.Vocab)
    for j := 0; j < m.Vocab; j++ {
        var sum float32
        for i := 0; i < m.Hidden; i++ {
            sum += m.h[i] * m.W.Row(i)[j]
        }
        logits[j] = sum + m.Bias[j]
    }
    return logits
}
