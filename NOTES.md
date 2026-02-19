# NOTES

- We don't currently parse "conv_bias" or "block_use_swiglu", if a model were to set these to true, the inference would be wrong because we'd never load the tensor.
