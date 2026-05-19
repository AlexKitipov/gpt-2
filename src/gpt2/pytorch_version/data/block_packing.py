def pack_tokens(tokens, block_size):
    """Split a long 1D token tensor into fixed-size blocks."""
    total_len = (len(tokens) // block_size) * block_size
    tokens = tokens[:total_len]
    return tokens.view(-1, block_size)
