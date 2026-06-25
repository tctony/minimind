def model_parameter_count(vocab_size=6400,
                          hidden_size=512,
                          n_layers=8,
                          n_atten_heads=8,
                          n_kv_heads=2):
    count = 0

    # lm head (encode 和 decode 共享)
    count += hidden_size * vocab_size

    intermediate_size = (int(hidden_size * 8 / 3) + 64 - 1 ) // 64 * 64
    head_dim = hidden_size // n_atten_heads

    def gqa_count():
        c = 0
        c += hidden_size  # norm
        c += hidden_size * (n_atten_heads * head_dim)  # q_proj
        c += hidden_size * (n_kv_heads * head_dim)     # k_proj (GQA: fewer heads)
        c += hidden_size * (n_kv_heads * head_dim)     # v_proj (GQA: fewer heads)
        c += (n_atten_heads * head_dim) * hidden_size  # o_proj
        return c

    def ffn_count():
        c = 0
        c += hidden_size # norm
        # SwiGLU
        c += hidden_size * intermediate_size # gate
        c += hidden_size * intermediate_size # up
        c += intermediate_size * hidden_size # down
        return c

    def layer_count():
        lc = 0
        lc += gqa_count()
        lc += ffn_count()
        return lc

    count += n_layers * layer_count()

    count += hidden_size  # norm

    print(f"total parameter count: {count/1e6:.2f}M")


if __name__ == '__main__':
    model_parameter_count()
