"""T4.138: the GatedDeltaNet PREFILL chunk's generic glue as NV kernels (the decode step has kernels/nv.py's gdn_decode since T4.123f+g).

Step 0 -- the map. Every generic kernel of one 128-token prefill chunk, identified from the schedule of ONE GatedDeltaNetBlock call at the
27B geometry (dim 5120, hidden 17408, 48 v-heads / 16 k-heads of 128, conv 4 taps over 10240 channels; symbolic `toks` <= 128) with the
NV gates mocked open on CPU, each generic kernel named by the sm_86 hand-coded optimizer -- the names the card's 10-31 profile shows
(logs/t4136/prof-2048/summary.txt; test_llm_nv_gdn_prep.py's census pins this list). ms/chunk = launches x us from that profile; bytes
at 128 tokens; floor = bytes / 936 GB/s, or the ~5 us launch floor.

 kernel (launches/chunk x us)   ms    op in model.py (GatedDeltaNetBlock._attention unless noted)             bytes moved         floor
 -- pre-scan (between attn_qkv/attn_gate/ssm_beta/ssm_alpha and gated_delta_prefill_nv) --------------------------------------------
 r_2_8_8_16_3_32_4  (48 x 124)  5.93  kq = (q*k).sum(-1) in nv.gated_delta_prefill: q, k RECOMPUTED inline per v-head      ~2.2 MB   5 us
                                      from the conv window (4-tap conv + SiLU, / the two norms below, * K^-0.5)
 r_128_16_16_8      (96 x 40)   3.85  the q and k L2 norms (sum of SiLU(conv)^2 over 128 per token x k-head), conv      1.1 MB each  5 us
                                      recomputed from the window
 E_16_16_8_8_16_3   (96 x 12)   1.17  q (* K^-0.5) and k normalized, .repeat'ed to 48 v-heads, materialized (1,48,T,128)  4.2 MB each  4.5 us
 E_48_2_16_16_8_4   (48 x 18)   0.85  v = SiLU(conv) of the 6144 v channels, materialized (1,48,T,128) fp32               6.3 MB    6.8 us
 E_28toks2A8029_128 (48 x 15)   0.73  attn_qkv's output (toks x 10240 fp32) copied into the window win[:, 3:3+T]          10.5 MB   11 us
 E_6_2_8_16_4       (96 x 7)    0.65  beta = sigmoid(ssm_beta(x)), alpha = exp(softplus(ssm_alpha(x)+dt_bias)*ssm_a),     6144 el.  5 us
                                      padded past T (beta 0, alpha 1)
 E_320_32_3         (96 x 6)    0.59  conv_state -> win[:, :3] (fp16 -> fp32), and conv_state <- win[:, T:T+3]           0.2 MB    5 us
 E_10480_32_4       (pruned)    --    the window's zero fill (1,131,10240): a write-only init, so the pruned prefill family runs it ONCE
                                      at capture -- the padded rows of a remainder chunk then read whatever the arena left there
 linear_f16_gemv_nv (98.6 x 47) 4.60  ssm_beta, ssm_alpha (5120 -> 48 fp16) at 128 tokens: one warp per (token, row)    1.8 MB DRAM  2 us
                                      re-reads a 10 KB weight row + 10 KB token row -- 126 MB of L2 traffic per launch
 E_5120_32_4        (49 x 9)    0.44  nv_f16_gemv's .contiguous() copy of the padded fp16 x (its input)                    2.6 MB    2.8 us
 sum, GDN pre-scan glue: ~287 us per block = 13.8 ms/chunk (the two f16 gemvs + the copy: another 4.8 ms)
 -- post-scan (gated_delta_prefill_nv's core -> ssm_out) -----------------------------------------------------------------------------
 r_8_48_8_4_4_16    (48 x 10)   0.47  ssm_norm's sum of squares over V per (token, v-head)                                3.1 MB    3.4 us
 E_32_6_2_8_16_4_4  (48 x 14)   0.65  z = ssm_norm(core) * SiLU(attn_gate) -> fp16                                        7.9 MB    8.4 us
 E_16_96_8_16_4     (48 x 7)    0.35  z[:, :T] padded back to 128 rows for ssm_out's q8 quantize (nv_wmma_linear)         3.1 MB    3.4 us
 q8_quantize_nv     (48 of 263) --    ssm_out's q8 activations (Q5_K on the card: + the group sums)                        2.6 MB    2.8 us
 -- every block's norm sites, quantizers and FFN (not this task's steps 1/2) --------------------------------------------------------
 q8_quantize_nv     (263 x 17)  4.57  q8 activations of every IMMA gemm, 4 per block: the attn input (5120 fp16), ssm_out/ 2.1/2.5/   avg
                                      attn_output (6144/... fp16), ffn input (5120 fp32), ffn_down (17408 fp32)        3.4/11.4 MB 5.2 us
 r_toks_16_320      (128 x 19)  2.37  RMSNorm sum of squares over 5120 per token: attn_norm and ffn_norm (FFNBlock)       2.6-5.2 MB 3-6 us
 E_4_80_8_16_4_4    (130 x 10)  1.26  RMSNorm apply + pad to 128 rows (the q8 input): attn_norm -> fp16 (GDN, also the    2.6-5.2 MB 3-6 us
                                      f16 gemv's input), ffn_norm -> fp32 with h = x + attn_out recomputed
 E_16_272_8_16_4    (65 x 37)   2.39  SiLU(gate) * up padded to 128 rows for ffn_down's q8 quantize (FFNBlock)            26.7 MB   28.5 us
 E_28toks2A13629_128 (64 x 25)  1.57  ffn_gate(x).silu().contiguous() (_feed_forward's "TODO: remove the need for this     17.8 MB   19 us
                                      contiguous")
 E_28toks2A4029_128 (65 x 15)   0.97  the block output x + attn_out + ffn_out (.contiguous())                              10.5 MB   11 us
"""
