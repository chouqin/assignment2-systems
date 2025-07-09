import torch
import triton
import triton.language as tl

import math


class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        nq = Q.shape[-2]
        d = Q.shape[-1]
        nk = K.shape[-2]
        bq = 16
        bk = 16
        tq = math.ceil(nq / bq)
        tk = math.ceil(nk / bk)

        O = torch.zeros(*Q.shape[:-2], nq, d, dtype=Q.dtype, device=Q.device)
        L = torch.zeros(*Q.shape[:-2], nq, dtype=Q.dtype, device=Q.device)
        
        for i in range(tq):
            start_i = i * bq
            end_i = min((i + 1) * bq, nq)
            Qi = Q[..., start_i:end_i, :]

            actual_bq = end_i - start_i
            Oi = torch.zeros(*Q.shape[:-2], actual_bq, d, dtype=Q.dtype, device=Q.device)
            li = torch.zeros(*Q.shape[:-2], actual_bq, dtype=Q.dtype, device=Q.device)
            mi = torch.full((*Q.shape[:-2], actual_bq), -math.inf, dtype=Q.dtype, device=Q.device)

            for j in range(tk):
                start_j = j * bk
                end_j = min((j + 1) * bk, nk)
                Kj = K[..., start_j:end_j, :]                   
                Vj = V[..., start_j:end_j, :]
                
                # Sij shape: (..., actual_bq, actual_bk) where actual_bk = end_j - start_j
                Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) / math.sqrt(d)
                current_mi = torch.max(mi, torch.max(Sij, dim=-1).values)
                Pij = torch.exp(Sij - current_mi.unsqueeze(-1))
                li = torch.exp(mi - current_mi) * li + torch.sum(Pij, dim=-1)
                # Correct matrix multiplication: Pij @ Vj where Pij is (..., actual_bq, actual_bk) and Vj is (..., actual_bk, d)
                Oi = torch.exp(mi - current_mi).unsqueeze(-1) * Oi + torch.matmul(Pij, Vj)
                mi = current_mi
            
            O[..., start_i:end_i, :] = Oi / li.unsqueeze(-1)
            L[..., start_i:end_i] = mi + torch.log(li)

        # Save tensors for backward pass (required by the test)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        
        return O

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE),
        order=(0,),
    )

    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    O = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    L = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    M = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(query_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(query_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        
        K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Compute attention scores: S = Q @ K^T / sqrt(d)
        S = tl.dot(Q, tl.trans(K)) * scale
        
        # Update running maximum
        M_new = tl.maximum(M, tl.max(S, axis=1))
        
        # Compute attention weights with numerical stability
        P = tl.exp(S - M_new[:, None])
        
        # Update running statistics
        alpha = tl.exp(M - M_new)
        L = alpha * L + tl.sum(P, axis=1)
        
        # Update output: O = alpha * O + P @ V
        O = (alpha[:, None] * O + tl.dot(P.to(V.dtype), V)).to(tl.float32)
        
        # Update running maximum
        M = M_new

        # advance K, V
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O = O / L[:, None]
    L = M + tl.log(L)
    
    tl.store(O_block_ptr, O, boundary_check=(0, 1))
    tl.store(L_block_ptr, L, boundary_check=(0))

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        nb = Q.shape[0]
        nq = Q.shape[-2]
        d = Q.shape[-1]
        nk = K.shape[-2]

        O = torch.zeros(*Q.shape[:-2], nq, d, dtype=Q.dtype, device=Q.device)
        L = torch.zeros(*Q.shape[:-2], nq, dtype=Q.dtype, device=Q.device)


        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_Size = 16

        flash_fwd_kernel[(math.cdiv(nq, ctx.Q_TILESize), nb)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            nq, nk,
            1 / math.sqrt(d),
            d,
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O