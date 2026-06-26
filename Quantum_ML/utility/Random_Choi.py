import torch
import math

# ---------------------------------------------------------------------------
# Global device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
TOL = 1e-5
# ---------------------------------------------------------------------------
# Complex Ginibre ensemble
# ---------------------------------------------------------------------------
def ginibre(d: int, s: int, num_samples: int) -> torch.Tensor:
    """
    Sample `num_samples` rectangular complex Ginibre matrices of shape (d2, d1).

    Entries ~ CN(0, 1):  G = (X + iY) / sqrt(2),  X, Y ~ N(0, 1)
    so E[|G_ij|^2] = 1  and  E[Tr(GG†)] = d1 * d2.

    Args:
        d:          Number of rows.
        s:          Number of columns.
        num_samples: Number of independent matrices.

    Returns:
        Complex tensor of shape (num_samples, d1, d2).
    """
    if not all(isinstance(x, int) for x in (d, s, num_samples)):
        raise TypeError(
            f"d ({d}), s ({s}), num_samples ({num_samples}) must be integers."
        )
    if d <= 0 or s <= 0 or num_samples <= 0:
        raise ValueError(
            f"d ({d}), s ({s}), num_samples ({num_samples}) must be positive."
        )

    real = torch.randn(num_samples, d, s, device=DEVICE, dtype=torch.float64)
    imag = torch.randn(num_samples, d, s, device=DEVICE, dtype=torch.float64)
    return (real + 1j * imag) / 2 ** 0.5       # CN(0,1) entries


# ---------------------------------------------------------------------------
# Choi matrix of a channel
# ---------------------------------------------------------------------------
def generate_choi_channel(num_samples: int = 1, dim: tuple[int, int] =[8,2], M: int=4) -> torch.Tensor:
    """
    Sample Choi states of a random CPTP map.

    Args:
        d1:          Input Hilbert space dimension.
        d2:          Output Hilbert space dimension.
        M:           M in {d1/d2,...,d1d2-1} -> Md2>=d1.
        num_samples: Number of independent sets of Kraus operators.

    Returns:
        Choi state of a channel.
    """
    d1, d2 = dim
    if not all(isinstance(x, int) for x in (d1, d2, M, num_samples)):
        raise TypeError(
            f"d1 ({d1}), d2 ({d2}), M ({M}), "
            f"num_samples ({num_samples}) must be integers."
        )
    if d1 <= 0 or d2 <= 0:
        raise ValueError(f"d1 ({d1}), d2 ({d2}) must be positive integers.")
    if M <= 0:
        raise ValueError(f"M ({M}) must be a positive integer.")
    if M * d2 < d1:
        raise ValueError(
            f"kraus_rank ({M}) must satisfy kraus_rank * d2 >= d1 "
            f"(i.e. >= ceil({d1}/{d2}) = {math.ceil(d1 / d2)}) "
            f"to ensure sum random matrix H is invertible."
        )
    if M > d1 * d2:
        raise ValueError(
            f"M ({M}) exceeds upper bound d1*d2 = {d1 * d2}."
        )

    d = d1 * d2

    # 1. Ginibre matrices G: (n, d1*d2, M)
    G = ginibre(d, M, num_samples)

    # 2. Wishart matrix W: (n, d, d)
    W = G @ G.conj().transpose(-2, -1)

    # 3. Partial trace over subsystem d2
    W4 = W.reshape(num_samples, d2, d1, d2, d1)   # (n, d2, d1, d2, d1)
    H = torch.einsum('nabac->nbc', W4)             # (n, d1, d1)

    # 4. Inverse square root of H
    S, V = torch.linalg.eigh(H)
    S = torch.clamp(S, min=1e-12)
    S_inv_sqrt = torch.diag_embed(S.rsqrt()).to(W.dtype)
    H_inv_sqrt = V @ S_inv_sqrt @ V.mH             # (n, d1, d1)

    # 5. Choi
    id_d2 = torch.eye(d2, dtype=W.dtype, device=W.device) #(d2, d2)
    A = torch.einsum('ab,nij->naibj', id_d2, H_inv_sqrt).reshape(num_samples, d, d)
    
    choi = A @ W @ A

    if not check_choi_psd(choi) or not check_channel_tp(choi, dim):
        raise ValueError("Fail to generate channel.")
    return choi

# ---------------------------------------------------------------------------
# Choi matrix of a 1-slot quantum comb
# ---------------------------------------------------------------------------
def generate_one_slot_choi_combs(num_samples : int = 1,
                                  dim: tuple [int,int,int,int] = [2,2,2,2],
                                  r_enc: int = 4,
                                  r_dec: int = 16) -> torch.Tensor:
    """
    Sample 1-slot quantum combs
    num_samples: # of samples
    dim = (d_0o, d_1i, d_1o, d_2i): Hilbert space dimension of system 0o, 1i, 1o, 2i
    Kraus ranks of encoder and decoder is, almost surely,  
                min(d_0o*d_1i, r_enc) and min(d_1i*d_1o*d_2i*d_2i, r_enc)
    """
    d_0o, d_1i, d_1o, d_2i = dim
    d = math.prod(dim)
    # 1. Generate two quantum channels related to the encoder and decoder of a single-slot comb
    E_0o1i = generate_choi_channel(num_samples = num_samples,
                          dim = [d_0o, d_1i],
                          M = r_enc)
    
    status = check_choi_psd(E_0o1i) and check_channel_tp(E_0o1i, [d_0o, d_1i])
    if not status:
        raise ValueError(f"Fail to generate E_0o1i (Not CPTP)")
    
    D_2i0o = generate_choi_channel(num_samples = num_samples,
                          dim = [d_1i*d_1o*d_2i, d_2i],
                          M = r_dec)
    status = check_choi_psd(D_2i0o) and check_channel_tp(D_2i0o, [d_1i*d_1o*d_2i, d_2i])
    if not status:
        raise ValueError(f"Fail to generate D_2i0o (Not CPTP)")
    
    # 2. Compute id ⊗ √E_0o1i    
    S, V = torch.linalg.eigh(E_0o1i)
    S = torch.clamp(S, min=1e-12)
    S_sqrt = torch.diag_embed(S.sqrt()).to(E_0o1i.dtype)
    E_0o1i_sqrt = V @ S_sqrt @ V.mH        # (n,d_0o*d_1i, d_0o*d_1i)     
    id_2i1o = torch.eye(d_2i*d_1o, dtype=E_0o1i.dtype, device=E_0o1i.device)  # (d_2i*d_1o, d_2i*d_1o)
    A = torch.einsum('nij,ab->naibj', E_0o1i_sqrt, id_2i1o).reshape(num_samples, d, d)
    
    comb = A @ D_2i0o @ A

    if not check_choi_psd(comb) or not check_comb_tp(comb):
        raise ValueError("Fail to generate comb.")
    return comb


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def check_choi_hermitian(choi: torch.Tensor) -> bool:
    """Check Choi matrix is Hermitian: C = C†."""
    return torch.allclose(choi, choi.mH, atol=TOL)


def check_choi_psd(choi: torch.Tensor) -> bool:
    """Check Choi matrix is positive semidefinite (all eigenvalues >= 0)."""
    eigs = torch.linalg.eigvalsh(choi)
    return bool((eigs >= -TOL).all())


def check_channel_tp(choi: torch.Tensor,
                     dim: tuple[int, int]) -> bool:
    """
    Check the Trace-Preserving (TP) condition: Tr_out(Choi) = I_in.
    Supports both single matrices (d, d) and batches (N, d, d).
    """
    d1, d2 = dim
    # 1. Standardize shape to (Batch, Total_Dim, Total_Dim)
    is_batch = (choi.ndim == 3)
    if not is_batch:
        choi = choi.unsqueeze(0)  # (1, d1*d2, d1*d2)
    
    num_samples, total_dim, _ = choi.shape
    if total_dim != d1 * d2:
        raise ValueError(f"Shape mismatch: Expected {d1*d2}, got {total_dim}")

    # 2. Reshape to separate input and output spaces
    # Shape: (N, d2, d1, d2, d1)
    choi_r = choi.reshape(num_samples, d2, d1, d2, d1)

    # 3. Partial trace over the output space (subsystem d2)
    tr_out = torch.einsum('nabac->nbc', choi_r)

    # 4. Compare to Identity matrix of size d1
    identity = torch.eye(d1, dtype=choi.dtype, device=choi.device)
    identity = identity.expand(num_samples, d1, d1)
    
    return torch.allclose(tr_out, identity, atol=TOL)

def check_comb_tp(
    choi: torch.Tensor,
    dim: tuple[int, int, int, int] = (2, 2, 2, 2)
) -> bool:
    """
    Check the generalized TP condition for a 1-slot quantum comb:

        Tr_{2i}(C) = I_{1o} otimes R

    where R is itself a valid channel Choi matrix on 0o -> 1i.

    Assumes the composite basis ordering:
        (2i, 1o, 1i, 0o)
    on both bra and ket sides.
    """
    d0o, d1i, d1o, d2i = dim
    d = d0o * d1i * d1o * d2i

    # Standardize to batch shape (N, d, d)
    if choi.ndim == 2:
        choi = choi.unsqueeze(0)

    num_samples, total_dim, total_dim2 = choi.shape
    if total_dim != d or total_dim2 != d:
        raise ValueError(f"Shape mismatch: expected (..., {d}, {d}), got {choi.shape}")

    # Reshape as (N, 2i,1o,1i,0o, 2i,1o,1i,0o)
    C = choi.reshape(num_samples, d2i, d1o, d1i, d0o, d2i, d1o, d1i, d0o)

    # 1) Trace out the final output system 2i:
    #    T has shape (N, 1o,1i,0o, 1o,1i,0o)
    T = C.diagonal(dim1=1, dim2=5).sum(-1)

    # 2) Extract the reduced channel Choi R by tracing out 1o on both sides:
    #    Tr_{1o}(T) = d1o * R
    #    R has shape (N, 1i,0o, 1i,0o)
    R = T.diagonal(dim1=1, dim2=4).sum(-1) / d1o

    # 3) Rebuild I_{1o} \otimes R and compare with T
    I_1o = torch.eye(d1o, dtype=choi.dtype, device=choi.device)
    rhs = torch.einsum('ab,ncdef->nacdbef', I_1o, R)

    comb_ok = torch.allclose(T, rhs, atol=TOL)

    # 4) Check that R is itself a TP channel Choi on 0o -> 1i
    R_mat = R.reshape(num_samples, d1i * d0o, d1i * d0o)
    channel_ok = check_channel_tp(R_mat, (d0o, d1i))

    return comb_ok and channel_ok

# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    total_start = time.perf_counter()
    print(f"Device: {DEVICE}\n")

    # --- Channel test ---
    dim = [2, 2]
    M = 4
    num = 20000
    print(f"Config: input_dim={dim[0]}, output_dim={dim[1]}, M={M}")
    print("--- Channel test ---")
    t_channel = time.perf_counter()

    choi = generate_choi_channel(num, dim, M)
    print(f"Choi of channels shape : {tuple(choi.shape)}")
    print(f"Kraus rank             : {torch.linalg.matrix_rank(choi, atol=TOL, rtol=TOL)}")
    print(f"Hermitian              : {check_choi_hermitian(choi)}")
    print(f"PSD                    : {check_choi_psd(choi)}")
    print(f"Causality              : {check_channel_tp(choi, dim)}")
    print(f"Channel test time      : {time.perf_counter()-t_channel:.3f}s")

    # --- Comb test ---
    print("\n--- Comb test ---")
    dim = [2, 2, 2, 2]
    r_enc = 1
    r_dec = 4
    print(f"Config: (d0o,d1i,d1o,d2i)={tuple(dim)}, r_enc={r_enc}, r_dec={r_dec}")
    t_comb = time.perf_counter()

    choi = generate_one_slot_choi_combs(num, dim, r_enc, r_dec)
    print(f"Choi of comb shape     : {tuple(choi.shape)}")
    print(f"Kraus rank             : {torch.linalg.matrix_rank(choi, atol=TOL, rtol=TOL)}")
    print(f"Hermitian              : {check_choi_hermitian(choi)}")
    print(f"PSD                    : {check_choi_psd(choi)}")
    print(f"Causality              : {check_comb_tp(choi, dim)}")
    print(f"Comb test time         : {time.perf_counter()-t_comb:.3f}s")

    print(f"\nTotal time             : {time.perf_counter()-total_start:.3f}s")
