import torch
from math import ceil

# ---------------------------------------------------------------------------
# Global device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
TOL = 1e-8

# ---------------------------------------------------------------------------
# Complex Ginibre ensemble
# ---------------------------------------------------------------------------
def ginibre(num_kraus: int, d2: int, d1: int, num_samples: int) -> torch.Tensor:
    """
    Sample `num_samples` rectangular complex Ginibre matrices of shape (d2, d1).

    Entries ~ CN(0, 1):  G = (X + iY) / sqrt(2),  X, Y ~ N(0, 1)
    so E[|G_ij|^2] = 1  and  E[Tr(GG†)] = d1 * d2.

    Args:
        num_kraus:   Kraus rank of channels
        d2:          Output size
        d1:          Input size
        num_samples: Number of independent channels.

    Returns:
        Complex tensor of shape (num_samples, num_kraus, d2, d1).
    """
    if not all(isinstance(x, int) for x in (num_kraus, d2, d1, num_samples)):
        raise TypeError(
            f"num_kraus ({num_kraus}), d ({d2}), s ({d1}), num_samples ({num_samples}) must be integers."
        )
    if num_kraus <= 0 or d2 <= 0 or d1 <= 0 or num_samples <= 0:
        raise ValueError(
            f"num_kraus ({num_kraus}), d ({d2}), s ({d1}), num_samples ({num_samples}) must be positive."
        )
    n = num_kraus * num_samples
    real = torch.randn(n, d2, d1, device=DEVICE, dtype=torch.float64)
    imag = torch.randn(n, d2, d1, device=DEVICE, dtype=torch.float64)
    G = (real + 1j * imag) / 2 ** 0.5
    return G.reshape(num_samples, num_kraus, d2, d1)


# ---------------------------------------------------------------------------
# Kraus operators of a channel
# ---------------------------------------------------------------------------
def generate_kraus_channel(
        num_samples: int = 1,
        M: int = 4,
        dim: tuple[int, int] = (2, 2)
    ) -> torch.Tensor:
    """
    Sample random CPTP maps as Kraus operators.

    Args:
        num_samples: Number of independent channels.
        M:           Kraus rank. Must satisfy M * d2 >= d1.
        dim:         (d_out, d_in) = (d2, d1).

    Returns:
        K: (num_samples, M, d_out, d_in)
    """
    d2, d1 = dim
    if not all(isinstance(x, int) for x in (d1, d2, M, num_samples)):
        raise TypeError(
            f"d1 ({d1}), d2 ({d2}), M ({M}), num_samples ({num_samples}) must be integers."
        )
    if d1 <= 0 or d2 <= 0:
        raise ValueError(f"d1 ({d1}), d2 ({d2}) must be positive integers.")
    if M <= 0:
        raise ValueError(f"M ({M}) must be a positive integer.")
    if M * d2 < d1:
        raise ValueError(
            f"kraus_rank ({M}) must satisfy kraus_rank * d2 >= d1 "
            f"(i.e. >= ceil({d1}/{d2}) = {ceil(d1 / d2)})."
        )
    # if M > d1 * d2:
    #     raise ValueError(f"M ({M}) exceeds upper bound d1*d2 = {d1 * d2}.")

    # 1. Ginibre matrices G: (n, M, d2, d1)
    G = ginibre(num_kraus=M, d2=d2, d1=d1, num_samples=num_samples)

    # 2. H = sum_a G_a† G_a: (n, d1, d1)
    H = (G.conj().transpose(-2, -1) @ G).sum(dim=1)

    # 3. H^{-1/2}: (n, d1, d1)
    S, V = torch.linalg.eigh(H)
    S = torch.clamp(S, min=1e-12)
    S_inv_sqrt = torch.diag_embed(S.rsqrt()).to(H.dtype)
    H_inv_sqrt = V @ S_inv_sqrt @ V.mH

    # 4. Kraus operators K = G H^{-1/2}: (n, M, d2, d1)
    K = G @ H_inv_sqrt.unsqueeze(1)
    return K


# ---------------------------------------------------------------------------
# Kraus operators of 1-slot quantum combs
# ---------------------------------------------------------------------------
def generate_one_slot_kraus_comb(
    num_samples: int = 1,
    sys_dim: tuple[int, int, int, int] = (2, 2, 2, 2),
    env_dim: int = 2,
    r_enc: int = 1,
    r_dec: int = 16,
) -> torch.Tensor:
    """
    Sample 1-slot quantum combs by linking an Encoder and Decoder via an environment.

    Returns
    -------
    combs : torch.Tensor
        Shape (n, r_enc * r_dec, d_2i * d_0o, d_1o * d_1i)
        Each Kraus operator maps (1o, 1i) -> (2i, 0o).
    """
    d_2i, d_1o, d_1i, d_0o = sys_dim
    d_env = env_dim

    # Encoder: d_0o -> (d_1i, d_env): (n, r_enc, d_1i*d_env, d_0o)
    E = generate_kraus_channel(num_samples=num_samples, dim=(d_1i * d_env, d_0o), M=r_enc)

    # Decoder: (d_1o, d_env) -> d_2i: (n, r_dec, d_2i, d_1o*d_env)
    D = generate_kraus_channel(num_samples=num_samples, dim=(d_2i, d_1o * d_env), M=r_dec)

    # Reshape into subsystem tensors
    E_t = E.reshape(num_samples, r_enc, d_1i, d_env, d_0o)   # (n, r_enc, d_1i, d_env, d_0o)
    D_t = D.reshape(num_samples, r_dec, d_2i, d_1o, d_env)   # (n, r_dec, d_2i, d_1o, d_env)

    # Link on shared environment: (n, r_enc, r_dec, d_2i, d_1o, d_1i, d_0o)
    linked = torch.einsum("naied,nbtje->nabtjid", E_t, D_t)

    # Merge Kraus indices: (n, r_enc*r_dec, d_2i, d_1o, d_1i, d_0o)
    linked = linked.reshape(num_samples, r_enc * r_dec, d_2i, d_1o, d_1i, d_0o)

    # output=(2i,0o), input=(1o,1i): (n, r_enc*r_dec, d_2i*d_0o, d_1o*d_1i)
    combs = linked.permute(0, 1, 2, 5, 3, 4).reshape(
        num_samples, r_enc * r_dec, d_2i * d_0o, d_1o * d_1i
    )
    return combs


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def is_kraus_tp(K: torch.Tensor, tol: float) -> torch.Tensor:
    """
    Check TP condition sum_a K_a† K_a = I for a batch of Kraus sets.

    Args:
        K:   (n, M, d_out, d_in)
        tol: absolute tolerance

    Returns:
        Bool tensor of shape (n,)
    """
    K_dag = K.conj().transpose(-2, -1)                          # (n, M, d_in, d_out)
    H = (K_dag @ K).sum(dim=1)                                  # (n, d_in, d_in)
    d_in = K.shape[-1]
    I = torch.eye(d_in, dtype=K.dtype, device=K.device)        # (d_in, d_in)
    return (H - I).abs().amax(dim=(-2, -1)) < tol              # (n,)


def kraus_to_choi(K: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of Kraus operators to Choi matrices.

    J[(m,i),(n,j)] = sum_a K_a[m,i] * conj(K_a[n,j])
    Layout: (d_out, d_in, d_out, d_in) -> (d_out*d_in, d_out*d_in)
    Tr_out(J)_ij = sum_m J[(m,i),(m,j)] = sum_a (K_a†K_a)_ij = I  ✓

    Args:
        K: (n, M, d_out, d_in)

    Returns:
        J: (n, d_out*d_in, d_out*d_in)
    """
    n, _, d_out, d_in = K.shape
    J4 = torch.einsum("cami,canj->cminj", K, K.conj())         # (n, d_out, d_in, d_out, d_in)
    return J4.reshape(n, d_out * d_in, d_out * d_in)


def is_psd(J: torch.Tensor, atol: float) -> torch.Tensor:
    """
    Check PSD condition for a batch of matrices.

    Args:
        J:    (n, d, d)
        atol: absolute tolerance

    Returns:
        Bool tensor of shape (n,)
    """
    evals = torch.linalg.eigvalsh(J)                           # (n, d)
    return (evals >= -atol).all(dim=-1)                        # (n,)


def is_choi_tp(J: torch.Tensor, d_out: int, d_in: int, atol: float) -> torch.Tensor:
    """
    Check Tr_out(J) = I_{d_in} for a batch of Choi matrices.
    J stored with layout (d_out, d_in, d_out, d_in).

    Args:
        J:     (n, d_out*d_in, d_out*d_in)
        d_out: output dimension
        d_in:  input dimension
        atol:  absolute tolerance

    Returns:
        Bool tensor of shape (n,)
    """
    n = J.shape[0]
    J4 = J.reshape(n, d_out, d_in, d_out, d_in)               # (n, m, i, m', j)
    J2 = torch.einsum("naiaj->nij", J4)                        # (n, d_in, d_in)
    I  = torch.eye(d_in, dtype=J.dtype, device=J.device)
    return (J2 - I).abs().amax(dim=(-2, -1)) < atol            # (n,)


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    total_start = time.perf_counter()

    print(f"Device       : {DEVICE}")
    print(f"Default Dtype: {torch.get_default_dtype()}")
    print(f"Tolerance    : {TOL}\n")

    # --- Setup Parameters ---
    num_samples = 20000
    r_sys = 3
    r_enc = 1
    r_dec = 4
    d_2i  = 2
    d_1o  = 2
    d_1i  = 2
    d_0o  = 2
    d_env = 4

    print("--- Settings ---")
    print(f"  num_samples : {num_samples}")
    print(f"  r_sys       : {r_sys}  (Kraus rank of input channel)")
    print(f"  r_enc       : {r_enc}  (Kraus rank of encoder)")
    print(f"  r_dec       : {r_dec}  (Kraus rank of decoder)")
    print(f"  d_2i        : {d_2i}  (output dim of decoder)")
    print(f"  d_1o        : {d_1o}  (input dim of decoder / output dim of channel)")
    print(f"  d_1i        : {d_1i}  (output dim of encoder / input dim of channel)")
    print(f"  d_0o        : {d_0o}  (input dim of encoder)")
    print(f"  d_env       : {d_env}  (environment dim)\n")

    # --- Step 1: Channels ---
    print("--- Step 1: Generating Input Channels ---")
    t0 = time.perf_counter()
    channels = generate_kraus_channel(num_samples=num_samples, M=r_sys, dim=(d_1o, d_1i))
    print(f"  Generated {len(channels)} channels.  [{time.perf_counter()-t0:.3f}s]")

    t0 = time.perf_counter()
    tp = is_kraus_tp(channels, TOL)
    print(f"  Kraus TP success: {tp.sum().item()}/{num_samples}  [{time.perf_counter()-t0:.3f}s]\n")

    # --- Step 2: Combs ---
    print("--- Step 2: Generating Combs ---")
    t0 = time.perf_counter()
    combs = generate_one_slot_kraus_comb(
        num_samples=num_samples,
        sys_dim=(d_2i, d_1o, d_1i, d_0o),
        env_dim=d_env,
        r_enc=r_enc,
        r_dec=r_dec,
    )
    print(f"  Generated {len(combs)} combs.  [{time.perf_counter()-t0:.3f}s]\n")

    # --- Step 3: Evaluate Choi matrices (fully batched, no loop) ---
    print("--- Step 3: Evaluating Choi Matrices ---")
    t0 = time.perf_counter()

    # (n, d_1o*d_1i, d_1o*d_1i)
    J_in = kraus_to_choi(channels)

    # Supermap: J_out[n,o,p] = sum_a K[n,a,o,i] J_in[n,i,j] K*[n,a,p,j]
    # (n, d_2i*d_0o, d_2i*d_0o)
    J_out = torch.einsum("naoi,nij,napj->nop", combs, J_in, combs.conj())

    in_psd  = is_psd(J_in,  TOL)
    in_tp   = is_choi_tp(J_in,  d_1o, d_1i, TOL)
    out_psd = is_psd(J_out, TOL)
    out_tp  = is_choi_tp(J_out, d_2i, d_0o, TOL)

    print(f"  Evalauted {len(J_out)} output Choi matrices.  [{time.perf_counter()-t0:.3f}s]\n")

    print("--- Results ---")
    print(f"  Input  Choi PSD success: {in_psd.sum().item()}/{num_samples}")
    print(f"  Input  Choi TP  success: {in_tp.sum().item()}/{num_samples}")
    print(f"  Output Choi PSD success: {out_psd.sum().item()}/{num_samples}")
    print(f"  Output Choi TP  success: {out_tp.sum().item()}/{num_samples}")
    print(f"\nTotal time: {time.perf_counter()-total_start:.3f}s")