import torch
from torch import Tensor
from typing import Optional
import string

class KrausOperators:
    """
    Represents a quantum channel via its vectorised Kraus operators |K_i>>.

    Each vectorised Kraus operator is obtained by column-stacking the
    d_out × d_in matrix K_i into a single vector of length d_out * d_in,
    consistent with the convention |K_i>> = vec(K_i).

    Attributes
    ----------
    vec_kraus : list[Tensor]
        Vectorised Kraus operators, each of shape (d_out * d_in,),
        dtype torch.complex128.
    systems : dict[str, int]
        System label → dimension, e.g. {"0o": 2, "1i": 2}.
        Labels conventionally end in 'o' (output) or 'i' (input).
    """

    def __init__(
        self,
        vec_kraus: list[Tensor],
        systems: dict[str, int],
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Parameters
        ----------
        vec_kraus : list[Tensor]
            Vectorised Kraus operators |K_i>>. Each tensor must be 1-D.
        systems : dict[str, int]
            System label → dimension mapping.
        device : torch.device, optional
            Target device. Defaults to CPU.
        """
        self.device = device or torch.device("cpu")
        self.systems = dict(systems)
        self.vec_kraus = [
            v.to(dtype=torch.complex128, device=self.device)
            if isinstance(v, Tensor)
            else torch.tensor(v, dtype=torch.complex128, device=self.device)
            for v in vec_kraus
        ]
        self._validate()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        for i, v in enumerate(self.vec_kraus):
            if v.ndim != 1:
                raise ValueError(
                    f"vec_kraus[{i}] must be 1-D, got shape {tuple(v.shape)}."
                )

    def _dim(self, label: str) -> int:
        if label not in self.systems:
            raise KeyError(
                f"System '{label}' not found. Available: {list(self.systems)}."
            )
        return self.systems[label]

    def _total_dim(self, labels: list[str]) -> int:
        d = 1
        for lbl in labels:
            d *= self._dim(lbl)
        return d

    def _unvectorise(
        self,
        output_labels: list[str],
        input_labels: list[str],
    ) -> list[Tensor]:
        """
        Reshape each |K_i>> back to a (d_out, d_in) matrix.

        Uses column-stacking (Fortran / 'F') convention:
            vec(K) stacks columns -> reshape with order='F'
        Equivalent in PyTorch: reshape(d_in, d_out).T
        """
        d_out = self._total_dim(output_labels)
        d_in  = self._total_dim(input_labels)

        matrices = []
        for v in self.vec_kraus:
            if v.numel() != d_out * d_in:
                raise ValueError(
                    f"Vectorised operator has length {v.numel()}, expected "
                    f"d_out * d_in = {d_out} * {d_in} = {d_out * d_in}."
                )
            # Fortran unvec: reshape as (d_in, d_out) then transpose
            K = v.reshape(d_in, d_out).T.contiguous()
            matrices.append(K)
        return matrices

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def kraus_matrices(
        self,
        output_labels: list[str],
        input_labels: list[str],
    ) -> list[Tensor]:
        """
        Return un-vectorised Kraus matrices K_i as (d_out, d_in) tensors.

        Parameters
        ----------
        output_labels : list[str]
            Ordered output system labels, e.g. ["1i"].
        input_labels : list[str]
            Ordered input system labels, e.g. ["0o"].
        """
        return self._unvectorise(output_labels, input_labels)

    def is_tp(
        self,
        output_labels: list[str],
        input_labels: list[str],
        atol: float = 1e-8,
    ) -> bool:
        """
        Check whether the channel satisfies the trace-preserving (TP) condition
        for the specified input and output systems.

        A channel is TP iff:

            sum_i  K_i† K_i  =  I_{d_in}

        Parameters
        ----------
        output_labels : list[str]
            Labels of the output systems, e.g. ["1i"].
        input_labels : list[str]
            Labels of the input systems, e.g. ["0o"].
        atol : float
            Absolute tolerance for the identity check.

        Returns
        -------
        bool
            True if the completeness relation holds within `atol`.

        Examples
        --------
        >>> channel.is_tp(output_labels=["1i"], input_labels=["0o"])
        True
        """
        matrices = self._unvectorise(output_labels, input_labels)
        d_in = self._total_dim(input_labels)

        identity = torch.eye(d_in, dtype=torch.complex128, device=self.device)

        # sum_i K_i† K_i
        completeness = sum(K.conj().T @ K for K in matrices)

        return torch.allclose(completeness, identity, atol=atol)

    def link(self, other: "KrausOperators") -> "KrausOperators":
        self_keys = list(self.systems.keys())
        other_keys = list(other.systems.keys())

        shared_keys = [k for k in self_keys if k in other.systems]

        remaining_self = [k for k in self_keys if k not in shared_keys]
        remaining_other = [k for k in other_keys if k not in shared_keys]

        # Check shared dimensions match
        for k in shared_keys:
            if self.systems[k] != other.systems[k]:
                raise ValueError(
                    f"Dimension mismatch for shared system {k}: "
                    f"{self.systems[k]} != {other.systems[k]}"
                )

        # Build unique system label list
        all_keys = []
        for k in self_keys + other_keys:
            if k not in all_keys:
                all_keys.append(k)

        letters = list(string.ascii_letters)
        if len(all_keys) > len(letters):
            raise ValueError("Too many systems for plain einsum labels.")

        key_to_char = {k: letters[i] for i, k in enumerate(all_keys)}

        subs_self = "".join(key_to_char[k] for k in self_keys)
        subs_other = "".join(key_to_char[k] for k in other_keys)

        # If no shared keys, this becomes tensor product:
        # output keeps all axes from self then all axes from other
        out_keys = remaining_self + remaining_other if shared_keys else self_keys + other_keys
        subs_out = "".join(key_to_char[k] for k in out_keys)

        einsum_expr = f"{subs_self},{subs_other}->{subs_out}"

        shape_self = [self.systems[k] for k in self_keys]
        shape_other = [other.systems[k] for k in other_keys]

        new_vecs = []
        for v1 in self.vec_kraus:
            t1 = v1.reshape(*shape_self)
            for v2 in other.vec_kraus:
                t2 = v2.reshape(*shape_other)
                res = torch.einsum(einsum_expr, t1, t2)
                new_vecs.append(res.reshape(-1))

        if shared_keys:
            new_systems = {k: self.systems[k] for k in remaining_self}
            new_systems.update({k: other.systems[k] for k in remaining_other})
        else:
            new_systems = dict(self.systems)
            new_systems.update(other.systems)

        return KrausOperators(new_vecs, new_systems, device=self.device)

        

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.vec_kraus)

    def __repr__(self) -> str:
        n = len(self.vec_kraus)
        sys_str = ", ".join(f"{k}: {v}" for k, v in self.systems.items())
        return (
            f"KrausOperators(n_operators={n}, systems={{{sys_str}}}, "
            f"device={self.device})"
        )


# ======================================================================
# Convenience factory
# ======================================================================

def from_kraus_matrices(
    matrices: list[Tensor],
    systems: dict[str, int],
    device: Optional[torch.device] = None,
) -> KrausOperators:
    """
    Build a KrausOperators instance from ordinary (un-vectorised) K_i matrices.

    Parameters
    ----------
    matrices : list[Tensor]
        List of (d_out, d_in) Kraus matrices.
    systems : dict[str, int]
        System label -> dimension mapping.
    device : torch.device, optional
        Target device.
    """
    # Column-stack vectorisation: vec(K) = K.T.flatten()
    vec_kraus = [K.T.contiguous().flatten() for K in matrices]
    return KrausOperators(vec_kraus, systems, device=device)


# ======================================================================
# Quick demo / smoke-test
# ======================================================================

if __name__ == "__main__":
    # --- Example 1: qubit depolarising channel (TP) ---
    p = 0.1
    I = torch.eye(2, dtype=torch.complex128)
    X = torch.tensor([[0, 1], [1,  0]], dtype=torch.complex128)
    Y = torch.tensor([[0,-1j],[1j, 0]], dtype=torch.complex128)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

    depol_matrices = [
        (1 - p) ** 0.5 * I,
        (p / 3)  ** 0.5 * X,
        (p / 3)  ** 0.5 * Y,
        (p / 3)  ** 0.5 * Z,
    ]
    channel = from_kraus_matrices(
        depol_matrices,
        systems={"0o": 2, "1i": 2},
    )
    print(channel)
    print(
        "Depolarising TP (output=['0o'], input=['1i']):",
        channel.is_tp(output_labels=["0o"], input_labels=["1i"]),
    )

    # --- Example 2: label swap - "1i" as output, "0o" as input ---
    print(
        "Depolarising TP (output=['1i'], input=['0o']):",
        channel.is_tp(output_labels=["1i"], input_labels=["0o"]),
    )

    # --- Example 3: single projector - NOT TP ---
    K0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex128)
    partial = from_kraus_matrices([K0], systems={"0o": 2, "1i": 2})
    print("\nSingle projector TP:", partial.is_tp(["0o"], ["1i"]))

    # --- Example 4: Testing the Link Function ---
    print("\n--- Testing Link Function ---")
    
    # Define an Identity channel on systems 'a' and 'b'
    # 'a' is input, 'b' is output
    chan_A = from_kraus_matrices([I], systems={"a": 2, "b": 2})
    
    # Define another Identity channel on systems 'c' and 'd'
    # 'b' is the shared "wire"
    chan_B = from_kraus_matrices([I], systems={"c": 2, "d": 2})
    
    # Link them: the shared system 'b' should be contracted
    chan_combined = chan_A.link(chan_B)
    
    print(f"Combined systems: {list(chan_combined.systems.keys())}")
    
    # Check if the linked channel is still Trace Preserving
    is_tp = chan_combined.is_tp(output_labels=["c","d"], input_labels=["a","b"])
    print(f"Is linked Identity channel TP? {is_tp}")

    # --- Example 5: Linking Bit-Flips (X * X = I) ---
    chan_X1 = from_kraus_matrices([X], systems={"in": 2, "mid": 2})
    chan_X2 = from_kraus_matrices([X], systems={"mid": 2, "out": 2})
    
    chan_double_X = chan_X1.link(chan_X2)
    print(chan_double_X)
    # Get the resulting matrix. Since it's Identity, vec(I) = [1, 0, 0, 1]
    res_matrix = chan_double_X.kraus_matrices(output_labels=["out"], input_labels=["in"])[0]
    is_identity = torch.allclose(res_matrix, I)
    print(f"Linking two X gates gives Identity? {is_identity}")