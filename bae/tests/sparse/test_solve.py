import torch
import pytest


def _make_spd_system(n: int, dtype: torch.dtype, device: torch.device):
    spd = torch.rand(n + 1, n, dtype=dtype, device=device)
    A = spd.mT @ spd
    A = A + torch.eye(n, dtype=dtype, device=device)
    b = torch.rand(n, dtype=dtype, device=device)
    return A.to_sparse_csr(), b


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cudss_preserves_rhs_shape(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CuDirectSparseSolver")

    try:
        from bae.sparse.solve import CuDirectSparseSolver
    except Exception as e:
        pytest.skip(f"CuDirectSparseSolver unavailable: {e}")

    device = torch.device("cuda")
    A, b = _make_spd_system(n=32, dtype=dtype, device=device)
    b_col = b.view(-1, 1)

    solver = CuDirectSparseSolver()
    x_vec = solver(A, b)
    x_col = solver(A, b_col)

    assert x_vec.shape == b.shape
    assert x_col.shape == b_col.shape
    assert torch.allclose(x_vec.view(-1, 1), x_col, atol=1e-5, rtol=1e-4)

    atol = 1e-5 if dtype == torch.float32 else 1e-10
    r_col = A @ x_col - b_col
    assert torch.linalg.norm(r_col).item() < atol + 1e-5 * torch.linalg.norm(b_col).item()


if __name__ == "__main__":
    from bae.sparse.solve import CuDirectSparseSolver

    A, b = _make_spd_system(n=3, dtype=torch.float64, device=torch.device("cuda"))
    solver = CuDirectSparseSolver()
    x = solver(A, b)
    print("||Ax - b|| =", (A @ x - b).norm().item())
