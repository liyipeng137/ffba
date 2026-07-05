import pytest
import torch

from bae.utils.linear_operator import NormalMatVec
from bae.utils.pysolvers import PCG


def test_normal_matvec_dense_matches_explicit():
    torch.manual_seed(0)
    m, n = 8, 5
    J = torch.randn(m, n, dtype=torch.float64)
    x = torch.randn(n, dtype=torch.float64)

    op = NormalMatVec(J)
    y_op = op.matvec(x)
    y_ex = J.mT @ (J @ x)

    torch.testing.assert_close(y_op, y_ex, rtol=1e-10, atol=1e-10)


def test_normal_matvec_dense_damping_matches_explicit():
    torch.manual_seed(1)
    m, n = 7, 4
    J = torch.randn(m, n, dtype=torch.float64)
    x = torch.randn(n, dtype=torch.float64)
    damping = 0.3

    diag = J.square().sum(dim=0)
    op = NormalMatVec(J, damping=damping)
    y_op = op @ x
    y_ex = J.mT @ (J @ x) + damping * diag * x

    torch.testing.assert_close(y_op, y_ex, rtol=1e-10, atol=1e-10)


def test_normal_matvec_sparse_csr_matches_explicit_and_cached_diag():
    torch.manual_seed(2)
    m, n = 10, 6
    dense = torch.randn(m, n, dtype=torch.float64)
    mask = (torch.rand_like(dense) < 0.5)
    dense = dense * mask
    J = dense.to_sparse_csr()
    x = torch.randn(n, dtype=torch.float64)

    diag = dense.square().sum(dim=0)
    op = NormalMatVec(J, diag=diag)
    y_op = op.matvec(x)
    y_ex = dense.mT @ (dense @ x)

    torch.testing.assert_close(y_op, y_ex, rtol=1e-10, atol=1e-10)


def test_normal_matvec_sparse_bsr_matches_explicit():
    torch.manual_seed(3)
    m, n = 6, 4
    dense = torch.randn(m, n, dtype=torch.float64).contiguous()
    x = torch.randn(n, dtype=torch.float64)

    try:
        J_bsr = dense.to_sparse_bsr(blocksize=(2, 2))
    except Exception:
        pytest.skip("BSR conversion not supported on this device/build.")

    op = NormalMatVec(J_bsr)
    y_op = op @ x
    y_ex = dense.mT @ (dense @ x)

    torch.testing.assert_close(y_op, y_ex, rtol=1e-10, atol=1e-10)


def test_pcg_smoke_with_normal_operator():
    torch.manual_seed(4)
    m, n = 12, 5
    J = torch.randn(m, n, dtype=torch.float64)
    damping = 1e-3
    op = NormalMatVec(J, damping=damping)

    b = torch.randn(n, dtype=torch.float64)
    solver = PCG(tol=1e-10, maxiter=200)
    x_pcg = solver(op, b)

    diag = J.square().sum(dim=0)
    A_dense = J.mT @ J + damping * torch.diag(diag)
    x_ex = torch.linalg.solve(A_dense, b)

    torch.testing.assert_close(x_pcg, x_ex, rtol=1e-6, atol=1e-6)

