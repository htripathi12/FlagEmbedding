import pytest
import torch
from torch.nn.functional import normalize

from FlagEmbedding.inference.embedder.encoder_only.m3 import M3Embedder


def test_opt_in_uses_tiled_maxsim_on_cpu(monkeypatch):
    monkeypatch.setenv("FLAGEMBEDDING_M3_TRITON", "1")
    queries = torch.ones(2, 17, 1)
    passages = -torch.ones(2, 513, 1)
    passages[1, -1] = 0

    maxima = M3Embedder._colbert_token_maxima(queries, passages)

    torch.testing.assert_close(maxima, torch.tensor([[-1.0], [0.0]]).expand(2, 17))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [65, 1024])
def test_cuda_kernel_matches_tiled_maxsim(dtype, width, monkeypatch):
    if not torch.cuda.is_available() or torch.version.cuda is None:
        pytest.skip("Requires NVIDIA CUDA")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Requires NVIDIA Ampere or newer")
    pytest.importorskip("triton")
    from FlagEmbedding.inference.embedder.encoder_only import _m3_triton

    rng = torch.Generator(device="cuda").manual_seed(42)
    queries = normalize(torch.randn(2, 17, width, device="cuda", generator=rng), dim=-1).to(dtype)
    passages = normalize(torch.randn(2, 545, width, device="cuda", generator=rng), dim=-1).to(dtype)
    # Exact similarities make it possible to check both kinds of padding explicitly.
    queries[:, 0] = 0
    queries[:, 0, 0] = 1
    passages[:, :, 0] = -1
    passages[1, -1] = 0
    monkeypatch.delenv("FLAGEMBEDDING_M3_TRITON", raising=False)
    expected = M3Embedder._colbert_token_maxima(queries, passages)

    calls = []
    kernel = _m3_triton.colbert_token_maxima

    def record_call(*args):
        calls.append(True)
        return kernel(*args)

    monkeypatch.setattr(_m3_triton, "colbert_token_maxima", record_call)
    monkeypatch.setenv("FLAGEMBEDDING_M3_TRITON", "1")
    actual = M3Embedder._colbert_token_maxima(queries, passages)

    assert calls == [True]
    torch.testing.assert_close(actual, expected)
    assert actual[:, 0].tolist() == [-1.0, 0.0]
