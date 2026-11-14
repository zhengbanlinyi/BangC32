"""Numerical and state-isolation regressions for the repaired operator entry points."""

import math
import re
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from load_cpu import load_operators

ROOT = Path(__file__).resolve().parents[1]
OPERATORS = (
    "ALiBi_attention_bias", "Attention_kv_cache", "Attention_score_with_bias", "Batch_norm_1D",
    "Cholesky_decomposition", "Conditional_LayerNorm", "Cross_Attention", "Depthwise_conv_2D",
    "Dilated_conv_2D", "DropPath", "Embedding_bag_mean", "Embedding_lookup", "GRU_forward",
    "Gather_rows", "Grid_sample", "Grouped_conv_2D", "Index_put", "LSTM_forward", "Linear",
    "Masked_cumsum", "Masked_fill", "Masked_select", "Masked_softmax", "Max_Pool_2D_with_indices",
    "One_hot_encoding", "Prefix_sum_2D", "QR_decomposition", "Relative_position_encoding",
    "SVD_decomposition", "Scaled_masked_softmax", "Scatter_add", "Sinusoidal_positional_encoding",
    "Softmax", "Sort", "Sparse_attention_mask", "Sparse_embedding", "TopK", "Triplet_loss",
    "Unfold", "Weight_standardization", "Where_conditional", "conv_depthwise_separable_2D",
    "cross_entropy", "cumsum", "cumsum_exclusive", "cumsum_reverse", "hstu_mask_var_len_fwd",
    "transform_vals",
    "All_over_dim", "CosineSimilarityLoss", "FrobeniusNorm_", "L1Norm_", "Matrix_inverse",
    "Power", "Product_reduction_over_a_dimension", "binned_gather", "binned_scatter", "gather",
    "MSE_Loss", "HuberLoss", "Cos", "Softplus", "Sqrt", "Adaptive_Max_Pool_2D",
    "Adaptive_Average_Pool_3D", "Add_RMSNorm", "conv_depthwise_2D_square_input_square_kernel",
)
DTYPES = (torch.float32, torch.float16)


@pytest.fixture(scope="session")
def ops():
    torch.set_num_threads(1)
    before = (F.softmax, torch.nn.Module._call_impl, torch.nn.Linear.forward, F.cross_entropy)
    module = load_operators(OPERATORS)
    yield module
    assert before == (F.softmax, torch.nn.Module._call_impl, torch.nn.Linear.forward, F.cross_entropy)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(721)


def close(actual, expected):
    tolerance = 5e-3 if actual.dtype == torch.float16 else 3e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance, equal_nan=True)


def test_reference_modifications_removed():
    forbidden = re.compile(r"PyRun_|PyGIL|ReferencePatch|findSchemaOrThrow|<Python\.h>")
    for path in ROOT.glob("*.mlu"):
        assert not forbidden.search(path.read_text()), path.name


def test_numbered_aliases_match():
    for path in ROOT.glob("*.mlu"):
        match = re.match(r"^\d{3}_(.*)$", path.name)
        if match:
            assert path.read_bytes() == (ROOT / match.group(1)).read_bytes(), path.name


@pytest.mark.parametrize("dtype", DTYPES)
def test_elementwise_and_broadcasting(ops, dtype):
    x = torch.randn(2, 4, 3, dtype=dtype).transpose(1, 2)
    y = torch.randn(1, 3, 1, dtype=dtype)
    mask = torch.rand(2, 1, 4) > 0.4
    close(ops.Where_conditional(mask, x, y), torch.where(mask, x, y))
    close(ops.Masked_fill(x, mask, -2.25), x.masked_fill(mask, -2.25))
    close(ops.transform_vals(x, x), x + x)
    close(ops.Masked_select(x, 0.1), torch.masked_select(x, x > 0.1))
    close(ops.Masked_select(x, 100.0), x.flatten()[:0])
    empty = torch.empty(2, 0, dtype=dtype)
    close(ops.transform_vals(empty, empty), empty)


def test_one_hot(ops):
    indices = torch.tensor([[1, 4, 0], [2, 1, 3]])
    torch.testing.assert_close(ops.One_hot_encoding(indices, 5), F.one_hot(indices, 5))
    with pytest.raises(RuntimeError, match="out of bounds"):
        ops.One_hot_encoding(torch.tensor([-1]), 5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_indexing(ops, dtype):
    x = torch.randn(4, 7, dtype=dtype)
    index = torch.tensor([[0, 6, 1], [2, 2, 0], [6, 3, 1], [4, 0, 5]])
    close(ops.Gather_rows(x, index), x.gather(1, index))
    source = torch.randn(5, 7, dtype=dtype)
    scatter = torch.tensor([0, 2, 0, 2, 1])
    expected = torch.zeros(3, 7).index_add_(0, scatter, source.float()).to(dtype)
    close(ops.Scatter_add(source, scatter, 3), expected)
    target = torch.randn(7, 4, dtype=dtype).t()
    expected = target.clone()
    values = torch.randn(3, 7, dtype=dtype)
    indices = torch.tensor([0, -1, 0])
    for i, row in enumerate(indices):
        expected[row] = values[i]
    result = ops.Index_put(target, values, indices)
    assert result.data_ptr() == target.data_ptr()
    close(result, expected)
    with pytest.raises(RuntimeError, match="out of bounds"):
        ops.Gather_rows(x, torch.full((4, 1), 7))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", (0, 1, -1))
def test_scans(ops, dtype, dim):
    x = torch.randn(2, 5, 3, dtype=dtype).transpose(0, 1)
    expected = x.float().cumsum(dim)
    close(ops.cumsum(x, dim), expected)
    close(ops.cumsum_exclusive(x, dim), expected - x.float())
    reverse = x.float().flip([dim]).cumsum(dim).flip([dim])
    close(ops.cumsum_reverse(x, dim), reverse)
    mask = torch.rand(x.shape) > 0.4
    close(ops.Masked_cumsum(x, mask, dim), (x.float() * mask).cumsum(dim))
    close(ops.Prefix_sum_2D(x), x.float().cumsum(-1).cumsum(-2))


@pytest.mark.parametrize("dtype", DTYPES)
def test_embeddings_and_linear(ops, dtype):
    weight = torch.randn(9, 5, dtype=dtype)
    index = torch.tensor([[0, 8, 2], [2, 0, 1]])
    close(ops.Embedding_lookup(index, weight), F.embedding(index, weight))
    close(ops.Sparse_embedding(index, weight), F.embedding(index, weight))
    close(ops.Embedding_lookup(torch.empty(0, dtype=torch.long), weight), weight[:0])
    indices = torch.tensor([0, 3, 3, 1, 7])
    offsets = torch.tensor([0, 3, 3, 5])
    expected = F.embedding_bag(indices, weight.float(), offsets, mode="mean").to(dtype)
    close(ops.Embedding_bag_mean(indices, offsets, weight), expected)
    x = torch.randn(2, 3, 5, dtype=dtype)
    matrix = torch.randn(4, 5, dtype=dtype)
    bias = torch.randn(4, dtype=dtype)
    close(ops.Linear(x, matrix, bias), F.linear(x.float(), matrix.float(), bias.float()).to(dtype))
    close(ops.Linear(x, matrix, torch.empty(0)), F.linear(x.float(), matrix.float()).to(dtype))
    table = torch.randn(13, 5, dtype=dtype)
    positions = torch.arange(4)
    expected = table[positions[:, None] - positions[None, :] + 6]
    close(ops.Relative_position_encoding(table, 4, 7), expected)


def test_missing_parameters_are_explicit(ops):
    x = torch.randn(2, 3, 4)
    calls = (
        lambda: ops.Linear(x, 4, 5, True),
        lambda: ops.Embedding_lookup(torch.tensor([1]), 5, 4),
        lambda: ops.Embedding_bag_mean(torch.tensor([1]), torch.tensor([0]), 5, 4),
        lambda: ops.Sparse_embedding(torch.tensor([1]), 5, 4),
        lambda: ops.Relative_position_encoding(3, 5, 4),
        lambda: ops.LSTM_forward(x, 4, 3, 1),
        lambda: ops.GRU_forward(x, 4, 3, 1),
        lambda: ops.Cross_Attention(x, x, 4, 2),
        lambda: ops.Conditional_LayerNorm(x, torch.randn(2, 2), 4, 2),
        lambda: ops.Weight_standardization(torch.randn(1, 2, 5, 5), 2, 3, 3),
    )
    for call in calls:
        with pytest.raises(RuntimeError, match="required"):
            call()


@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax(ops, dtype):
    x = torch.randn(2, 3, 7, 4, dtype=dtype).transpose(-1, -2)
    mask = torch.rand(2, 1, 4, 7) > 0.3
    mask[0, :, 0] = False
    close(ops.Masked_softmax(x, mask), x.masked_fill(~mask, -torch.inf).float().softmax(-1).to(dtype))
    scaled = (x * 0.75).masked_fill(~mask, -torch.inf)
    close(ops.Scaled_masked_softmax(x, mask, 0.75), scaled.float().softmax(-1).to(dtype))
    close(ops.Softmax(x), x.float().softmax(-1).to(dtype))
    close(ops.Softmax(torch.empty(0, 3, dtype=dtype)), torch.empty(0, 3, dtype=dtype))


def attention_reference(q, k, v, bias=None, window=None):
    scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(q.shape[-1])
    if bias is not None:
        scores += bias.float()
    if window is not None:
        positions = torch.arange(q.shape[-2])
        mask = (positions[:, None] - positions[None, :]).abs() > window
        scores.masked_fill_(mask, -torch.inf)
    return (scores.softmax(-1) @ v.float()).to(q.dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_attention(ops, dtype):
    q = torch.randn(2, 3, 4, 5, dtype=dtype).transpose(-1, -2)
    k = torch.randn(2, 3, 7, 4, dtype=dtype)
    v = torch.randn(2, 3, 7, 6, dtype=dtype)
    bias = torch.randn(1, 1, 5, 7, dtype=dtype)
    close(ops.Attention_score_with_bias(q, k, v, bias), attention_reference(q, k, v, bias))
    close(ops.Attention_kv_cache(q, k[:, :, :5], v[:, :, :5], k[:, :, 5:], v[:, :, 5:]),
          attention_reference(q, k, v))
    for window in (0, 1, 10):
        close(ops.Sparse_attention_mask(q, k[:, :, :5], v[:, :, :5], window),
              attention_reference(q, k[:, :, :5], v[:, :, :5], window=window))


@pytest.mark.parametrize("dtype", DTYPES)
def test_cross_attention_projection_weights(ops, dtype):
    model = torch.nn.MultiheadAttention(8, 2, batch_first=True).to(dtype).eval()
    q = torch.randn(2, 3, 8, dtype=dtype)
    kv = torch.randn(2, 5, 8, dtype=dtype)
    expected = model(q, kv, kv, need_weights=False)[0]
    actual = ops.Cross_Attention(q, kv, model.in_proj_weight, model.in_proj_bias,
                                 model.out_proj.weight, model.out_proj.bias, 2)
    close(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_encoding_and_drop_path(ops, dtype):
    x = torch.randn(2, 4, 5, dtype=dtype)
    channel = torch.arange(5)
    angle = torch.arange(4)[:, None] * 10000.0 ** (-2.0 * (channel // 2) / 5)
    pe = torch.where(channel % 2 == 0, angle.sin(), angle.cos()).to(dtype)
    close(ops.Sinusoidal_positional_encoding(x, 5, 10), x + pe)
    close(ops.Sinusoidal_positional_encoding(x, 5), x + pe)
    for probability in (0.0, 0.5, 1.0):
        close(ops.DropPath(x, probability), x)
        if probability in (0.0, 1.0):
            close(ops.DropPath(x, probability, True), x if probability == 0 else torch.zeros_like(x))
        else:
            torch.manual_seed(57)
            expected = x * (torch.rand(2) < 0.5).reshape(2, 1, 1).to(dtype) * 2
            torch.manual_seed(57)
            close(ops.DropPath(x, probability, True), expected)


@pytest.mark.parametrize("heads", (1, 3, 8))
def test_alibi(ops, heads):
    distance = (torch.arange(9)[:, None] - torch.arange(9)[None, :]).abs()
    slopes = 2.0 ** (-8.0 * (torch.arange(heads) + 1) / heads)
    close(ops.ALiBi_attention_bias(9, heads), -slopes[:, None, None] * distance)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("training", (False, True))
def test_batch_norm(ops, dtype, training):
    x = torch.randn(4, 3, 5, dtype=dtype)
    weight, bias = torch.randn(3), torch.randn(3)
    mean, variance = torch.randn(3), torch.rand(3) + 0.2
    expected_mean, expected_variance = mean.clone(), variance.clone()
    expected = F.batch_norm(x.float(), expected_mean, expected_variance, weight, bias,
                           training, 0.1, 1e-5).to(dtype)
    actual = ops.Batch_norm_1D(x, weight, bias, mean, variance, 3, training, 1e-5)
    close(actual, expected)
    close(mean, expected_mean)
    close(variance, expected_variance)


@pytest.mark.parametrize("dtype", DTYPES)
def test_conditional_norm(ops, dtype):
    x = torch.randn(2, 3, 5, dtype=dtype)
    cond = torch.randn(2, 4, dtype=dtype)
    gw, bw = torch.randn(5, 4, dtype=dtype), torch.randn(5, 4, dtype=dtype)
    gb, bb = torch.randn(5, dtype=dtype), torch.randn(5, dtype=dtype)
    gamma = F.linear(cond.float(), gw.float(), gb.float()).to(dtype)
    beta = F.linear(cond.float(), bw.float(), bb.float()).to(dtype)
    expected = (F.layer_norm(x.float(), [5]) * gamma.float()[:, None] + beta.float()[:, None]).to(dtype)
    close(ops.Conditional_LayerNorm(x, cond, gw, gb, bw, bb, 5, 4), expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_convolutions_and_unfold(ops, dtype):
    x = torch.randn(2, 4, 9, 8, dtype=dtype)
    dw = torch.randn(4, 1, 3, 3, dtype=dtype)
    close(ops.Depthwise_conv_2D(x, dw, 4, 3, 2, 1),
          F.conv2d(x.float(), dw.float(), stride=2, padding=1, groups=4).to(dtype))
    weights = torch.randn(6, 4, 3, 3, dtype=dtype)
    close(ops.Dilated_conv_2D(x, weights, 4, 6, 3, 2, 2),
          F.conv2d(x.float(), weights.float(), dilation=2, padding=2).to(dtype))
    grouped = torch.randn(6, 2, 3, 3, dtype=dtype)
    close(ops.Grouped_conv_2D(x, grouped, 4, 6, 3, 2),
          F.conv2d(x.float(), grouped.float(), padding=1, groups=2).to(dtype))
    pointwise = torch.randn(6, 4, 1, 1, dtype=dtype)
    first = F.conv2d(x.float(), dw.float(), stride=2, padding=2, dilation=2, groups=4).to(dtype)
    expected = F.conv2d(first.float(), pointwise.float()).to(dtype)
    close(ops.conv_depthwise_separable_2D(x, dw, pointwise, 4, 6, 3, 2, 2, 2), expected)
    close(ops.Unfold(x, 3, 2, 1), F.unfold(x.float(), 3, padding=1, stride=2).to(dtype))
    close(ops.Max_Pool_2D_with_indices(x, 3, 2), F.max_pool2d(x.float(), 3, 2).to(dtype))
    normalized = (weights.float() - weights.float().mean((1, 2, 3), keepdim=True))
    normalized = (normalized * torch.rsqrt(weights.float().var((1, 2, 3), unbiased=False, keepdim=True) + 1e-5)).to(dtype)
    close(ops.Weight_standardization(x, weights, 4, 6, 3),
          F.conv2d(x.float(), normalized.float(), padding=1).to(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("align", (False, True))
def test_grid_sample(ops, dtype, align):
    x = torch.randn(2, 3, 5, 7, dtype=dtype)
    grid = (torch.rand(2, 4, 6, 2) * 3 - 1.5).to(dtype)
    expected = F.grid_sample(x.float(), grid.float(), align_corners=align).to(dtype)
    close(ops.Grid_sample(x, grid, align), expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_matrix_factorizations(ops, dtype):
    x = torch.randn(2, 5, 3).to(dtype)
    q, r = ops.QR_decomposition(x)
    close((q.float() @ r.float()).to(dtype), x)
    u, s, vh = ops.SVD_decomposition(x)
    close((u.float() @ torch.diag_embed(s.float()) @ vh.float()).to(dtype), x)
    a = (x.float().transpose(-1, -2) @ x.float() + torch.eye(3)).to(dtype)
    lower = ops.Cholesky_decomposition(a)
    close((lower.float() @ lower.float().transpose(-1, -2)).to(dtype), a)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layers", (1, 2))
@pytest.mark.parametrize("kind", ("LSTM", "GRU"))
def test_recurrent(ops, dtype, layers, kind):
    model = getattr(torch.nn, kind)(3, 4, layers, batch_first=True).to(dtype).eval()
    x = torch.randn(2, 5, 3, dtype=dtype)
    # CPU LSTM/GRU in float32 provides an independent recurrence reference.
    reference = getattr(torch.nn, kind)(3, 4, layers, batch_first=True).float().eval()
    reference.load_state_dict({k: v.float() for k, v in model.state_dict().items()})
    expected = reference(x.float())[0].to(dtype)
    actual = getattr(ops, kind + "_forward")(x, list(model._flat_weights), 3, 4, layers)
    close(actual, expected)
    if kind == "GRU" and layers == 2:
        close(ops.GRU_forward(x, *model._flat_weights, 3, 4, layers), expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("smoothing", (0.0, 0.2, 1.0))
@pytest.mark.parametrize("reduce", (False, True))
def test_cross_entropy(ops, dtype, smoothing, reduce):
    x = torch.randn(2, 3, 7, dtype=dtype)
    targets = torch.randint(0, 7, (2, 3))
    targets[0, 0] = -100
    expected = F.cross_entropy(x.float().flatten(0, 1), targets.flatten(), label_smoothing=smoothing,
                              reduction="mean" if reduce else "none")
    expected = expected.reshape(1 if reduce else (2, 3)).to(dtype)
    actual = ops.cross_entropy(x, targets, torch.tensor(smoothing), torch.tensor(reduce),
                               torch.empty(0), torch.tensor(-100))
    close(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_losses_and_all_ignored_targets(ops, dtype):
    a, p, n = [torch.randn(5, 7, dtype=dtype) for _ in range(3)]
    close(ops.Triplet_loss(a, p, n, 0.7), F.triplet_margin_loss(a.float(), p.float(), n.float(), margin=0.7).to(dtype))
    x = torch.randn(2, 3, 7, dtype=dtype)
    y = torch.full((2, 3), -100)
    actual = ops.cross_entropy(x, y, torch.tensor(0.1), torch.tensor(True), torch.tensor(0), torch.tensor(-100))
    assert actual.isnan().all()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("descending", (False, True))
def test_sort(ops, dtype, descending):
    x = torch.randn(3, 13, dtype=dtype)
    x[0, :5] = torch.tensor([1.0, -torch.inf, torch.inf, torch.nan, 1.0], dtype=dtype)
    if dtype == torch.float32:
        x[1, :2] = torch.tensor([1e20, 1.0])
    close(ops.Sort(x, 1, descending), x.sort(1, descending=descending).values)
    close(ops.Sort(x.t(), 0, descending), x.t().sort(0, descending=descending).values)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k", (0, 1, 20, 25))
def test_topk(ops, dtype, k):
    x = torch.randn(3, 25, dtype=dtype)
    x[0, :5] = torch.tensor([-torch.inf, -torch.inf, torch.inf, torch.nan, 0.0], dtype=dtype)
    values, indices = ops.TopK(x, k, 1)
    close(values, x.topk(k, 1).values)
    close(values, x.gather(1, indices))


@pytest.mark.parametrize("dtype", DTYPES)
def test_hstu_dynamic_batches_and_grouped_heads(ops, dtype):
    lengths = [4, 0, 5]
    offsets = torch.tensor([0, 4, 4, 9])
    context, history, realtime = torch.tensor([1, 0, 1]), torch.tensor([1, 0, 2]), torch.tensor([1, 0, 0])
    causal, expand, labels = torch.tensor([False, True, True]), torch.tensor([True, False, True]), torch.tensor([1, 0, 1])
    q = torch.randn(9, 3, 6, dtype=dtype).transpose(1, 2)
    k = torch.randn(9, 2, 3, dtype=dtype)
    v = torch.randn(9, 2, 5, dtype=dtype)
    expected = torch.empty(9, 6, 5, dtype=dtype)
    for batch, length in enumerate(lengths):
        start = offsets[batch].item()
        positions = torch.arange(length)
        limits = positions.clone()
        if not causal[batch]:
            limits[positions < context[batch] + history[batch]] = context[batch] + history[batch] - 1
        limits[positions < context[batch]] = context[batch] + (history[batch] if expand[batch] else 0) - 1
        keep = limits[:, None] >= positions[None, :]
        label_begin = context[batch] + history[batch] + realtime[batch]
        for column in range(label_begin.item(), length):
            end = label_begin + ((column - label_begin) // labels[batch] + 1) * labels[batch]
            if end <= length:
                keep[:, column] &= limits < end
        for head in range(6):
            key_head = head // 3
            scores = q[start:start + length, head].float() @ k[start:start + length, key_head].float().t()
            probabilities = (F.silu(scores * 0.3) / 8).to(dtype)
            expected[start:start + length, head] = ((probabilities * keep).float()
                                                   @ v[start:start + length, key_head].float()).to(dtype)
    actual = ops.hstu_mask_var_len_fwd(q, k, v, 0.3, offsets, 8, context, history, realtime,
                                      causal, expand, labels, 0)
    close(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_replaced_constant_answers(ops, dtype):
    x = torch.randn(4, 7, dtype=dtype)
    close(ops.L1Norm_(x), (x.float() / x.float().abs().sum(1, keepdim=True)).to(dtype))
    close(ops.FrobeniusNorm_(x), (x.float() / x.float().norm()).to(dtype))
    for exponent in (0.0, 1.0, 2.0, 3.0, -1.0, 0.5):
        close(ops.Power(x, exponent), x.float().pow(exponent).to(dtype))
    y = torch.randn(4, 7, dtype=dtype)
    close(ops.CosineSimilarityLoss(x, y), (1 - F.cosine_similarity(x.float(), y.float(), dim=1)).mean().to(dtype))
    close(ops.CosineSimilarityLoss(x, x), torch.zeros((), dtype=dtype))
    matrix = torch.randn(2, 4, 4).to(dtype)
    matrix = (matrix.float() @ matrix.float().transpose(-1, -2) + 0.5 * torch.eye(4)).to(dtype)
    close(ops.Matrix_inverse(matrix), torch.linalg.inv(matrix.float()).to(dtype))


@pytest.mark.parametrize("dim", (0, 1, -1))
def test_logical_and_product_reductions(ops, dim):
    x = torch.randn(2, 3, 5).transpose(0, 1)
    x[0, 0, 0] = 0
    close(ops.Product_reduction_over_a_dimension(x, dim), x.prod(dim))
    torch.testing.assert_close(ops.All_over_dim(x, dim), x.all(dim))
    torch.testing.assert_close(ops.All_over_dim(torch.ones_like(x), dim), torch.ones_like(x).all(dim))
    empty = torch.empty(2, 0, 5)
    close(ops.Product_reduction_over_a_dimension(empty, 1), empty.prod(1))
    torch.testing.assert_close(ops.All_over_dim(empty, 1), empty.all(1))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("capacity", (0, 3, 9))
def test_moe_routing_uses_inputs_and_metadata(ops, dtype, capacity):
    tokens, top_k, width = 5, 3, 7
    x = torch.randn(tokens, width, dtype=dtype)
    indices = torch.randperm(tokens * top_k).to(torch.int32)
    weights = torch.randn(tokens * top_k, dtype=dtype)
    counts = torch.tensor([4, 4, 0, 7])
    bins = counts.cumsum(0).to(torch.int32)
    bin_ids = torch.repeat_interleave(torch.arange(4), counts).to(torch.int32)
    routes = indices.long()
    expected = (x[routes // top_k].float() * weights[routes].float()[:, None]).to(dtype)
    close(ops.gather(x, indices, bin_ids, weights, bins, top_k), expected)
    binned = torch.zeros(4, capacity, width, dtype=dtype)
    sources, scales, token_indices = [], [], []
    start = 0
    for expert, count in enumerate(counts):
        accepted = min(capacity, count.item())
        binned[expert, :accepted] = expected[start:start + accepted]
        selected_routes = routes[start:start + accepted]
        sources.append(binned[expert, :accepted].float())
        scales.append(weights[selected_routes].float())
        token_indices.append(selected_routes // top_k)
        start += count.item()
    close(ops.binned_gather(x, indices, weights, bins, capacity, top_k), binned)
    restored = torch.zeros(tokens, width)
    restored.index_add_(0, torch.cat(token_indices), torch.cat(sources) * torch.cat(scales)[:, None])
    close(ops.binned_scatter(binned, indices, weights, bins, top_k), restored)


@pytest.mark.parametrize("dtype", DTYPES)
def test_mse_and_huber_compute_from_targets(ops, dtype):
    prediction = torch.randn(3, 7, dtype=dtype)
    target = torch.randn(1, 7, dtype=dtype)
    close(ops.MSE_Loss(prediction, target), (prediction.float() - target.float()).square().mean().to(dtype))
    close(ops.MSE_Loss(prediction, prediction), torch.zeros((), dtype=dtype))
    close(ops.HuberLoss(prediction, target), F.huber_loss(prediction.float(), target.float().expand_as(prediction)).to(dtype))
    close(ops.HuberLoss(prediction, target, 0.4),
          F.huber_loss(prediction.float(), target.float().expand_as(prediction), delta=0.4).to(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_unary_numerical_boundaries(ops, dtype):
    values = torch.tensor([-10000.0, -100.0, -3.0, -0.0, 0.0, 1.0, 20.0, 10000.0, torch.inf, torch.nan], dtype=dtype)
    close(ops.Cos(values), values.float().cos().to(dtype))
    close(ops.Softplus(values), F.softplus(values.float()).to(dtype))
    close(ops.Sqrt(values), values.float().abs().sqrt().to(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_adaptive_pooling_and_rmsnorm(ops, dtype):
    image = torch.randn(2, 3, 11, 13, dtype=dtype)
    close(ops.Adaptive_Max_Pool_2D(image, [4, 5]), F.adaptive_max_pool2d(image.float(), [4, 5]).to(dtype))
    volume = torch.randn(2, 3, 7, 11, 13, dtype=dtype)
    close(ops.Adaptive_Average_Pool_3D(volume, [3, 4, 5]), F.adaptive_avg_pool3d(volume.float(), [3, 4, 5]).to(dtype))
    x = torch.randn(2, 3, 7, dtype=dtype) * 1000
    residual = torch.randn(2, 1, 7, dtype=dtype)
    weight = torch.randn(7, dtype=dtype)
    summed = (x + residual).float()
    expected = (summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-5) * weight.float()).to(dtype)
    close(ops.Add_RMSNorm(x, residual, weight, 7, 1e-5), expected)
    kernel = torch.randn(3, 1, 3, 3, dtype=dtype)
    close(ops.conv_depthwise_2D_square_input_square_kernel(image, kernel, 3, 3, 2, 1),
          F.conv2d(image.float(), kernel.float(), stride=2, padding=1, groups=3))
