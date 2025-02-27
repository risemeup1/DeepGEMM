import random
import paddle
from paddle import Tensor
from typing import Tuple

import deep_gemm
from deep_gemm import calc_diff, cell_div, get_col_major_tma_aligned_tensor


def per_token_cast_to_fp8(x: Tensor) -> Tuple[Tensor, Tensor]:
    assert x.dim() == 2 and x.shape[1] % 128 == 0
    m, n = x.shape
    x_view=paddle.view(x,(m,-1,128))
    x_abs = paddle.abs(x_view).astype(paddle.float32)
    x_amax = paddle.amax(x_abs, axis=2)
    x_amax = paddle.view(x_amax, (m, -1))
    x_amax= paddle.clip(x_amax, min=1e-4)

    scaled_x = x_view * (448.0 / x_amax.unsqueeze(2))
    # 假设要转换到更小的精度，可以使用 float16 或者保持 float32
    scaled_x_converted = paddle.view(scaled_x.astype(paddle.float8_e4m3fn), (m, n))

    # 处理 x_amax 的操作
    x_amax_scaled = paddle.view((x_amax / 448.0), (m, -1))
    

    # 输出结果
    result = (scaled_x_converted, x_amax_scaled)
    return result
    # return (x_view * (448.0 / x_amax.unsqueeze(2))).to(paddle.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: Tensor) -> Tuple[Tensor, Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = paddle.zeros((cell_div(m, 128) * 128, cell_div(n, 128) * 128), dtype=x.dtype)
    x_padded[:m, :n] = x
    # x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_view = paddle.view(x_padded, (-1, 128, x_padded.shape[1] // 128, 128))
    
    # x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_abs = paddle.abs(x_view).astype(paddle.float32)
    x_amax = paddle.amax(x_abs, axis=(1,3),keepdim=True)
    x_amax= paddle.clip(x_amax, min=1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).astype(paddle.float8_e4m3fn)

    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (paddle.view(x_amax / 448.0,(x_view.shape[0], x_view.shape[2])))

def construct(m: int, k: int, n: int) -> \
        Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tensor, Tensor]:
    x = paddle.randn((m, k), dtype=paddle.bfloat16)
    y = paddle.randn((n, k), dtype=paddle.bfloat16)
    out = paddle.empty((m, n), dtype=paddle.bfloat16)
    ref_out = x @ y.t()

    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out

# def construct_grouped(num_groups: int, m: int, k: int, n: int, is_masked: bool) -> \
#         Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tensor, Tensor]:
#     x = paddle.randn((num_groups, m, k), dtype=paddle.bfloat16)
#     y = paddle.randn((num_groups, n, k), dtype=paddle.bfloat16)
#     out = paddle.empty((num_groups, m, n), dtype=paddle.bfloat16)
#     ref_out = paddle.einsum('gmk,gnk->gmn', x, y)

#     assert m % 4 == 0, f'TMA alignment error: {m}'
#     x_fp8 = (paddle.empty_like(x, dtype=paddle.float8_e4m3fn), paddle.empty((num_groups, m, k // 128), dtype=paddle.float32))
#     y_fp8 = (paddle.empty_like(y, dtype=paddle.float8_e4m3fn), paddle.empty((num_groups, (n + 127) // 128, k // 128), dtype=paddle.float32))
#     for i in range(num_groups):
#         #set_value_tensor not support float8_e4m3fn
#         # x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])
#         # y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])
#         x_fp8_0_i, x_fp8_1_i = per_token_cast_to_fp8(x[i])
#         paddle.assign(x_fp8_0_i, x_fp8[0][i])
#         paddle.assign(x_fp8_1_i, x_fp8[1][i])

#         y_fp8_0_i, y_fp8_1_i = per_block_cast_to_fp8(y[i])
#         paddle.assign(y_fp8_0_i, y_fp8[0][i])
#         paddle.assign(y_fp8_1_i, y_fp8[1][i])

#     # For non-masked input, we must merge the group and M dims
#     if not is_masked:
#         # x_fp8 = (x_fp8[0].view(-1, k), per_token_cast_to_fp8(x.view(-1, k))[1])
#         # out, ref_out = out.view(-1, n), ref_out.view(-1, n)

#     # Transpose earlier so that the testing will not trigger transposing kernels
#     x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
#     return x_fp8, y_fp8, out, ref_out


def test_gemm() -> None:
    print('Testing GEMM:')
    for m in (64,128):
        for k, n in [(7168, 2112)]:
            x_fp8, y_fp8, out, ref_out = construct(m, k, n)
            deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
            diff = calc_diff(out, ref_out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'
            print("diff:",diff)
            # def test_func():
            #     # Construct new tensors every time to avoid L2 cache acceleration
            #     x_fp8, y_fp8, out, ref_out = construct(m, k, n)
            #     deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)

    print()

# def test_m_grouped_gemm_contiguous() -> None:
#     print('Testing grouped contiguous GEMM:')

#     for num_groups, m, k, n in ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168), (8, 4096, 7168, 4096), (8, 4096, 2048, 7168)):
#         # TODO: make a stronger test
#         x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=False)
#         m_indices = paddle.arange(0, num_groups, device='cuda', dtype=paddle.int32)
#         m_indices = m_indices.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)
#         deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)
#         diff = calc_diff(out, ref_out)
#         assert diff < 0.001, f'm={m * num_groups}, {k=}, {n=}, {diff:.5f}'

#         # noinspection PyShadowingNames
#         def test_func():
#             # Construct new tensors every time to avoid L2 cache acceleration
#             x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=False)
#             m_indices = paddle.arange(0, num_groups, device='cuda', dtype=paddle.int32)
#             m_indices = m_indices.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)
#             deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)

#         t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
#         print(f' > Performance ({num_groups=}, m_per_group={m:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
#               f'throughput: {2 * num_groups * m * n * k / t / 1e12:4.0f} TFLOPS, '
#               f'{(num_groups * (m * k + k * n + m * n * 2)) / 1e9 / t:4.0f} GB/s')
#     print()

if __name__ == '__main__':
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    paddle.seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')
    # for i in range(10):
    test_gemm()
    # test_m_grouped_gemm_contiguous()
    # test_m_grouped_gemm_masked()

