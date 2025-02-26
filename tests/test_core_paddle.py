import random
import paddle
from paddle import Tensor
from typing import Tuple

import deep_gemm
from deep_gemm import bench_kineto, calc_diff, cell_div, get_col_major_tma_aligned_tensor


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



def test_gemm() -> None:
    print('Testing GEMM:')
    for m in (64,128):
        for k, n in [(7168, 2112)]:
            x_fp8, y_fp8, out, ref_out = construct(m, k, n)
            deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
            diff = calc_diff(out, ref_out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'
            print("diff:",diff)
            def test_func():
                # Construct new tensors every time to avoid L2 cache acceleration
                x_fp8, y_fp8, out, ref_out = construct(m, k, n)
                deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)

    print()

def test_simple():
    matrix1 = paddle.randint(low=1, high=11, shape=[2, 4], dtype='int32')
    matrix2 = paddle.randint(low=1, high=11, shape=[4, 2], dtype='int32')
    result = paddle.matmul(matrix1, matrix2)
    print("matrix1:",matrix1)
    print("matrix2:",matrix2)
    print("result:",result)
    
    x_fp8, y_fp8, out, ref_out = construct(2, 4, 2)
    deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
    diff = calc_diff(out, ref_out)
    print(diff)

if __name__ == '__main__':
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    paddle.seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')
 
    test_simple()
    #test_gemm()
    # test_m_grouped_gemm_contiguous()
    # test_m_grouped_gemm_masked()

