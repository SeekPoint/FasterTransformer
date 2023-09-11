/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "cuda_kernels.h"
#include <assert.h>
#include <cstdio>
#include <cstdlib>

namespace fastertransformer{

#define FINAL_MASK 0xffffffff
#define CUDART_PI_F 3.141592654f

template <typename T>
__inline__ __device__
T gelu(T x)
{
  float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

template <>
__inline__ __device__
half2 gelu(half2 val)
{
  half2 val_pow3 = __hmul2(val, __hmul2(val, val));
  float2 tmp_pow = __half22float2(val_pow3);
  float2 tmp =  __half22float2(val);

  tmp.x = 0.5f * (1.0f + tanhf((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
  tmp.y = 0.5f * (1.0f + tanhf((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
  return __hmul2(val, __float22half2_rn(tmp));
}

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32]; 
  int lane = threadIdx.x & 0x1f; 
  int wid = threadIdx.x >> 5;  

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)0.0f;
  val = warpReduceSum(val);
  return val;
}
/*
5.4.2 add_bias_act_kernel
顾名思义，add_bias_act_kernel 核函数包含了 add bias 和 activation 两个操作。
 源码中 block_size = n / 4 实际就是 latent_dim，为什么不直接取上一次运算后的矩阵宽度 n = 4 * latent_dim 呢？
 这里是希望一行元素（4 * latent_dim）能在一个 block 内处理，如果 block_size 直接取 n = 4 * latent_dim，可能超过 1024，
 因此还取 latent_dim，线程内循环 4 次处理即可。同样，源码中针对 grid_size 也取了 m / 4，在线程中通过循环每次跨 m / 4 步长处理 4 行数据。


 核函数中先对列进行循环，ite = 4，从全局内存读出当前列的 bias，然后针对行进行循环，步长为 m / 4，
 循环体内部对当前行当前列的元素进行 add bias 和 gelu 操作，这里gelu 操作是一个简单的 element-wise 操作，比较简单不再介绍。

笔者点评：这里笔者私以为没有必要 grid_size 也取 m / 4，cuda 本身对线程块的数量没有限制，完全可以直接取 m，
 每次每个线程只处理一行数据，一方面可以增加并行程度，另一方面代码可阅读性也更好。笔者给出代码如下，亲测可用。
dim3 grid(m);
dim3 block(n / 4);
assert(block.x <= 1024);
add_bias_act_v2<T><<<grid, block, 0, stream>>>(out, bias, m, n);

template <typename T>
__global__
void add_bias_act_v2(T* out, const T* bias, int m, int n) {
  T val, reg_bias;

  int row_id = blockIdx.x;
  int ite = n / blockDim.x;
  int tid = threadIdx.x;

  for(int i = 0; i < ite; ++i) {
    reg_bias = __ldg(&bias[i * blockDim.x + tid]);
    val = out[tid + i * blockDim.x + row_id * n]+ reg_bias;
    out[tid + i * blockDim.x + row_id * n] = gelu<T>(val);
  }
}

 * */
template <typename T>
__global__ 
void add_bias_act(T* out, const T* bias, int m, int n)
{
  T val, reg_bias;

  int row_id = blockIdx.x;
  int ite = n / blockDim.x;
  int tid = threadIdx.x;

  for(int i = 0; i < ite; ++i)
  {
    reg_bias = __ldg(&bias[i * blockDim.x + tid]);
    row_id = blockIdx.x;

    while(row_id < m){
      val = out[tid + i * blockDim.x + row_id * n]+ reg_bias;
      out[tid + i * blockDim.x + row_id * n] = gelu<T>(val);
      row_id += gridDim.x;
    }
  }
}

template <>
__global__ 
void add_bias_act(__half* out, const __half* bias, int m, int n)
{
  half2 val, reg_bias;
  int row_id = blockIdx.x;
  int ite = n / blockDim.x / 2;
  int tid = threadIdx.x;

  half2* out_ptr = (half2*) out;
  const half2* bias_ptr = (half2*) bias;
  for(int i = 0; i < ite; ++i)
  {
    reg_bias = __ldg(&bias_ptr[i * blockDim.x + tid]);
    row_id = blockIdx.x;

    while(row_id < m){
      val = out_ptr[tid + i * blockDim.x + row_id * n / 2];
      val = __hadd2(val, reg_bias);
      out_ptr[tid + i * blockDim.x + row_id * n / 2] = gelu<half2>(val);
      row_id += gridDim.x;
    }
  }
}

/*
5.3 add_bias_input_layernorm_kernel
从核函数名字可以看出，这个核函数实现了 3 个操作：add bias、add input、layernomalization。
 其中 add bias 是完成上一步线性变换未完成的加偏置工作，add input 是 transformer 模型中的残差结构，layernomalization 则是层归一化操作。
 综合起来这个核函数的作用是：对线性变换后的 attention out 加偏置，然后加上原始输入 tensor 组成一个残差结构，最后进行一次层归一化变换。
 源码中针对 fp16 和 fp32 分别提供了一个核函数实现，计算逻辑都一样，这里只以 fp32 为例介绍。
006.png
 * @brief                       grid_size = m, block_size = n
 *
 * @tparam T
 * @param out                   [batch_size, sql_len, latent_dim]
 * @param input                 [batch_size, sql_len, latent_dim]
 * @param bias                  [latent_dim,]
 * @param gamma
 * @param beta
 * @param m                     batch_size * seq_len
 * @param n                     latent_dim
 * @return __global__
 *
 *
 如示意图所示，核函数中每个 block 处理一行数据，共 latent_dim = head_num * size_per_head 个元素，
 核函数中首先计算了 add bias、add input 两个操作，并将计算结果存储在寄存器变量 local_out 中。
接下来就是标准的 layerNormalization 操作，我们先来看一下 layerNormalization 的操作步骤，
 可以参照一下 tensorflow 框架 API 文档。
007.png

具体地，第一步计算均值和方差，核函数中使用块内规约计算出均值 s_mean 存储在共享内存中，所有块内线程都可以访问。
 然后根据 s_mean 和线程内的 local_out 以及 epsilon 系数再进行一次块内规约计算出方差 s_variance 存储在共享内存中。
第二步进行归一化和线性变换，对应 tensorflow API 的二、三步，直接计算即可，没有其他技巧，公式如下：
008.png

5.5 add_bias_input_layernorm_kernel
这个核函数的计算逻辑在 5.3 中已经介绍过了，包含加偏置项、残差结构、层归一化三个操作，不再赘述。

 * */
template <typename T>
__global__ 
void add_bias_input_layernorm(T* out, const T* input, const T* bias, const T* gamma, const T* beta, int m, int n)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;

    // add，一个block处理一行
  for(int i = tid; i < n; i += blockDim.x)
    local_out += (float)(out[blockIdx.x * n + i] + input[blockIdx.x * n + i] + __ldg(&bias[i]));

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>((local_out - s_mean) * (local_out - s_mean));
  if(threadIdx.x == 0)
    s_variance = variance / n + 1e-6f;
  __syncthreads();

  for(int i = tid; i < n; i += blockDim.x)
    out[blockIdx.x * n + i] = 
	    (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[i])) + (float)(__ldg(&beta[i])));
}

template <>
__global__ 
void add_bias_input_layernorm(__half* out, const __half* input, const __half* bias, 
  const __half* gamma, const __half* beta, int m, int n)
{

  int tid = threadIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;
  float2 local_out_fp2;

  half2* out_ptr = (half2*)out;
  const half2* input_ptr = (const half2*)input;
  const half2* bias_ptr = (const half2*)bias;
  const half2* gamma_ptr = (const half2*)gamma;
  const half2* beta_ptr = (const half2*)beta;
 
  float local_out = 0.0f;
  int id = blockIdx.x * n / 2 + tid; 
  local_out_fp2 = __half22float2(__hadd2(__hadd2(out_ptr[id], input_ptr[id]), __ldg(&bias_ptr[tid])));
  local_out += local_out_fp2.x;
  local_out += local_out_fp2.y;

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
  variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
  variance = blockReduceSum<float>(variance);
  if(threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  float2 gamma_val = __half22float2(__ldg(&gamma_ptr[tid]));
  float2 beta_val = __half22float2(__ldg(&beta_ptr[tid]));
  local_out_fp2.x = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
  local_out_fp2.y = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
  out_ptr[id] = __float22half2_rn(local_out_fp2);
}

template <typename T>
void add_bias_act_kernelLauncher(T* out, const T* bias, int m, int n, cudaStream_t stream)
{
  dim3 grid(m / 4);
  dim3 block(n / 4);
  assert(block.x <= 1024);
  add_bias_act<T><<<grid, block, 0, stream>>>(out, bias, m, n);
}

template<typename T>
void add_bias_input_layernorm_kernelLauncher(T* out, const T* input, const T* bias, 
  const T* gamma, const T* beta, int m, int n, cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(n);
  assert(block.x <= 1024);
  add_bias_input_layernorm<T><<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
}


template <>
void add_bias_input_layernorm_kernelLauncher(__half* out, const __half* input, const __half* bias, 
  const __half* gamma, const __half* beta, int m, int n, cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(n / 2);
  assert(block.x <= 1024);
  add_bias_input_layernorm<__half><<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
}

template void add_bias_act_kernelLauncher<float>(
  float* out, const float* bias, int m, int n, cudaStream_t stream);

template void add_bias_input_layernorm_kernelLauncher<float>(
  float* out, const float* input, const float* bias, const float* gamma, const float* beta, 
  int m, int n, cudaStream_t stream);

template void add_bias_act_kernelLauncher<__half>(
  __half* out, const __half* bias, int m, int n, cudaStream_t stream);

template void add_bias_input_layernorm_kernelLauncher<__half>(
  __half* out, const __half* input, const __half* bias, const __half* gamma, const __half* beta, 
  int m, int n, cudaStream_t stream);

}//namespace 
