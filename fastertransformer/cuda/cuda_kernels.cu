/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
#include "fastertransformer/common.h"

#include "cuda_kernels.h"
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cfloat>
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

template <typename T>
  __inline__ __device__
T warpReduceMax(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
  __inline__ __device__
T blockReduceMax(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f; // in-warp idx
  int wid = threadIdx.x >> 5;  // warp idx

  val = warpReduceMax(val); // get maxx in each warp

  if(lane == 0) // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();


  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)-1e20f;
  val = warpReduceMax<T>(val);

  return val;
}


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
void add_bias_act(half* out, const half* bias, int m, int n)
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
void add_bias_input_layernorm(half* out, const half* input, const half* bias, 
  const half* gamma, const half* beta, int m, int n)
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

template<typename T>
__global__
void broadcast_kernel(T* log_probs, T* cum_log_probs, const int batch_size, const int beam_width, const int vocab_size, const int N)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = tid / vocab_size;

  if(tid < N)
    log_probs[tid] += cum_log_probs[bid];
}

template <typename T>
void add_bias_act_kernelLauncher(T* out, const T* bias, int m, int n, cudaStream_t stream)
{
//  dim3 grid(m / 64);
  dim3 grid(m / 4);
  dim3 block(n / 4);
  assert(block.x <= 1024);
//  dim3 block(n);
  add_bias_act<T><<<grid, block, 0, stream>>>(out, bias, m, n);
}

template<typename T>
void add_bias_input_layernorm_kernelLauncher(T* out, const T* input, const T* bias, 
  const T* gamma, const T* beta, int m, int n, cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(n);
  assert(n <= 1024);
  add_bias_input_layernorm<T><<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
}


template <>
void add_bias_input_layernorm_kernelLauncher(half* out, const half* input, const half* bias, 
  const half* gamma, const half* beta, int m, int n, cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(n / 2);
  assert(n / 2 <= 1024);
  add_bias_input_layernorm<half><<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
}

/*4.10.1 broadcast_kernelLauncher
beam search 的计算依据就是整个序列的条件概率，
 也就是说要把每个 step 的概率连乘起来，
 所以对于当前 step 来说各 beam 下每个 word 的概率首先应该乘以前面所有 step 组成序列的累计概率，
 由于我们这里的概率值在 update 函数中计算的是 log 值，所以把这里把连乘换成累加。

cum_log_probs 的形状是 [batch_size, beam_width]，表示每个 beam 下的累计概率，
 这里给 log_probs 加上累计概率之后，
 就表示一个 batch 中第 batch_id 个样本第 beam_id 个 beam 的第 word_id 个 word 作为当前 step 下输出 token 的概率。
 核函数计算逻辑非常简单，可以直接看代码。*/
void broadcast_kernelLauncher(float* log_probs, float* cum_log_probs, const int batch_size, const int beam_width, 
  const int vocab_size, cudaStream_t stream)
{
  
  int N = batch_size * beam_width * vocab_size;
  dim3 block(1024);
  dim3 grid((N - 1) / block.x + 1);

  broadcast_kernel<float><<<grid, block, 0, stream>>>(log_probs, cum_log_probs, batch_size, beam_width, vocab_size, N);
}

/*4.10.2 topK
对于一个样本而言拿到 log_probs 后我们就得到了 beam_width 个分支下共 beam_width * vocab_size 条路径的概率，
 按照 beam search 的计算思想，
 我们需要找到这 beam_width * vocab_size 个路径中概率最大的 beam_width 个路径，
 这是一个求 topK 的问题。

 由于 vocab_size 数值比较大为了保障效率，源码通过两轮 topK 操作求出 topK。*/
template <typename T>
__global__
void topK_kernel(const T* log_probs, int* ids, const int batch_size, const int N, const int K)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val, max_val;
  __shared__ float s_max_val;
  for(int ite = 0; ite < batch_size; ++ite)
  {
    bool choosed = false;
    val = (tid < N ) ? (float)log_probs[ite * N + tid] : -1e20f;
    
    for(int kids = 0; kids < K; ++kids)
    {
      max_val = blockReduceMax<float>(val);
      
      if(threadIdx.x == 0)
        s_max_val = max_val;
      __syncthreads();

      if(s_max_val == val && !choosed && tid < N) 
      {
        ids[ite * gridDim.x * K + blockIdx.x * K + kids] = tid + ite * N;
        val = -1e20f;
        choosed = true;
      }
    }
  }
}

/**
 * @brief for each batch, get the final TopK values out from grid.x * K values
 *
 * @tparam T
 * @param log_probs             [batch_size, beam_width, vocab_size]
 * @param ids                   [batch_size, N]
 * @param batch_size
 * @param N                     gridDim.x(1st) * beam_width
 * @param K                     beam_width
 * @param id_offset             beam_width * vocab_size
 * @return __global__
 */
template <typename T>
__global__
void topK_kernel_2nd(const T* log_probs, int* ids, const int batch_size, const int N, const int K, const int id_offset)
{
  int tid = threadIdx.x;
  float val, max_val;
  __shared__ float s_max_val;
  __shared__ int beam_index;
  __shared__ int ids_before_sort[16];

  for(int ite = 0; ite < batch_size; ++ite)
  {
    bool choosed = false;
    const int id = (tid < N) ? ids[ite * N + tid] : -1;
    val = (tid < N) ? (float)log_probs[id] : -1e20f;

    __syncthreads();

    if(tid == 0) beam_index = 0;
    if(tid < 16) ids_before_sort[tid] = -1;
    
    __syncthreads();
    while(beam_index < K){
      int begin_beam_index = beam_index;
      max_val = blockReduceMax<float>(val);
      if(threadIdx.x == 0){
        s_max_val = max_val;
      }
      __syncthreads();
      if(s_max_val == val && !choosed && id != -1)
      {
        int id_offset_ = atomicAdd(&beam_index, 1);
        ids_before_sort[id_offset_] = id;
        val = -1e20f;
        choosed = true;
      }
      __syncthreads();
     
      // simply sort the ids
      if(threadIdx.x == 0 && beam_index - begin_beam_index > 1){
        for(int i = begin_beam_index; i < beam_index; i++){
          for(int j = i; j < beam_index; j++){
            if(ids_before_sort[j] < ids_before_sort[i]){
              int tmpid = ids_before_sort[j];
              ids_before_sort[j] = ids_before_sort[i];
              ids_before_sort[i] = tmpid;
            }
          }
        }
      }
    }
    __syncthreads();
    if(tid < K) ids[ite * K + tid] = ids_before_sort[tid];
    __syncthreads();
  }
}

/*
第一轮 topK 操作将 beam_width * vocab_size 个路径划分到每个 block 中计算，取 block_size 为 1024，
对于一个样本来说，每个线程只处理一种可能路径，每个 block 内部求出一个 topK，所以最终计算完成后共有 grid_size_1st 个 topK。
核函数中首先是一轮针对 batch_size 的循环，表示一个线程内部会处理多个样本，然后取出当前线程对应的路径的概率 val，
然后开始求 topK。源码中求 topK 的思路非常之朴实无华，循环 K 次，每次块内规约取最大值，然后给当前最大值赋一个极小值防止干扰。
取到最大值后，就把最大值所在的位置存入 ids 中，注意这里的位置包含 3 个信息：batch_id、beam_id、word_id，
分别对应 log_probs 的三个维度，ids 的形状为 [batch_size, grid_size_1st, beam_width]。

第二轮 topK 操作将 grid_size_1st * beam_width 个路径进一步缩小到 beam_width 个路径，
本轮计算全部在一个 block 内完成，取 block_size 为 1024，对于一个样本来说，每个线程只处理一种可能路径。
说实话这个核函数实现过程过于复杂，笔者看了几遍都不能完全理解，笔者的想法是完全可以复用第一轮 topK 的核函数进行计算，
 所以以下的解读仅代表笔者本人揣测，如有错误请读者评论或私信指出。

核函数内部定义了一个共享内存数组 ids_before_sort 用来临时存储 topK 的位置，至于数组大小为什么是 16，
 笔者猜可能是目前 beam_width 最大支持取 16，但是笔者没有在任何官方声明里面看到这个信息。
 然后是针对 batch_size 的循环，表示一个线程内部会处理多个样本，然后取出当前线程对应的路径的概率 val 和 word 的位置 id。
 再来一轮循环，每次块内规约取最大值，然后给最大值赋一个极小值，把最大值的位置信息 id 存入 ids_before_sort，
 这里还使用了原子操作 atomicAdd，猜测是为了防止有两个 word 的概率一样大的情况下可能由于内存读写竞争导致计算错误。
 每轮选出最大值后还根据 word 的位置给排了个序，咱也不知道啥意图。。。
 最后把 topK 的位置信息存入 ids 中，注意这里的位置信息仍然是 batch_id、beam_id、word_id，
 不过这时候 ids 里面只有 [batch_size, beam_width] 范围内的元素是有效的。*/
void topK(const float* log_probs, int* ids, const int batch_size, const int beam_width, const int vocab_size,
  cudaStream_t stream)
{
  int N = beam_width * vocab_size;
  dim3 block(1024);
  dim3 grid((N - 1) / block.x + 1);
  /* First round topK, for each batch, get grid.x * K values */
  topK_kernel<float><<<grid, block, 0, stream>>>(log_probs, ids, batch_size, N, beam_width);
  /*Second round, for each batch, get the final TopK values out from grid.x * K values. */
  topK_kernel_2nd<float><<<1, block, 0, stream>>>(log_probs, ids, batch_size, beam_width * grid.x, beam_width, N);
}

/*4.10.3 update
update 函数主要是对累计概率、序列长度、解码 token 等信息根据本次 beam search 的结果进行更新。
 更新过程全部在 1 个 block 内完成，
 block_size 取 batch_size * beam_width，
 每个线程内部处理一个样本的一条路径。

 核函数内首先对 sequence_length 进行更新，如果 finished 标识为 false，sequence_length 长度增加 1，
 否则认为当前 step 当前 beam 已经终止了。
 然后计算了几个索引的值，beam_id 表示当前线程处理的 topK 路径是属于哪一个 beam。
 word_id 表示当前线程处理的路径对应的 word 在词表中的位置。
 首先将 cum_log_probs 更新到最新，也就是取之前计算的 topK 条路径对应的概率。
 更新 sequence_length 为 topK 路径对应的 beam 的长度。
 根据 topK 对应的 word_id 是否为 end_id 更新 finished 标识，
 这其实就标识下一轮 step 的输入 token 是否是 end_token。
 更新 parent_ids 为 beam_id，标识当前这个 topK 路径是从哪个 beam 经过的，
 说白了这个变量存储着下一轮 beam 中每个路径是从上一轮哪个 beam 经过的。
 word_ids 和 output_ids 存储着本轮输出的 topK 个 word 在词表中的位置，
 这里的 word_id 在下一轮 step 经过 embedding lookup 之后就变成了了 from_tensor。
 */
template <typename T>
__global__
void update_kernel(T* log_probs, T* cum_log_probs, 
                  int* ids, bool* finished, 
                  int* parent_ids, int* sequence_length, 
                  int* word_ids, int* output_ids, 
                  const int batch_size, const int beam_width, 
                  const int vocab_size, const int end_id, 
                  int* finished_count)
{
  int tid = threadIdx.x;
  sequence_length[tid] = finished[tid] ? sequence_length[tid] : sequence_length[tid] + 1;

  int beam_id = ids[tid];
  beam_id /= vocab_size;
  int word_id = ids[tid];
  word_id %= vocab_size;

  cum_log_probs[tid] = log_probs[ids[tid]];
  sequence_length[tid] = sequence_length[beam_id];
  finished[tid] = word_id == end_id ? 1 : 0;
  parent_ids[tid] = beam_id;
  word_ids[tid] = word_id;
  output_ids[tid] = word_id;

  // TODO use reduce sum to compute how many sentence are finished
  // int fi = finished[tid]
  // int total_finish = reduceSum(fi);
}

//    4.5 embedding_lookup
//            顾名思义，embedding_lookup 函数的功能就是把输入 token 从 word_id 映射为词向量，
//            其实现逻辑就是根据 word_id 去 decoding_params.embedding_table 中查表，把向量存进 from_tensor[0] 中。

/**
 * @brief 读 word_ids[blockIdx.x] 获取 word_id，embedding_table 的 word_id 行就是词向量
 *
 * @tparam T
 * @param embedding_table           [vocab_size, hidden_size]
 * @param word_ids                  [batch_size, beam_width]
 * @param hidden_units
 * @param from_tensor               [batch_size, beam_width, hidden_size]
 * @return __global__
 */
template <typename T>
__global__ void embedding_lookup_kernel(const T* embedding_table, const int* word_ids,
    const int hidden_units, T* from_tensor)
{
  int write_pos = threadIdx.x + blockIdx.x * hidden_units;
  from_tensor[write_pos] = embedding_table[word_ids[blockIdx.x] * hidden_units + threadIdx.x];
}

void update(float* log_probs, float* cum_log_probs, 
            int* ids, bool* finished, 
            int* parent_ids, int* sequence_length,
            int* word_ids, int* output_ids, 
            const int batch_size, const int beam_width, 
            const int vocab_size, cudaStream_t stream, 
            const int end_id, int* finished_count)
{ 
  
  dim3 grid(1);
  dim3 block(batch_size * beam_width);

  assert(block.x <= 1024);

  update_kernel<float><<<grid, block, 0, stream>>>(log_probs, cum_log_probs, ids, 
                                                  finished, parent_ids, sequence_length,
                                                  word_ids, output_ids, batch_size, 
                                                  beam_width, vocab_size, end_id, 
                                                  finished_count);
}

/**
 * @brief embedding_lookup
 *
 * @tparam T
 * @param embedding_table               [vocab_size, hidden_size]
 * @param word_ids                      [batch_size, beam_width]
 * @param from_tensor   from_tensor[0]: [batch_size, beam_width, hidden_size]
 * @param batch_size
 * @param beam_width
 * @param hidden_units
 * @param stream
 *
 * 通过核函数逻辑可以发现，就是把 word_ids_buf_ lookup 成了 from_tensor_[0]，
 * 形状为 [batch_size, beam_width, hidden_size]。
 */
template <typename T>
void embedding_lookup(const T* embedding_table, const int* word_ids, T* from_tensor,
  const int batch_size, const int beam_width, const int hidden_units, cudaStream_t stream)
{
   dim3 grid(batch_size * beam_width);
   dim3 block(hidden_units);
   assert(hidden_units <= 1024);
   embedding_lookup_kernel<<<grid, block, 0, stream>>>(embedding_table, word_ids, hidden_units, from_tensor);
}

/*4.9 计算 logits
在深度学习中，logits 的计算一般是通过一个 Dense 层加一个 softmax 激活（二分类中使用 sigmoid，属于 softmax 的特化版本）来实现，
 源码中通过一个 cuBLAS API 调用和一个 update_logits 实现。
 cuBLAS API 实现矩阵乘法，
 add bias 和 softmax 放在 update_logits 中实现，
 下面重点介绍 update_logits 的逻辑。

 * @brief
 *
 * @tparam T
 * @param logits                          [batch_size, beam_width, vocab_size]
 * @param bias                            [vocab_size,]
 * @param end_id
 * @param finished                        [batch_size, beam_width]
 * @param n                               vocab_size
 * @return __global__

 通常我们的词典大小 vocab_size 远大于 1024，
 所以 block_size 绝大多数都取 1024，一个 block 处理一行元素的计算，
 一个线程会处理多个元素，步长为 blockDim.x。

 第一个循环体内包含 3 个计算任务：
 首先判断当前 step 的 finish flag，若为 true 就把 end_id 的那个分量的 logit 直接设置为最大值，
 否则就正常 add bias，最后计算当前线程处理的 logit 的最大值 max_val。

 然后通过块内规约求出整个 block 内的 max_val 的最大值，
 这也是整行 vocab_size 个元素的最大值，
 存进共享内存 s_max_val 中。第二、三个循环体分别完成了求指数和以及除以指数和的任务，
 最终计算结果取对数就是 LogSoftmax 结果。
*/
template <typename T>
__global__ void update_logits_kernel(T* logits, const T* bias, const int end_id, const bool* finished, const int n)
{
  int bid = blockIdx.x;
  bool finish = finished[bid];
  int offset = bid * n;

  float max_val = -1 * FLT_MAX;
  __shared__ float s_max_val;
  __shared__ float s_sum_val;

// tid 对应的其实就是 word_id
  for(int tid = threadIdx.x; tid < n; tid += blockDim.x)
  {
    if(finish)
      logits[offset + tid] = (tid == end_id) ? FLT_MAX : -1 * FLT_MAX;
    else
      logits[offset + tid] += bias[tid];
    max_val = max(max_val, logits[offset + tid]);
  }

  max_val = blockReduceMax<float>((float)max_val);
  if(threadIdx.x == 0)
    s_max_val = max_val;
  __syncthreads();

  float sum_val = 0.0f;
  for(int tid = threadIdx.x; tid < n; tid += blockDim.x)
  {
    logits[offset + tid] = __expf((float)logits[offset + tid] - s_max_val);
    sum_val += (float)logits[offset + tid];
  }

  sum_val = blockReduceSum<float>(sum_val);
  if(threadIdx.x == 0)
    s_sum_val = sum_val;
  __syncthreads();

  for(int tid = threadIdx.x; tid < n; tid += blockDim.x)
  {
    logits[offset + tid] = logf((float)logits[offset + tid] / s_sum_val);
  }
}

void update_logits(float* logits, const float* bias, const int end_id, const bool* finished, 
  const int m, const int n, cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(min(n, 1024));
  /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big. */
  update_logits_kernel<float><<<grid, block, 0, stream>>>(logits, bias, end_id, finished, n);
}

/*
4.4 init 函数
这个函数主要实现以下几个功能：
    decoding_params.sequence_length 初始化为 0
    finished_buf_ 初始化为 false
    word_ids 初始化为 start_id
    cum_log_probs 将 beam_id 为 0 的位置初始化为 0，其他位置初始化为 -lnf

 初始化完成之后，就开始逐 step 解码了，下面的函数均在 loop for step 中执行。
 * */
template <typename T>
__global__ void init_kernel(bool* finished, int* sequence_length, int* word_ids, T* cum_log_probs, const int sentence_id, const int n, const int beam_width)
{
  int tid = threadIdx.x;
  finished[tid] = false;
  sequence_length[tid] = 0;
  word_ids[tid] = sentence_id;
  cum_log_probs[tid] = (T)(tid % beam_width == 0 ? 0.0f: -1e20f);
}

/*
前面 3.5.2.3 节中讲过，对于下一轮 step 的 query 来说 self-attention 的 key 是上一轮 step 的输入 token 对应的 tensor 变换后的结果。
 由于我们每一轮 beam search 取 topK 会打乱顺序，
 直观上咱们并不知道下一轮 topK 分别来源于上一轮哪个 beam，这时候我们就要用上 parent_ids 了，
 根据 parent_ids 获取上一轮 step 经过的 beam。
 前面讲过，在 Decoder layer 计算的过程中会把当前 beam 的 token 对应的 key 和 value 计算好存入 K_cache_ 和 V_cache_ 中，
 现在我们直接根据 beam_id 去取值就可以了，基于双缓存机制，
 从 key_cache[src_id] 中取值更新 key_cache[tgt_id]。这块稍微复杂的地方就是 hidden_id 的计算逻辑，
 不理解的读者可以结合笔者的这段话多读几遍。
  */
template <typename T>
__global__ void update_KV_cache_kernel(
  T* key_src_cache, T* key_tgt_cache,
  T* value_src_cache, T* value_tgt_cache,
  const int* beam_ids, const int batch_size, const int beam_width, const int hidden_dim, const int cache_size, const int step, const int decoder_layers)
{
  int layer_id = blockIdx.x / batch_size / beam_width / step;
  int batch_id = (blockIdx.x % (batch_size * beam_width * step)) / (beam_width * step);
  int beam_id = (blockIdx.x % (beam_width * step)) / step;
  int step_id = blockIdx.x % step;

  int hidden_id = step_id * batch_size * beam_width * hidden_dim + 
    beam_ids[batch_id * beam_width + beam_id] * hidden_dim;

  int tgt_hidden_id = step_id * batch_size * beam_width * hidden_dim + 
    batch_id * beam_width * hidden_dim + beam_id * hidden_dim;

  T* key_src_ptr = key_src_cache + layer_id * cache_size;
  T* key_tgt_ptr = key_tgt_cache + layer_id * cache_size;
  T* value_src_ptr = value_src_cache + layer_id * cache_size;
  T* value_tgt_ptr = value_tgt_cache + layer_id * cache_size;


  for(int tid = threadIdx.x; tid < hidden_dim; tid += blockDim.x)
  {
    key_tgt_ptr[tgt_hidden_id + tid] = key_src_ptr[hidden_id + tid];
    value_tgt_ptr[tgt_hidden_id + tid] = value_src_ptr[hidden_id + tid];
  }
  
}
template <typename T>
void update_KV_cache(T** key_cache, T** value_cache, const int* beam_ids, const int batch_size, const int beam_width, const int hidden_dim,
  const int step, const int cache_size, const int decoder_layers, cudaStream_t stream)
{
  dim3 grid(decoder_layers * batch_size * beam_width * step);
  dim3 block(min(1024, hidden_dim));

  int src_id = step & 0x1;
  int tgt_id = 1 - src_id;

  update_KV_cache_kernel<<<grid, block, 0, stream>>>(
    key_cache[src_id], key_cache[tgt_id],
    value_cache[src_id], value_cache[tgt_id],
    beam_ids, batch_size, beam_width, hidden_dim, cache_size, step, decoder_layers);
}

void init(bool* finished, int* sequence_length, int* word_ids, float* cum_log_probs, const int sentence_id, const int batch_size, 
  const int beam_width, cudaStream_t stream)
{
  dim3 grid(1);
  dim3 block(min(1024, batch_size * beam_width));

  assert(batch_size * beam_width <= 1024);
  
  init_kernel<float><<<grid, block, 0, stream>>>(finished, sequence_length, word_ids, cum_log_probs, sentence_id, batch_size * beam_width, beam_width);
}

/*
4.6 sine_position_encoder
我们知道 attention 本身是无序的，每个 token 在计算过程中地位都是等同的，为了
 表达这种 token 之间的顺序效果，transformer 加入了 Position Encoding layer 给输入 tensor 添加一种位置信息。
 原始论文中使用下面的公式来实现位置嵌入：
        209.png
可以看到 position encoding 和 token 位置和 hidden 维度奇偶有关，实际计算过程中我们不考虑 hidden 元素的奇偶，
 这个维度的顺序本身也没什么意义，所以我们直接前半 hidden_units 使用 sin 后半部分使用 cos，其实 tensorflow api 也是这么简化的，
 具体如下：

def positional_encoding(length, depth):
  depth = depth/2
  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)
  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)
  return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x
位置编码示意图如下：
        210.png
下面我们来看一下源码，首先给每个元素乘以 根号n  达到缩放效果，然后来计算 ....的部分，
 源码可能是为了保证精度和数值溢出考虑，使用先取对数再计算指数的策略将其分解为两部分。

 log_timescale_increment 计算的是...，这里源码中对 half_n 做了略微修正。
 inv_timescales 计算的是 ....，
 随后再乘以 step（也就是pos），得到三角函数里面的内容。
 最后根据 tid 判断应该使用正弦还是余弦然后将计算结果加在 tensor 上即可。

 * */
template<typename T>
__global__
void sine_position_encoder_kernel(T* output, int step, int n){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  float half_n = (float)n / 2.;

  // input = input * hidden_dim**0.5
  output[bid * n + tid] = output[bid * n + tid] * (T)sqrtf(float(n));

  float log_timescale_increment = __logf(10000) / (half_n - 1.f);
  float inv_timescales = __expf( (tid % (int)half_n) * -1 * log_timescale_increment );
  float scaled_time = inv_timescales * step;
  
  T encoding_val = (tid < half_n) ? (T) __sinf(scaled_time) : (T) __cosf(scaled_time);
  output[bid * n + tid] = output[bid * n + tid]  + encoding_val;
}
/**
 * @brief position encoding
 *
 * @tparam T
 * @param output              [m, hidden_units]
 * @param step
 * @param m                   batch_size * beam_width
 * @param n                   hidden_units
 * @param stream
 */
template<typename T>
void sine_position_encoder(
  T* output,
  int step,
  int m, int n, cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(n);
  assert(n <= 1024);
  sine_position_encoder_kernel<T><<<grid, block, 0, stream>>>(output, step, n);
}

template void add_bias_act_kernelLauncher<float>(
  float* out, const float* bias, int m, int n, cudaStream_t stream);

template void add_bias_input_layernorm_kernelLauncher<float>(
  float* out, const float* input, const float* bias, const float* gamma, const float* beta, 
  int m, int n, cudaStream_t stream);

template void add_bias_act_kernelLauncher<half>(
  half* out, const half* bias, int m, int n, cudaStream_t stream);

template void add_bias_input_layernorm_kernelLauncher<half>(
  half* out, const half* input, const half* bias, const half* gamma, const half* beta, 
  int m, int n, cudaStream_t stream);

template void embedding_lookup(const float* embedding_table, const int* word_ids, float* from_tensor,
  const int batch_size, const int beam_width, const int hidden_units, cudaStream_t stream);

template void embedding_lookup(const half* embedding_table, const int* word_ids, half* from_tensor,
  const int batch_size, const int beam_width, const int hidden_units, cudaStream_t stream);

template void update_KV_cache(float** key_cache, float** value_cache, const int* beam_ids, const int batch_size, const int beam_width, const int hidden_dim,
  const int step, const int cache_size, const int decoder_layers, cudaStream_t stream);

template void update_KV_cache(half** key_cache, half** value_cache, const int* beam_ids, const int batch_size, const int beam_width, const int hidden_dim,
  const int step, const int cache_size, const int decoder_layers, cudaStream_t stream);

template void sine_position_encoder(
  float* output,
  int step,
  int m, int n,
  cudaStream_t stream);

template void sine_position_encoder(
  half* output,
  int step,
  int m, int n,
  cudaStream_t stream);

}//namespace 
