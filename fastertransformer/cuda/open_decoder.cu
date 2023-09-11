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
/**
 * Open sourced multi-head attention
 **/

#include "fastertransformer/open_decoder.h"

namespace fastertransformer{

/**
  masked multi-head attention
 */
#define FINAL_MASK 0xffffffff
template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}
/* Calculate the sum of all elements in a block */
template <typename T>
  __inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32]; 
  //__shared__ T shared[32]; 
  int lane = threadIdx.x & 0x1f; 
  int wid = threadIdx.x >> 5;  

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
                              
  return val;
}
/*3.9.2 add_bias_relu
顾名思义，add_bias_relu 核函数包含了 add bias 和 relu 两个操作。源码中 block_size = n1 / 4 实际就是 hudden_units_，
 为什么不直接取上一次运算后的矩阵宽度 n1 = 4 * hudden_units_ 呢？
 这里是希望一行元素（4 * hudden_units_）能在一个 block 内处理，
 如果 block_size 直接取 n1，可能超过 1024，因此还取 hudden_units_，线程内循环 4 次处理即可。
 核函数逻辑非常简单，注意按步长 blockDim.x 取数即可。*/
template <typename T>
__global__ 
void add_bias_relu(T* out, const T* bias, int m, int n)
{
  T val, reg_bias;

  int row_id = blockIdx.x;
  int ite = n / blockDim.x;
  int tid = threadIdx.x;

  for(int i = 0; i < ite; ++i)
  {
    reg_bias = __ldg(&bias[i * blockDim.x + tid]);
    row_id = blockIdx.x;

    while(row_id < m)
    {
      val = out[tid + i * blockDim.x + row_id * n] + reg_bias;
      out[tid + i * blockDim.x + row_id * n] = (T)(val > 0.0f ? val : 0.0f);
      row_id += gridDim.x;
     }
  }
}

template <>
  __global__ 
void add_bias_relu(half* out, const half* bias, int m, int n)
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

    while(row_id < m)
    {
      val = out_ptr[tid + i * blockDim.x + row_id * n / 2];
      val = __hadd2(val, reg_bias);
      val.x = val.x > (half)0.0f ? val.x : (half)0.0f;
      val.y = val.y > (half)0.0f ? val.y : (half)0.0f;
      out_ptr[tid + i * blockDim.x + row_id * n / 2] = val;
      row_id += gridDim.x;
    }
  }
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
//  __shared__ T shared[32]; 
  int lane = threadIdx.x & 0x1f; // in-warp idx
  int wid = threadIdx.x >> 5;  // warp idx

  val = warpReduceMax(val); // get maxx in each warp

  if(lane == 0) // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();


  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : 0;
  val = warpReduceMax(val);

  return val;
}

/*3.5.2 masked_attention_kernel
核函数内部主要完成两个操作：add bias 和 计算 attention。其中 add bias 是上一步 Dense 变换的补充。
 masked_attention_kernel 是目前为止 Decoder 里面逻辑最复杂的核函数，
 下面笔者将尽量讲解明白，笔者水平有限，如果理解有误，欢迎各位多提意见。

3.5.2.1 核函数执行配置参数的确定
源码中 grid_size 直接取 batch_size_ * head_num_，也就是说一个 block 处理一个 head 的元素。
 而 block_size 的确认逻辑就比较复杂了，感觉这块代码有些乱，笔者用自己的理解总结一下：

block_size 满足基本规范，如 2 的 n 次幂，最大不超过 1024，最小取 64
block_size 不小于 step
block_size 不小于 size_per_head
读到这里不禁产生了一个疑问，为什么 block_size 同时跟 step 和 size_per_head 有关？
 原因很直观，一个 block 内做了两件事，分别完成了 size_per_head 个元素的计算以及 step 轮计算。
 作者把两个功能放在了一个 kernel 里，通常情况下为了代码的可维护性我们不会这么干，
 这里这么实现我想是为了尽可能地融合 kernel 减小 kernel 调动的开销，
 代价就是代码可维护性降低，不容易理解。

除此之外，代码中提前定义了一个 shared_size，作为核函数内部动态共享内存的大小参数，在启动核函数时传入，
 可以看到，总的内存大小是 size_per_head_ + step 个元素。

3.5.2.2 add query bias
核函数内部首先定义了一个共享内存变量数组 s_buf，数组大小对应前面传入的 size_per_head_ + step，
 然后根据偏移量分别定义了两个变量 sq 和 logits，分别用来存储 add bias 和 attention 的中间结果。
 这种写法提供了一个思路，就是如果我们核函数内部不止一个地方需要使用动态大小的共享内存时，
 由于核函数执行参数里面只让传一个表示共享内存大小的参数，
 可以传一个总的内存大小，在核函数内部再通过偏移量自行取用，注意规划好内存大小，不要越界访问。
 */

template <typename T>
__global__ 
void masked_attention_kernel(T* query_buf, const T* self_Q_bias, 
  T* key_cache, const T* self_K_bias, T* value_cache, const T* self_V_bias,
  T* context_buf, int batch_size, int head_num, int size_per_head, const int step, const T scalar)
{
  extern __shared__ __align__(sizeof(T)) unsigned s_buf[];
  T* sq = reinterpret_cast<T *>(s_buf);
  T* logits = reinterpret_cast<T *>(&sq[size_per_head]);

  int tid = threadIdx.x;
  int bid = blockIdx.x / head_num;
  int head_id = blockIdx.x % head_num;

/*
 * 然后定义了一个 qkv_id 变量，用来描述当前线程处理的元素在 query_buf 中的位置。定义了一个 qkv_bias_id 用来描述对应元素在 bias 中的位置，
 * 可以看到 bias 的形状为 [head_num, size_per_head]。随后计算 query 的 add bias 并将其存入 sq 中。
 * 这里笔者有个疑问，就是有没有必要将 add bias 的结果存入共享内存？
 * 似乎这里并没有需要块内通信的场景，笔者认为直接存在寄存器中可能更好。
 *
 * 至此，核函数内部完成了 query 的 add bias 操作，下一步该进行 softmax 的计算了。
 * */

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head + tid;
  int qkv_bias_id = head_id * size_per_head + tid;

  if(tid < size_per_head)
    sq[tid] = query_buf[qkv_id] + self_Q_bias[qkv_bias_id];
  __syncthreads();

  /*3.5.2.3 add key bias & softmax
我们知道 attention 中 softmax 计算的对象是 query 和 key 的乘积，query 我们已经拿到了，
   就是当前解码 step 的输入 tensor 变换后的结果，分别存在每个 block 的 sq 中。key 是什么？
   对于当前 step 的 query 来说这里的 key 应该是前面 step 的 token 对应的 tensor 变换后的结果，前面我们讲过，
   由于 Dense 变换的权重是固定的且 token 也是确定的，所以 key 也是固定的，
   那么我们每轮 step 的时候就可以计算好当前 step 的 key 存入 key_cache 中供后面的 step 计算时使用，
   同时在当前 step 也可以从 key_cache 中取前面 step 的 key 用于计算。

首先对 step 进行循环，通过偏移量 offset 获取前面 step 的 key，如果循环到了当前 step，还需要在 key 上 add bias 然后存进 key_cache。
   拿到 key 之后和 sq[tid] 相乘相当于两个向量的相应位置元素相乘，再乘以一个缩放因子 scalar 之后进行块内规约得到 qk，相当于两个向量的内积。
   这个 qk 就是当前 step 的输入 token 对应的 tensor 和第 ite+1 step 的 token 对应的 tensor 在某个 head 中的 attention scores，
   每一轮计算获取一个 qk，将其存入共享内存变量 logits 中，最终 logits 中将存储 step 个元素。
   */
  //offset for each step
  int offset = batch_size * head_num * size_per_head;
  for(int ite = 0; ite < step; ++ite)
  {
    T key = tid < size_per_head ? key_cache[ite * offset + qkv_id] : (T)0.0f;
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1 && tid < size_per_head)
    {
      key += self_K_bias[qkv_bias_id];
      key_cache[ite * offset + qkv_id] = key;
    }

    T val = (tid < size_per_head) ? key * sq[tid] * scalar : (T)(0.0f);
    T qk = blockReduceSum(val);
    if(threadIdx.x == 0)
      logits[ite] = qk;
    __syncthreads(); //try to remove
  }
  __syncthreads(); //try to remove

  /*
   * 接下来要对 logits 中的元素求 softmax，为了避免数值溢出，我们要让每个元素减去 logits 中的最大值。
   * 调整后的 softmax 公式如下：

  205.png

定义了两个共享内存变量 s_max_val 和 s_sum 分别存储 logits 中的最大值以及元素的指数和，
   最大值和指数和都使用块内规约获得。这块逻辑比较清晰，读者可以看代码理解。*/
  __shared__ float s_max_val, s_sum;
  float local_i = tid < step ? (float)logits[tid] : -1e20f; 
  float max_val = blockReduceMax<float>(local_i);
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  local_i -= s_max_val;
  float local_o = tid < step ? __expf(local_i) : 0.0f;
  float val = blockReduceSum<float>(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();

  if(tid < step)
    logits[tid] = local_o / s_sum;
  __syncthreads();

  /*3.5.2.4 计算 attention
根据 attention 的计算逻辑，得到 attention scores 后，
   右乘一个 value 矩阵就得到 attention out。
   具体含义就是，attention scores 代表 query 和 key 的相似度，用相似度当做权重系数，把所有 value 向量加权平均就得到 attention out。
   带着这个思路我们来看一下代码：

   首先是对 step 的循环，有多少个 step 就有多少个 value 向量，具体就是从 value_cache 中拿到 value 对应的元素，
   如果是当前 step 的 value，还需要先进行 add bias 操作并存入缓存变量，
   然后通过 value 值和 logits 中的 attention score 计算加权平均值 sum。
   循环结束后把 sum 存进 context 中的对应位置。可以用下面的公式表示。
   206.png
   */
  if(tid < size_per_head)
  {
    T sum = (T)0.0f;
    for(int ite = 0; ite < step; ++ite)
    {
      T value = value_cache[ite * offset + qkv_id];
      //for the last step, we should update K + bias_K to the cache
      if(ite == step - 1)
      {
        value += self_V_bias[qkv_bias_id];
        value_cache[ite * offset + qkv_id] = value;
      }
      sum += value * logits[ite];
    }
    context_buf[qkv_id] = sum;
  }
}

template <typename T>
__global__ 
void masked_attention_kernel_v2(T* query_buf, const T* self_Q_bias, 
  T* key_cache, const T* self_K_bias, T* value_cache, const T* self_V_bias,
  T* context_buf, int batch_size, int head_num, int size_per_head, const int step, const T scalar)
{
  extern __shared__ __align__(sizeof(T)) unsigned s_buf[];
  T* sq = reinterpret_cast<T *>(s_buf);
  T* logits = reinterpret_cast<T *>(&sq[size_per_head]);

  int tid = threadIdx.x;
  int bid = blockIdx.x / head_num;
  int head_id = blockIdx.x % head_num;

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head + tid;
  int qkv_bias_id = head_id * size_per_head + tid;

  if(tid < size_per_head)
    sq[tid] = query_buf[qkv_id] + self_Q_bias[qkv_bias_id];
  __syncthreads();

  int warp_size = 32;
  int offset = batch_size * head_num * size_per_head;
  int warp_ite = size_per_head / warp_size;

  T qk = (T)0.0f;

  //each warp process one step
  int step_id = threadIdx.x >> 5;
  if(step_id < step)
  {
    for(int wite = 0; wite < warp_ite; ++wite)
    {
      T key = key_cache[step_id * offset + bid * head_num * size_per_head + head_id * size_per_head 
        + tid % warp_size + wite * warp_size];
      //for the last step, we should update K + bias_K to the cache
      if(step_id == step - 1)
      { 
        key += self_K_bias[bid * head_num * size_per_head + head_id * size_per_head + 
          tid % warp_size + wite * warp_size];
        key_cache[step_id * offset + bid * head_num * size_per_head + head_id * size_per_head
          + tid % warp_size + wite * warp_size] = key;
      }
      qk += key * sq[tid % warp_size + wite * warp_size];
    }
  
    qk = warpReduceSum(qk * scalar);
    if(threadIdx.x % warp_size == 0)
    {
      logits[step_id] = qk;
      printf("step_id %d %f\n", step_id, qk);
    }
    
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;
  float local_i = tid < step ? (float)logits[tid] : -1e20f; 
  float max_val = blockReduceMax<float>(local_i);
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  local_i -= s_max_val;
  float local_o = tid < step ? __expf(local_i) : 0.0f;
  float val = blockReduceSum<float>(local_o);

  if(tid == 0)
    s_sum = val;
  __syncthreads();
  if(tid < step)
    logits[tid] = local_o / s_sum;
  __syncthreads();

  
  if(tid < size_per_head)
  {
    T sum = (T)0.0f;
    for(int ite = 0; ite < step; ++ite)
    {
      T value = value_cache[ite * offset + qkv_id];
      //for the last step, we should update K + bias_K to the cache
      if(ite == step - 1)
      {
        value += self_V_bias[qkv_bias_id];
        value_cache[ite * offset + qkv_id] = value;
      }
      sum += value * logits[ite];
    }
    context_buf[qkv_id] = sum;
  }
}

/*
3.5 masked_multi_head_attention
这一步的作用是对 tensor 进行 self-attention 操作，总共拆分了 5 个步骤，如下图所示。
204.png

 * @brief decoder layer masked_multi_head_attention
 *
 * @tparam OpType_
 * @param from_tensor               [batch_size_, hidden_units_]
 * @param key_cache_                [cache_size,] = [seq_len, batch_size_, hidden_units_]
 * @param value_cache_              [cache_size,] = [seq_len, batch_size_, hidden_units_]
 * @param decoder_output            [batch_size_, hidden_units_]
 * @param step

 我们先来看一下该函数的几个重要参数。

from_tensor：上一步经过 layerNorm 后的 from_tensor，形状为 [batch_size_, hidden_units_]

key_cache_：存储的是所有 step 的经过 Dense 变换（from_tensor_ * weight_K + bias）后的 from_tensor，
 形状为 [seq_len, batch_size_, hidden_units_]

value_cache_：存储的是所有 step 的经过 Dense 变换（from_tensor_ * weight_V + bias）后的 from_tensor，
 形状为 [seq_len, batch_size_, hidden_units_]

decoder_output：最终输出 tensor，形状为 [batch_size_, hidden_units_]

函数体内首先定义了两个变量 key_buf_ 和 value_buf_ 用来标记 key 和 value 在 key_cache_ 和 value_cache_ 中的存储位置，
 通过代码可以发现是分 step 存储的。看到这里，有些读者可能会问，为什么要搞 key_cache_ 和 value_cache_ 这么麻烦，存它有什么用？
 答案是计算结果复用。对于每一个 step 而言，只有 query 是新的，key 和 value 实际就是前面 step 的 from_tensor 经过 Dense 变换的，
 换句话说前面都计算过，重复计算没有意义，所以申请了两块内存存起来。
 * */
template<OperationType OpType_>
void OpenDecoder<OpType_>::masked_multi_head_attention(
  const DataType_* from_tensor,
  DataType_* key_cache_,
  DataType_* value_cache_,
  DataType_* decoder_output,
  const int step)
{
  int m = batch_size_;
  int n = hidden_units_;
  int k = hidden_units_;

  DataType_* key_buf_ = key_cache_ + (step - 1) * m * n;
  DataType_* value_buf_ = value_cache_ + (step - 1) * m * n;

  DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;
/*
 * 3.5.1 Dense 变换
这里直接使用 cuBLAS API 进行矩阵乘法，add bias 操作放在后面的 kernel 中进行。
 对 from_tensor 使用 3 个 矩阵乘法，计算结果分别存在 query_buf_、key_buf_ 和 value_buf_ 中。


 3.5.3 Dense 变换
这里直接使用 cuBLAS API 进行矩阵乘法，对 context_buf 右乘一个 output kernel 矩阵。
 add bias 操作放到后面的 kernel 中进行。

 3.7.3 Dense 变换
与 3.5.3 节一样，这里直接使用 cuBLAS API 进行矩阵乘法，
 对 context_buf 右乘一个 output kernel 矩阵得到 decoder_output。add bias 操作放到后面的 kernel 中进行。
 */
  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.self_attention.query_weight.kernel , AType_, n, 
    from_tensor, BType_, k, 
    &beta, 
    query_buf_, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.self_attention.key_weight.kernel, AType_, n, 
    from_tensor, BType_, k, 
    &beta, 
    key_buf_, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.self_attention.value_weight.kernel, AType_, n, 
    from_tensor, BType_, k, 
    &beta, 
    value_buf_, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

  dim3 grid(batch_size_ * head_num_);
  dim3 block(128);

  //suppose size_per_head <= 128
  if(step <= 64)
    block.x = 64;
  else if(step <= 128 && step > size_per_head_)
    block.x = 128;
  else if(step > 128 && step <= 256)
    block.x = 256;
  else if(step > 256 && step <= 512)
    block.x = 512;
  else
    block.x = 1024;

  if(block.x < size_per_head_)
    block.x = size_per_head_;
  
  assert(block.x <= 1024);

  DataType_ scalar = 1 / sqrtf(size_per_head_ * 1.0f);

  int shared_size = sizeof(DataType_) * (size_per_head_ + step);

  masked_attention_kernel<DataType_><<<grid, block, shared_size, param_.stream>>>(
    query_buf_, param_.self_attention.query_weight.bias, 
    key_cache_, param_.self_attention.key_weight.bias,
    value_cache_, param_.self_attention.value_weight.bias,
    context_buf_, batch_size_,
    head_num_, size_per_head_, step, scalar);

  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.self_attention.attention_output_weight.kernel, AType_, n, 
    context_buf_, BType_, k, 
    &beta, 
    decoder_output, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
} 

/*
3.7.1 Dense 变换
使用 cuBLAS API 进行矩阵乘法，主要注意一点，memory_tensor 的形状是 [batch_size_, memory_max_seq_len, memory_hidden_units_]，
 其只在 first step 进行计算，计算后形状为 [batch_size_, memory_max_seq_len, hidden_units_]。
 add bias 操作放在后面的 kernel 中进行。

3.7.2 cross_attention_kernel
核函数的执行配置参数确定方法和 masked_attention_kernel，具体见 3.5.2.1 节，这里不再重复介绍。

有所不同的是，这里提前定义的核函数共享内存大小是 size_per_head_ + seq_len，这个和之前有所不同，具体要看下源码。


 阅读源码可以发现，cross_attention_kernel 的实现逻辑完全就是照搬 masked_attention_kernel，
 只不过之前是当前 step 的 token 对应的 tensor 逐个和前面 step 的 token 对应的 tensor 进行 attention，
 现在是当前 step 的 token 对应的 tensor 逐个和 encoder out 中每个 token 对应的 tensor 进行 attention，
 把 step 换成了 length，最终计算结果存入 context_buf，形状为 [bacth_size_, hidden_units_]。
 另外就是注意这次的 key 和 value 在 first step 时就可以完成 add bias 计算并存入缓存变量。

笔者点评：在 encoder out 已知且当前 step 的 from_tensor 确定的情况下，笔者私以为完全可以考虑更大尺度的并行计算 cross attention，
 而不是在核函数内部使用循环一次计算一个 encoder token 和 decoder token 间的 attention score。
 为此笔者给出以下代码实现，因为实现思路不同所以一些参数和变量不能和上下文完全衔接，读者理解思路即可。

 * @brief add QKV bias and transpose kv
 *
 * @tparam T
 * @param Q                         [batch_size, head_num, size_per_head]
 * @param K                         [batch_size, seq_len, head_num, size_per_head]
 * @param V                         [batch_size, seq_len, head_num, size_per_head]
 * @param query_buf                 [batch_size, head_num, size_per_head]
 * @param key_buf                   [batch_size, head_num, seq_len, size_per_head]
 * @param value_buf                 [batch_size, head_num, seq_len, size_per_head]
 * @param Q_bias                    [head_num, size_per_head]
 * @param K_bias                    [head_num, size_per_head]
 * @param V_bias                    [head_num, size_per_head]
 * @param seq_len
 * @param head_num
 * @param size_per_head
 * @return __global__

    template<typename T>
    __global__ void add_QKV_bias_kernel_transpose(T* Q, T* K, T* V, T* query_buf, T* key_buf, T* value_buf, const T* Q_bias,
                                                  const T* K_bias, const T* V_bias, const int seq_len, const int head_num, const int size_per_head) {
        // grid_size = batch_size * seq_len, block_size = hidden_units
        int tid = threadIdx.x;
        int batch_id = blockIdx.x / seq_len;
        int seq_id = blockIdx.x % seq_len;
        int head_id = tid / size_per_head;
        int id_in_head = tid % size_per_head;
        int hidden_units = head_num * size_per_head;
        T bias_tmp;
        if (seq_id == 0) {
            bias_tmp = tid < hidden_units ? Q_bias[tid] : 0.0f;
            query_buf[batch_id * hidden_units + tid] = Q[batch_id * hidden_units + tid] + bias_tmp;
        }
        __syncthreads();
        int target_index = batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head +
                           seq_id * size_per_head + id_in_head;
        bias_tmp = tid < hidden_units ? K_bias[tid] : 0.0f;
        key_buf[target_index] = K[blockIdx.x * hidden_units + tid] + bias_tmp;
        bias_tmp = tid < hidden_units ? V_bias[tid] : 0.0f;
        value_buf[target_index] = V[blockIdx.x * hidden_units + tid] + bias_tmp;
    }

    template<typename T>
    __global__ void softmax_kernel(T* qk_buf, int* length_per_sample, const int seq_len, const int head_num, const T scalar) {
        // grid_size = batch_size * head_num
        int tid = threadIdx.x;
        int batch_id = blockIdx.x / head_num;
        int offset = blockIdx.x * seq_len;
        __shared__ T s_sum, s_max;
        int length = length_per_sample[batch_id];
        T qk = tid < length ? qk_buf[offset + tid] * scalar: -1000000.0f;
        T max_val = blockReduceMax<T>(qk);
        if (tid == 0) {
            s_max = max_val;
        }
        __syncthreads();
        T qk_tmp = tid < seq_len ? __expf(qk - s_max) : 0.0f;
        T sum_val = blockReduceSum<T>(qk_tmp);
        if (tid == 0) {
            s_sum = sum_val + 1e-6f;
        }
        if (tid < seq_len) {
            qk_buf[offset + tid] = qk_tmp / s_sum;
        }
    }

 * @brief
 *
 * @tparam T
 * @param from_tensor                [batch_size_, hidden_units_]
 * @param K                          [batch_size_, mem_max_seq_len, hidden_units_]
 * @param V                          [batch_size_, mem_max_seq_len, hidden_units_]
 * @param query_buf                  [batch_size_, hidden_units_]
 * @param key_buf                    [batch_size_, mem_max_seq_len, hidden_units_]
 * @param value_buf                  [batch_size_, mem_max_seq_len, hidden_units_]
 * @param Q_bias                     [hidden_units_,]
 * @param K_bias                     [hidden_units_,]
 * @param V_bias                     [hidden_units_,]
 * @param context_buf                [batch_size_, hidden_units_]
 * @param batch_size
 * @param head_num
 * @param length                     [bacth_size,]
 * @param size_per_head
 * @param step
 * @param seq_len                    mem_max_seq_len
 * @param scalar
 * @return

    template<typename T>
    void customer_cross_attention(T* from_tensor, T* K, T* V, T* query_buf, T* key_buf, T* value_buf, T*qk_buf, const T* Q_bias,
                                  const T* K_bias, const T* V_bias, T* context_buf, int batch_size, int head_num, int* length,
                                  int size_per_head, const int seq_len, const T scalar) {

        dim3 grid(batch_size * seq_len);
        dim3 block(1024);
        add_QKV_bias_kernel<T><<<grid, block, 0, steam>>>(from_tensor, K, V, query_buf, key_buf, value_buf,
                Q_bias, K_bias, V_bias, seq_len, head_num, size_per_head);

        int m = 1;
        int n = seq_len;
        int k = size_per_head;
        check_cuda_error(cublasGemmStridedBatchedEx(param_.cublas_handle,
                                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                                    n, m, k,
                                                    &alpha,
                                                    key_buf, AType_, k, n * k,
                                                    query_buf, BType_, k, m * k,
                                                    &beta,
                                                    qk_buf, CType_, n, m * n,
                                                    batch_size * head_num,
                                                    computeType_,
                                                    static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));

        grid(batch_size * head_num);
        block(1024);
        softmax_kernel<T><<<grid, block, 0, stream>>>(qk_buf, length, seq_len, head_num, scalar);
        k = seq_len;
        n = size_per_head;
        check_cuda_error(cublasGemmStridedBatchedEx(param_.cublas_handle,
                                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                                    n, m, k,
                                                    &alpha,
                                                    value_buf, AType_, n, n * k,
                                                    qk_buf, BType_, k, m * k,
                                                    &beta,
                                                    context_buf, CType_, n, m * n,
                                                    batch_size * head_num,
                                                    computeType_,
                                                    static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));
    }
    总的来说，笔者把 cross attention 分为 4 步。
    第一步调用核函数 add_QKV_bias_kernel 对 from_tensor 和 key、query 进行 add bias 操作，
    grid_size 设置为 batch_size * seq_len，由于 from_tensor 没有 seq_len 这个维度，所以只有当 seq_id == 0 的 block 需要进行计算，
    另外结束后再对 key 和 value 进行 transpose 将其形状变为 [batch_size, head_num, seq_len, size_per_head]。
    第二步，调用 cuBLAS API 进行矩阵乘法，实现
    ，得到 attention scores qk_buf，形状为 [batch_size_, head_num, 1, seq_len]。
    第三步调用核函数 softmax_kernel 对 qk_buf 进行 softmax，实现
    。第四步调用 cuBLAS API 进行矩阵乘法，实现
    ，得到 context_buf，形状为 [batch_size_, head_num, 1, size_per_head] 相当于 [batch_size_, hidden_units_]。

    写完之后回顾，发现似乎也并不见得真的提升了计算效率，毕竟笔者的计算逻辑虽然容易理解，且由于没有在核函数内部使用循环从而并行程度更高，
    但总的来说需要四步，核函数调用开销要大于源码的实现方式，姑且也算一种思路吧。

    */
/**
 * @brief
 *
 * @tparam T
 * @param query_buf                   [batch_size_, hidden_units_]
 * @param Q_bias                      [hidden_units_,] = [head_num, size_per_head]
 * @param key_cache                   [batch_size_, mem_max_seq_len, hidden_units_]
 * @param K_bias                      [hidden_units_,] = [head_num, size_per_head]
 * @param value_cache                 [batch_size_, mem_max_seq_len, hidden_units_]
 * @param V_bias                      [hidden_units_,] = [head_num, size_per_head]
 * @param length_per_sample           [batch_size_,]
 * @param context_buf
 * @param batch_size
 * @param head_num
 * @param size_per_head
 * @param step
 * @param seq_len                     mem_max_seq_len
 * @param scalar
 * @return __global__
 */
 * */
template<typename T>
__global__
void cross_attention_kernel(
  T* query_buf, const T* Q_bias,
  T* key_cache, const T* K_bias,
  T* value_cache, const T* V_bias,
  const int* length_per_sample, T* context_buf, 
  int batch_size, int head_num, int size_per_head, int step, const int seq_len, const T scalar)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x / head_num;
  int head_id = blockIdx.x % head_num;

  extern __shared__ __align__(sizeof(T)) unsigned s_buf[];
  T* sq = reinterpret_cast<T *>(s_buf);
  T* logits = reinterpret_cast<T *>(&sq[size_per_head]);

  int length = __ldg(&length_per_sample[bid]);

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head + tid;
  int qkv_bias_id = head_id * size_per_head + tid;

  if(tid < size_per_head)
    sq[tid] = query_buf[qkv_id] + Q_bias[qkv_bias_id];
  __syncthreads();

  for(int ite = 0; ite < length; ++ite)
  {
    int key_id = bid * (seq_len * head_num * size_per_head) + ite * (head_num * size_per_head)
     + head_id * size_per_head + tid;

    T key = tid < size_per_head ? key_cache[key_id] : (T)(0.0f);

    //For the first step, we should add bias to key memory cache.
    //The KV memory cache only need to be updated at the first step.
    if(step == 1 && tid < size_per_head)
    {
      key += K_bias[head_id * size_per_head + tid];
      key_cache[key_id] = key;
    }

    T val = (tid < size_per_head) ? key * sq[tid] * scalar : (T)(0.0f);
    T qk = blockReduceSum(val);
    if(threadIdx.x == 0)
      logits[ite] = qk;
    __syncthreads(); //try to remove
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;

  float local_i = tid < length ? (float)logits[tid] : -1e20f; 
  float max_val = blockReduceMax<float>(local_i);
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  local_i -= s_max_val;
  float local_o = tid < length ? __expf(local_i) : 0.0f;
  float val = blockReduceSum<float>(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();
  if(tid < length)
    logits[tid] = local_o / s_sum;
  __syncthreads();

  if(tid < size_per_head)
  {
    T sum = (T)0.0f;
    for(int ite = 0; ite < length; ++ite)
    {
      int value_id = bid * seq_len * head_num * size_per_head + ite * head_num * size_per_head 
        + head_id * size_per_head + tid;

      T value = value_cache[value_id];

      //for the first step, we should add bias to key memory cache
      if(step == 1)
      {
        value += V_bias[head_id * size_per_head + tid];
        value_cache[value_id] = value;
      }  
      sum += value * logits[ite];
    }
    context_buf[bid * head_num * size_per_head + head_id * size_per_head + tid] = sum;
  }
}

/*
3.7 cross_multi_head_attention
cross attention 计算的是 decoder 的 tensor 与 encoder out 之间的 attention，
 具体地，decoder 的每个 step 下 token 对应的 from_tensor 作为 query，encoder out 作为 key，
 由于 encoder out 是不变的，权重参数也是不变的，所以 query_K 和 query_V 只在 first step 即可，
 把结果存到 key_mem_cache 和 value_mem_cache 中。
207.png
/**
 * @brief attention with source sentence
 *
 * @tparam OpType_
 * @param from_tensor                   [batch_size_, hidden_units_]
 * @param memory_tensor                 [batch_size_, memory_sequence_length, memory_hidden_units_]
 * @param key_mem_cache                 [batch_size_, memory_sequence_length, hidden_units_]
 * @param value_mem_cache               [batch_size_, memory_sequence_length, hidden_units_]
 * @param decoder_output                [batch_size_, hidden_units_]
 * @param length                        memory_sequence_length
 * @param seq_len                       mem_max_seq_len_
 * @param step
 */
 * */
/* attention with source sentence */
template<OperationType OpType_>
void OpenDecoder<OpType_>::cross_multi_head_attention(
  const DataType_* from_tensor,
  const DataType_* memory_tensor,
  DataType_* key_mem_cache,
  DataType_* value_mem_cache,
  DataType_* decoder_output,
  const int* length,
  const int seq_len,
  const int step)
{
  int m = batch_size_;
  int n = hidden_units_;
  int k = hidden_units_;

  DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;

  //reuse the query_buf 
  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.cross_attention.query_weight.kernel, AType_, n, 
    from_tensor, BType_, k, 
    &beta, 
    query_buf_, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

  if(step == 1)
  {
    m *= seq_len;
    k = memory_hidden_units_;
    check_cuda_error(cublasGemmEx(param_.cublas_handle, 
      CUBLAS_OP_N, CUBLAS_OP_N, 
      n, m, k, 
      &alpha, 
      param_.cross_attention.key_weight.kernel, AType_, n, 
      memory_tensor, BType_, k, 
      &beta, 
      key_mem_cache, CType_, n, 
      computeType_, 
      static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));

    check_cuda_error(cublasGemmEx(param_.cublas_handle, 
      CUBLAS_OP_N, CUBLAS_OP_N, 
      n, m, k, 
      &alpha, 
      param_.cross_attention.value_weight.kernel, AType_, n, 
      memory_tensor, BType_, k, 
      &beta, 
      value_mem_cache, CType_, n, 
      computeType_, 
      static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));
    k = hidden_units_;
  }

  dim3 grid(batch_size_ * head_num_);
  dim3 block(128);

  if(seq_len <= 64)
    block.x = 64;
  else if(seq_len <= 128 && seq_len > size_per_head_)
    block.x = 128;
  else if(seq_len > 128 && seq_len <= 256)
    block.x = 256;
  else if(seq_len > 256 && seq_len <= 512)
    block.x = 512;
  else
    block.x = 1024;

  if(block.x < size_per_head_)
    block.x = size_per_head_;

  assert(block.x <= 1024);
  
  DataType_ scalar = 1 / sqrtf(size_per_head_ * 1.0f);

  int shared_size = sizeof(DataType_) * (size_per_head_ + seq_len);
  cross_attention_kernel<DataType_><<<grid, block, shared_size, param_.stream>>>(
    query_buf_, param_.cross_attention.query_weight.bias, 
    key_mem_cache, param_.cross_attention.key_weight.bias,
    value_mem_cache, param_.cross_attention.value_weight.bias,
    length, context_buf_,  
    batch_size_,
    head_num_, size_per_head_, step, seq_len, scalar);

  m = batch_size_;
  n = head_num_ * size_per_head_;
  k = n;

  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.cross_attention.attention_output_weight.kernel, AType_, n, 
    context_buf_, BType_, k, 
    &beta, 
    decoder_output, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
}

template <typename T>
__global__
void decoder_norm1_kernel(const T* input, const T* gamma, const T* beta, T* output, int m, int n)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = tid < n ? (float)(__ldg(&input[blockIdx.x * n + tid])) : 0.0f;

  mean = blockReduceSum<float>(local_out);

  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>(tid < n ? (local_out - s_mean) * (local_out - s_mean) : 0.0f);

  if(threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6);

  __syncthreads();

  if(tid < n)
    output[blockIdx.x * n + tid] = 
      (T)(((local_out - s_mean) * s_variance) * (float)(__ldg(&gamma[tid])) + (float)(__ldg(&beta[tid])));
}

/*
3.6 decoder_norm2
该函数包含了 add bias、残差结构、layerNorm 等 3 个主要操作。
 函数内部直接调用一个核函数 decoder_norm2_kernel，核函数的的 grid_size 取 batch_size_，
 block_size 取 hidden_units_ 和 1024 中能被 32 整除的较小值。

相比 decoder_norm1，只是加了两行代码，分别实现 add input 和 add bias 操作，
 逻辑较为简单，建议读者直接看代码。


 3.8 decoder_norm2
与 3.6 节一样，实现了 add bias、残差结构、layerNorm 等 3 个主要操作，只不过入参变了而已。
/**
 * @brief 该函数包含了 add bias、残差结构、layerNorm 等 3 个主要操作。
 *
 * @tparam T
 * @param input               from_tensor      [batch_size_, hidden_units_]
 * @param gamma
 * @param beta
 * @param bias                                 [hidden_units,]
 * @param output              masked_attn_out  [batch_size_, hidden_units_]
 * @param norm_output         norm_out         [batch_size_, hidden_units_]
 * @param m                   batch_size_
 * @param n                   hidden_units_
 * @return __global__
 */
 * */
template <typename T>
__global__
void decoder_norm2_kernel(const T* input, const T* gamma, const T* beta, const T* bias, T* output, T* norm_output, int m, int n)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  if(tid < n)
  {
    local_out = (float)(__ldg(&input[blockIdx.x * n + tid]));
    local_out += (float)(output[blockIdx.x * n + tid]);
    local_out += (float)(__ldg(&bias[tid]));
    output[blockIdx.x * n + tid] = (T)local_out;
  }

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>(tid < n ? (local_out - s_mean) * (local_out - s_mean) : 0.0f);
  if(threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6);
  __syncthreads();

  if(tid < n)
    norm_output[blockIdx.x * n + tid] = 
      (T)((local_out - s_mean) * s_variance * (float)(__ldg(&gamma[tid])) + (float)(__ldg(&beta[tid])));
}

//    3.4 decoder_norm1
//            这个函数的作用是对输入 tensor 进行层归一化操作，内部调用了一个核函数 decoder_norm1_kernel 完成计算，代码如下：

/**
 * @brief 对输入 tensor 进行层归一化操作
 *
 * @tparam OpType_
 * @param input                 [batch_size_, hidden_units_]
 * @param gamma
 * @param beta
 * @param output                [batch_size_, hidden_units_]
 * @param m                     batch_size_
 * @param n                     hidden_units_
 *
 *
有读者看了注释可能要问，为什么这里的输入输出 tensor 的形状是 [batch_size_, hidden_units_]，
 而不是 [batch_size_, seq_len, hidden_units_] ？

 这是因为推理场景下解码的过程是一个 step 一个 step 进行的，一次只能解码一个 token，所以没有 seq_len 这个维度了。

 另外补充说一下，这里的 hidden_units_ = size_per_head * head_num。
通过源码可以发现，这里 grid_size 直接设置为 batch_size_，一个 block 处理一个 token。
 当 hidden_units_ 小于 1024 且可以被 32 整除时，把 block_size 设置为 hidden_units_，
 除此之外 block_size 都直接取 1024。

 这里可以预想到计算规模不会太大，所以两个核函数的超参数不需要考虑太多，直接根据业务需要设置即可。
 具体 layerNormalization 的计算公式原理，可以参照一下笔者上一篇文章【CUDA编程】Faster Transformer v1.0 源码详解，这里不再赘述。

核函数void decoder_norm1_kernel内先定义了两个共享内存变量 s_mean 和 s_variance，以及两个寄存器变量 mean 和 variance。
 先从全局内存 input 中取出当前线程对应的变量的值 local_out，执行一次块内规约得到 block 内元素的和，存入 mean，
 在 0 号线程内求出均值 mean，然后存入 s_mean，同样的方法求出 s_variance，然后根据公式算出结果即可。
 */
template<OperationType OpType_>
void OpenDecoder<OpType_>::decoder_norm1(
  const DataType_* input,
  const DataType_* gamma,
  const DataType_* beta,
  DataType_* output,
  int m, int n)
{
  dim3 grid(m);
  dim3 block(min(n, 1024));

  /* For general cases, n is equal to hidden_units, e.g., 512/1024.
     Since we have warp shuffle inside the code, block.x % 32 should be 0.
  */
  if(n % 32 != 0)
    block.x = 1024;

  assert(n <= 1024);

/* should pay attention to the rsqrt precision*/
  decoder_norm1_kernel<DataType_><<<grid, block, 0, param_.stream>>>(input, gamma, beta, output, m, n);
}

template<OperationType OpType_>
void OpenDecoder<OpType_>::decoder_norm2(
  const DataType_* input,
  const DataType_* gamma,
  const DataType_* beta,
  const DataType_* bias,
  DataType_* output,
  DataType_* norm_output,
  int m, int n)
{
  dim3 grid(m);
  dim3 block(min(n, 1024));

  assert(n <= 1024);

  /* For general cases, n is equal to hidden_units, e.g., 512/1024.
     Since we have warp shuffle inside the code, block.x % 32 should be 0.
  */

  if(n % 32 != 0)
    block.x = 1024;

  /* should pay attention to the rsqrt precision*/
  decoder_norm2_kernel<DataType_><<<grid, block, 0, param_.stream>>>(input, gamma, beta, bias, output, norm_output, m, n);
}

/*3.9.1 from_tensor * inter kernel
FeedForward 层第一次线性变换会扩展 from_tensor 的最后一个维度的长度，
 源码中将 hudden_units_ 扩展为原来的 4 倍，
 所以这里的 inter kernel 的形状为 [hudden_units_, 4 * hudden_units_]，
 矩阵运算后的输出 ffn_inner 形状为 [batch_size_, 4 * hudden_units_]。
 */
template<OperationType OpType_>
void OpenDecoder<OpType_>::ffn(
  const DataType_* input,
  DataType_* ffn_inner,
  DataType_* output,
  const int m,
  const int inner_size,
  const int n)
{
  int m1 = m, k1 = n, n1 = inner_size;
  DataType_ alpha = (DataType_)1.0f;
  DataType_ beta = (DataType_)0.0f;

  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n1, m1, k1, 
    &alpha, 
    param_.ffn.intermediate_weight.kernel, AType_, n1, 
    input, BType_, k1, 
    &beta, 
    ffn_inner, CType_, n1, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[2])));

  dim3 grid(m1);
  dim3 block(n1 / 4);

  assert(block.x <= 1024);

  add_bias_relu<DataType_><<<grid, block, 0, param_.stream>>>(ffn_inner, param_.ffn.intermediate_weight.bias, m1, n1);

  /*3.9.3 inter out * out kernel
FeedForward 层第二次线性变换将 tensor 的最后一个维度的长度转换为原始大小，
   源码中将 n2 赋值为 hudden_units_，
   所以这里的 out kernel 的形状为 [4 * hudden_units_, hudden_units_]，
   矩阵运算后的输出 tensor 形状为 [batch_size, hudden_units_]。*/
  int m2 = m, n2 = n, k2 = inner_size;
  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n2, m2, k2, 
    &alpha, 
    param_.ffn.output_weight.kernel, AType_, n2, 
    ffn_inner, BType_, k2, 
    &beta, 
    output, CType_, n2, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[3])));
}

/*3.9.4 add_bias_input
顾名思义，add_bias_input 函数包含了 add bias 和 add input 两个操作，
 前者是 Dense 变换的一部分，后者是残差结构。朴实无华的计算逻辑，没什么好说的，直接上代码。

 至此 Decoder 模块的计算逻辑已经介绍完毕。下面将对 Decoding 模块的源码进行解读
 */
template <typename T>
__global__ 
void add_bias_input_kernel(T* output, const T* input, const T* bias, const int m, const int n)
{
  int id = blockIdx.x * n + threadIdx.x;
  output[id] = output[id] + input[id] + __ldg(&bias[threadIdx.x]);
}

template<OperationType OpType_>
void OpenDecoder<OpType_>::add_bias_input(DataType_* output, const DataType_* input, const int m, const int n)
{
  dim3 grid(m);
  dim3 block(n);
  assert(n <= 1024);
  add_bias_input_kernel<<<grid, block, 0, param_.stream>>>(output, input, param_.ffn.output_weight.bias, m, n);
}

template void OpenDecoder<OperationType::FP32>::masked_multi_head_attention(
  const float* from_tensor,
  float* key_cache,
  float* value_cache,
  float* decoder_output,
  const int step);

template void OpenDecoder<OperationType::FP16>::masked_multi_head_attention(
  const half* from_tensor,
  half* key_cache,
  half* value_cache,
  half* decoder_output,
  const int step);

template void OpenDecoder<OperationType::FP32>::cross_multi_head_attention(
  const float* from_tensor,
  const float* memory_tensor,
  float* key_mem_cache,
  float* value_mem_cache,
  float* decoder_output,
  const int* length,
  const int max_seq_len,
  const int step);

template void OpenDecoder<OperationType::FP16>::cross_multi_head_attention(
  const half* from_tensor,
  const half* memory_tensor,
  half* key_mem_cache,
  half* value_mem_cache,
  half* decoder_output,
  const int* length,
  const int max_seq_len,
  const int step);

template void OpenDecoder<OperationType::FP32>::ffn(
  const float* input,
  float* ffn_inner, 
  float* otuput,
  const int m,
  const int inner_size,
  const int n);

template void OpenDecoder<OperationType::FP16>::ffn(
  const half* input,
  half* ffn_inner, 
  half* otuput,
  const int m,
  const int inner_size,
  const int n);

template void OpenDecoder<OperationType::FP32>::decoder_norm1(
  const float* input,
  const float* gamma,
  const float* beta,
  float* output,
  int m, int n);

template void OpenDecoder<OperationType::FP16>::decoder_norm1(
  const half* input,
  const half* gamma,
  const half* beta,
  half* output,
  int m, int n);

template void OpenDecoder<OperationType::FP32>::decoder_norm2(
  const float* input,
  const float* gamma,
  const float* beta,
  const float* bias,
  float* output,
  float* norm_output,
  int m, int n);

template void OpenDecoder<OperationType::FP16>::decoder_norm2(
  const half* input,
  const half* gamma,
  const half* beta,
  const half* bias,
  half* output,
  half* norm_output,
  int m, int n);

template void OpenDecoder<OperationType::FP32>::add_bias_input(
  float* output,
  const float* input,
  const int m,
  const int n);

template void OpenDecoder<OperationType::FP16>::add_bias_input(
  half* output,
  const half* input,
  const int m,
  const int n);

}//namespace FasterTransformer
