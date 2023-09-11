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
/**
 * Open sourced multi-head attention
 **/

#include "fastertransformer/allocator.h"
#include "fastertransformer/cuda/multi_head_attention.h"
#include "fastertransformer/cuda/open_attention.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
namespace fastertransformer{
namespace cuda{

/**
 * Multi-head attetion open sourced
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


  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : -1e20f;
  val = warpReduceMax(val);

  return val;
}

  __inline__ __device__
int target_index(int id1, int id2, int id3, int id4, int dim_1, int dim_2, int dim_3, int dim_4)
{
  return id1 * (dim_2 * dim_3 * dim_4) + id3 * (dim_2 * dim_4) + id2 * dim_4 + id4;
}

/*
 *4.2 multiHeadAttr_nofuse_kernelLauncher
4.2.1 add_QKV_bias
上面说过 Dense 层包含矩阵乘法和 add_bias 操作，其中 add_bias 操作在核函数 add_QKV_bias 中完成，
 源码针对两种数据类型 fp16 和 fp32 分别提供了一个 kernel，只是网络结构有所差异。
针对 fp32，每个 block 处理一个 word，总共有 batch_size * seq_len * 3 个 block，对于 Q、K、V 3 个 tensor 而言，
 前 batch_size * seq_len 个 block 处理 Q，中间 batch_size * seq_len 个 block 处理 K，
 后 batch_size * seq_len 个 block 处理 V。示意图如下：

 002.png

 核函数第一部分先根据 block_id 确定当前处理的 tensor 具体是 Q、K、V 中的哪一个，从而拿到输入输出变量的内存地址。
第二部分求出 tensor 中对应元素的索引，首先我们知道输入输出 tensor 是一个四维的 array，所以应该有四个索引，
 按维度顺序依次是 batch_id、word_start_id、head_id、id_in_head，
 有读者看到这里可能会有疑问：为什么要计算这些索引，前面计算了矩阵偏移量 row_offset，完全可以在 block 内按 thread_id 索引就可以拿到对应元素。
 原因是在 add_QKV_bias 核函数中计算逻辑不仅仅是 add，还有 transpose，
 熟悉 dAttention multiHea的读者都知道对 Q、K、V 线性映射之后，紧接着就是一个 transpose 操作，
 目的是把 embedding_dim 这个维度划分成多个独立的 head，每个 head 后面单独进行 attention，所以要把 head 维度移到 seq_len 维度前面。

 换句话说这里的 transpose 解决的是“多头”的问题，和 attention 无关。
理解了前面的逻辑，第三部分就比较简单了，先进行 add 操作，
 然后将结果按照 [bacth_size, head_num, seq_len, size_per_head] 的维度顺序写在输出 tensor 中，
 这里隐含了一个 transpose，需要注意的是这个 transpose 操作是输出的 tensor 中元素存储顺序相对于输入 tensor 而言的，并不是对输入 tensor 做了变换。

针对 fp16，每个 block 同时处理 Q、K、V 上的同一个 word，同一个线程先后处理 3 个 word 上对应元素的计算逻辑，
 实际计算中把 half 都转成了 half2，使用标准库中的函数 __hadd2 运算。网络结构如下：
 003.png

从图中可以看出，block_size 为 embedding_dim 的一半，这是因为用了 half2 这个数据结构，实际上每个线程处理了 2 个元素，所以线程数量缩减一半。
 核函数内部逻辑分为 2 个部分：1、求出 tensor 中对应元素的索引。2、一次对 Q、K、V 进行 add 和 transpose 操作。
 ===>>>>void add_QKV_bias(__half* Q, const __half* bias_Q, __half* K, const __half* bias_K,

 * */
template<typename T>
__global__
void add_QKV_bias(T* Q, const T* bias_Q, T* K, const T* bias_K, T* V, const T* bias_V, T* q_buf_, T* k_buf_, T* v_buf_, 
  const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int word_per_block)
{

  T* data_ptr;
  T* buf_ptr;
  const T* bias_ptr;
  
  int m = batch_size * seq_len;
  int n = head_num * size_per_head;

    // 总共有3m个block，第一部分处理q，第二部分处理k，第三部分处理v，这里使用qkv_id区分处理哪个矩阵

  int qkv_id = blockIdx.x * word_per_block / m;

    // 矩阵偏移量
  int row_offset = (blockIdx.x * word_per_block % m) * n;

  if(qkv_id == 0)
  {
    data_ptr = Q + row_offset;
    buf_ptr = q_buf_;
    bias_ptr = bias_Q;
  }
  else if(qkv_id == 1)
  {
    data_ptr = K + row_offset;
    buf_ptr = k_buf_;
    bias_ptr = bias_K;
  }
  else
  {
    data_ptr = V + row_offset;
    buf_ptr = v_buf_;
    bias_ptr = bias_V;
  }

  int batch_id = (blockIdx.x * word_per_block % m) / seq_len;
  int head_id = threadIdx.x / size_per_head;
  int id_in_head = threadIdx.x % size_per_head;
  int word_start_id = (blockIdx.x * word_per_block) % seq_len;

  T bias = __ldg(&bias_ptr[threadIdx.x]);

  for(int i = word_start_id; i < word_start_id + word_per_block; ++i)
  {
    T tmp = data_ptr[threadIdx.x] + bias;

    int target_id = batch_id * (seq_len * head_num * size_per_head) + head_id * seq_len * size_per_head + 
      i * size_per_head + id_in_head;

    buf_ptr[target_id] = tmp;
    data_ptr += n;
  }
}

template <>
__global__
void add_QKV_bias(__half* Q, const __half* bias_Q, __half* K, const __half* bias_K, __half* V, const __half* bias_V, 
  __half* q_buf_, __half* k_buf_, __half* v_buf_, 
  const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int word_per_block)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = tid / (head_num * seq_len * size_per_head);
  int seq_id = (tid % (head_num * seq_len * size_per_head)) / (head_num * size_per_head);
  int head_id = (tid % (head_num * size_per_head)) / size_per_head;
  int id = tid % size_per_head;
  int target_id = target_index(batch_id, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);

  int bias_id = threadIdx.x;

  half2* src_ptr = (half2*)Q;
  half2* dst_ptr = (half2*)q_buf_;
  const half2* bias_ptr = (const half2*)bias_Q;

  dst_ptr[target_id] = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));

  src_ptr = (half2*)K;
  dst_ptr = (half2*)k_buf_;
  bias_ptr = (const half2*)bias_K;
  dst_ptr[target_id] = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));

  src_ptr = (half2*)V;
  dst_ptr = (half2*)v_buf_;
  bias_ptr = (const half2*)bias_V;
  dst_ptr[target_id] = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));
}

/**
 * @brief
 *
 * @tparam T
 * @param qk_buf_                 [batch_size, head_num, seq_len, seq_len]
 * @param attr_mask               [batch_size, seq_len, seq_len]
 * @param batch_size
 * @param head_num
 * @param seq_len
 * @param scaler                  缩放因子
 * @return __global__
 核函数内首先计算了每个元素偏移量，对于输入 tensor 而言，每个 block 处理 seq_len * seq_len 个数据，
 所以 block 内元素偏移量为 blockIdx.x * seq_len * seq_len，而对于 mask 矩阵而言，其维度为 [batch_size, seq_len, seq_len]，
 跟 head_num 无关，所以其偏移量为 batch_id * seq_len * seq_len。

接下来是一层循环，对于 seq_len * seq_len 矩阵而言，每个线程处理当前 thread_id 列的元素，
 每轮循环结束，处理该列下一行的元素。在每一轮循环中，所有的线程一起处理一行数据，首先拿到数据 qk 以及 mask_val。
 如果 mask_val 为 0，则给 mask_val 赋一个很小的值最后加在 qk 上使 qk 值很小，以致最终这个 softmax 分量趋于 0；
 如果 mask_val 为 1，则 mask 不干预后续计算。每个线程拿到处理后的 qk 值即 tmp 后，进行一次块内规约，
 即可求出当前行的最大值 max_val，然后为了避免指数运算导致数值溢出，让 tmp 减去 max_val 并求其指数值赋给 qk ，
 然后对 qk 再一次块内规约求出当前行的和 s_sum，最后让 qk 除以 s_sum 即可得到 softmax 值。
 核函数内要注意在两次块内规约后一定要进行一次块内同步，否则可能计算错误。

 当 batch_size * head_num <= 120 时，此时 batch 较小，grid_size 取 batch_size * head_num * seq_len，
 这时一个线程块内处理一行数据，每个线程内只处理一个的数据。
void softmax_kernel_v2(T* qk_buf_, const
 这种情况下不涉及循环处理，计算逻辑与前面 softmax_kernel 循环体内部计算逻辑相同，不再赘述。

 */
template <typename T>
__global__
void softmax_kernel(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, const int seq_len, 
  const T scaler)
{
    int batch_id = blockIdx.x / head_num;

    // batch偏移量
    int qk_offset = blockIdx.x * seq_len * seq_len;
    int mask_offset = batch_id * seq_len * seq_len;

    __shared__ float s_sum, s_max;

    // 每次处理一个seq_len的数据
    for(int i = 0; i < seq_len; ++i)
    {
      float qk = threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
      float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;

        // 对于某些padded的word，给一个很小的值使其近似达到不参与运算的目的
      mask_val = (1.0f - mask_val) * -10000.0f;

      float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scaler + mask_val): -1e20f;

      float max_val = blockReduceMax<float>(tmp);

      if(threadIdx.x == 0)
        s_max = max_val;
      __syncthreads();

      qk = threadIdx.x < seq_len ? __expf(tmp - s_max) : 0.0f;

      float sum_val = blockReduceSum<float>(qk);

      if(threadIdx.x == 0)
      {
        s_sum = sum_val + 1e-6f;
      }
      __syncthreads();

      if(threadIdx.x < seq_len)
        qk_buf_[threadIdx.x + qk_offset] = (T)(qk / s_sum);

      qk_offset += seq_len;
      mask_offset += seq_len;
    }
}


template <typename T>
__global__
void softmax_kernel_v2(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, 
  const int seq_len, const float scaler)
{
    int batch_id = blockIdx.x / head_num / seq_len;
    int seq_id = blockIdx.x % seq_len;
    int qk_offset = blockIdx.x * seq_len;
    int mask_offset = batch_id * seq_len * seq_len + seq_id * seq_len;

    __shared__ float s_sum, s_max;

    float qk = threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
    float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;
      
    mask_val = (1.0f - mask_val) * -10000.0f;

    float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scaler + mask_val) : -1e20f;
    float max_val = blockReduceMax<float>(tmp);
    if(threadIdx.x == 0)
      s_max = max_val;
    __syncthreads();

    float qk_tmp = threadIdx.x < seq_len ? __expf((float)(tmp - s_max)) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);

    if(threadIdx.x == 0)
    {
      s_sum = sum_val + 1e-6f;
    }
    __syncthreads();

    if(threadIdx.x < seq_len)
      qk_buf_[threadIdx.x + qk_offset] = (T)(qk_tmp / s_sum);
}

/*
4.2.5 transpose
前面说过，多头 attention out 的维度为 [batch_size, head_num, seq_len, size_per_head]，此时这些 head 已经完成使命了，
 通过独立的 head_num 组 attention 参数计算出了 attention out，最后需要做的就是把这 head_num 组 attention out 拼接起来，
 体现在 tensor 上就是做一次 transpose，将维度变为 [batch_size, seq_len, head_num, size_per_head]。
 源码针对 fp16 和 fp32 分别提供了一个核函数 transpose，计算逻辑和 add_QKV_bias 中 transpose 计算逻辑相同，索引按顺序乘即可。
 具体代码如下：
 * */
template<typename T>
__global__
void transpose(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
  int batch_id = blockIdx.x / (head_num * seq_len);
  int seq_id = blockIdx.x % seq_len;
  int head_id = (blockIdx.x % (head_num * seq_len))/ seq_len;
  dst[batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head
    + head_id * size_per_head + threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
}

template<>
  __global__
void transpose(__half* src, __half* dst,
    const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int batch_id = tid / (head_num * seq_len * size_per_head);
  int head_id = (tid % (head_num * seq_len * size_per_head)) / (seq_len * size_per_head);
  int seq_id = (tid % (seq_len * size_per_head)) / size_per_head;
  int id = tid % size_per_head;

  int target_id = target_index(batch_id, head_id, seq_id, id, batch_size, head_num, seq_len, size_per_head);
  half2* src_ptr = (half2*)src;
  half2* dst_ptr = (half2*)dst;

  dst_ptr[target_id] = src_ptr[tid];
}

template<OperationType OpType_>
void OpenMultiHeadAttention<OpType_>::multiHeadAttr_nofuse_kernelLauncher(
      cudaStream_t stream,
      cublasHandle_t cublas_handle,
      DataType_* Q,
      const DataType_* bias_Q,
      DataType_* K,
      const DataType_* bias_K,
      DataType_* V,
      const DataType_* bias_V,
      const DataType_* attr_mask,
      DataType_* dst,
      const int batch_size,
      const int seq_len,
      const int head_num,
      const int size_per_head,
      const DataType_ scaler)
{

    int m = batch_size * seq_len;
    int k = head_num * size_per_head;

    dim3 grid;
    dim3 block;

    if(OpType_ == OperationType::FP32)
    {
      const int word_per_block = 1;
      assert(k <= 1024);
      assert(m / word_per_block * 3 <= 65536);

      dim3 grid(m / word_per_block * 3);
      dim3 block(k);
      add_QKV_bias<DataType_><<<grid, block, 0, stream>>>(Q, bias_Q, K, bias_K, V, bias_V, q_buf_, k_buf_, v_buf_,
          batch_size, seq_len, head_num, size_per_head, word_per_block);
    }
    else
    {
      const int word_per_block = 1;
      grid.x = batch_size * seq_len / word_per_block;
      block.x = head_num * size_per_head * word_per_block / 2;

      add_QKV_bias<DataType_><<<grid, block, 0, stream>>>(Q, bias_Q, K, bias_K, V, bias_V, q_buf_, k_buf_, 
      v_buf_, batch_size, seq_len, head_num, size_per_head / 2, word_per_block);
    }

    /*
     * 4.2.2 计算 attention scores
    先来看一下 attention 的计算公式，定义如下：
    004.png
    其中，Attention(Q, K, V) = QK^T，也就是说这一步要解决的是一个矩阵计算，用 tensorflow 代码表示如下：
            scores = tf.matmul(query, key, transpose_b=True)
    针对矩阵运算，源码中直接调用了 cuBLAS API，具体代码如下：
不熟悉 attention 的读者可能会问 attention scores 的具体含义是什么，
     笔者在早期的文章中有过介绍，其实就是两个矩阵的词向量两两相乘，向量相乘有什么含义？
     相似度，这个分数就代表 Q、K 的相似度。
     */
    DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;
    //计算 q * k^T
    check_cuda_error(cublasGemmStridedBatchedEx(cublas_handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      seq_len, seq_len, size_per_head,
      &alpha,
      k_buf_, AType_, size_per_head, seq_len * size_per_head,
      q_buf_, BType_, size_per_head, seq_len * size_per_head,
      &beta,
      qk_buf_, CType_, seq_len, seq_len * seq_len,
      batch_size * head_num,
      computeType_,
      static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));

    /*
    4.2.3 softmax_kernel
    拿到 Q、K 的相似度之后，直观上只要右乘一个 V 就可以得到 attention out，其含义就是一个加权平均的概念，
     既然要加权平均，必然要对权值进行归一化处理，这里的 softmax 就是这个作用。
     关于 softmax 核函数的实现方法笔者在前两篇文章也有介绍，
     OneFlow 官方给出了更为高效的实现方式，其高效的原因主要在访存带宽处理上，有兴趣的读者可以移步。
     【CUDA编程】OneFlow Softmax 算子源码解读之WarpSoftmax，
     【CUDA编程】OneFlow Softmax算子源码解读之BlockSoftmax

     源码中核函数的 block_size 是根据 seq_len 确定的，取大于 seq_len 且为 32 的的最小值。

     另外在调用 softmax kernel 之前，会根据 batch_size * head_num 选择不同的 softmax kernel，
     主要是为了保证在大 batch 的情况下的计算效率，这里以 120 为阈值，应该是作者的经验数值。
     这里作者给出了 2 个 softmax kernel 的实现。

    当 batch_size * head_num > 120 时，此时 batch 内元素较多，
     grid_size 取 batch_size * head_num，这时一个线程内处理一个 seq_len 的数据。

     005.png

     * */
    // 计算softmax(qk)
    if(seq_len <= 32)
      block.x = 32;
    else if(seq_len > 32 && seq_len <= 64)
      block.x = 64;
    else if(seq_len > 64 && seq_len <= 128)
      block.x = 128;
    else if(seq_len > 128 && seq_len <= 256)
      block.x = 256;
    else if(seq_len > 256 && seq_len <= 512)
      block.x = 512;
    else
      block.x = 1024;

    if(batch_size * head_num <= 120)
    {
      grid.x = batch_size * head_num * seq_len;
      softmax_kernel_v2<DataType_><<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scaler); 
    }
    else
    {
      grid.x = batch_size * head_num;
      softmax_kernel<DataType_><<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scaler); 
    }

    /*
    4.2.4 计算多头 attention out
     这一步的意思就是使用 softmax 后的相似度矩阵右乘一个 V，得到多头注意力输出，
     注意这时候输出 tensor 的维度为 [batch_size, head_num, seq_len, size_per_head]。
     源码中直接调用了 cuBLAS API，具体代码如下：
     * */
    // 计算qk * v
    check_cuda_error(cublasGemmStridedBatchedEx(cublas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      size_per_head, seq_len, seq_len,
      &alpha,
      v_buf_, AType_, size_per_head, seq_len * size_per_head,
      qk_buf_, BType_, seq_len, seq_len * seq_len,
      &beta,
      transpose_dst_, CType_, size_per_head, seq_len * size_per_head,
      batch_size * head_num,
      computeType_,
      static_cast<cublasGemmAlgo_t>(cublasAlgo_[2])));

/* for half2 only */
    if(OpType_ == OperationType::HALF)
    {
      const int seq_per_block = 4;
      grid.x = batch_size * head_num * seq_len / seq_per_block;
      block.x = seq_per_block * size_per_head / 2;

      assert(grid.x * seq_per_block == batch_size * head_num * seq_len);

      transpose<DataType_><<<grid, block, 0, stream>>>(transpose_dst_, dst, 
          batch_size, seq_len, head_num, size_per_head / 2);
    }
    else
    {
      const int seq_per_block = 1;
      grid.x = batch_size * head_num * seq_len / seq_per_block;
      block.x = seq_per_block * size_per_head;
      transpose<DataType_><<<grid, block, 0, stream>>>(transpose_dst_, dst, 
          batch_size, seq_len, head_num, size_per_head);
    }
}

template void OpenMultiHeadAttention<OperationType::FP32>::multiHeadAttr_nofuse_kernelLauncher(
      cudaStream_t stream,
      cublasHandle_t handle,
      float* Q,
      const float* bias_Q,
      float* K,
      const float* bias_K,
      float* V,
      const float* bias_V,
      const float* attr_mask,
      float* dst,
      const int batch_size,
      const int seq_len,
      const int head_num,
      const int size_per_head,
      const float scaler);

template void OpenMultiHeadAttention<OperationType::HALF>::multiHeadAttr_nofuse_kernelLauncher(
      cudaStream_t stream,
      cublasHandle_t handle,
      __half* Q,
      const __half* bias_Q,
      __half* K,
      const __half* bias_K,
      __half* V,
      const __half* bias_V,
      const __half* attr_mask,
      __half* dst,
      const int batch_size,
      const int seq_len,
      const int head_num,
      const int size_per_head,
      const __half scaler);
}//namespace cuda
}//namespace fastertransformer
