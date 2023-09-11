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
  #pragma unroll
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
  #pragma unroll
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

  int qkv_id = blockIdx.x * word_per_block / m;
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
void add_QKV_bias(half* Q, const half* bias_Q, half* K, const half* bias_K, half* V, const half* bias_V, 
  half* q_buf_, half* k_buf_, half* v_buf_, 
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

/*3.4.2 add_QKV_bias_rebuild_padding
顾名思义，核函数内部实现了两个计算逻辑：add Q/K/V bias 和恢复填充值，我们来看下源码。

 核函数 grid_size 设置为 valid_word_num，一个 block 内部处理 hidden_uints = head_num*size_per_head 个元素。
 由于模型整体超参数要求 hidden_uints 不大于 1024，所以这里 block_size 直接就设置为 hidden_units，一个 thread 就处理一个元素。
核函数内部最重要的逻辑就是 tgt_id 的计算，有两点需要把握。
 首先是 tgt_batch_id 和 tgt_seq_id 的确定，通过源矩阵的索引加偏移量后计算得到。
 然后是 tgt_id 的确定，按照 [batch_size, head_num, seq_len, size_per_head] 的顺序计算即可。
不熟悉 attention 计算逻辑的读者可能会问，为什么就加了一个 add bias 的操作，核函数变得如此复杂？
 这其实是因为这里面其实还隐藏了一个 transpose 操作，
 把原来形状如 [valid_word_num, head_num, size_per_head] 的矩阵转换成了形状如 [batch_size, head_num, seq_len, size_per_head] 的矩阵，
 笔者在前面的文章说过，这是“多头”独立计算的逻辑使然。
 */
template<typename T>
__global__
void add_QKV_bias_rebuild_padding(T* Q, const T* bias_Q, T* K, const T* bias_K, T* V, const T* bias_V, T* q_buf_, T* k_buf_, T* v_buf_, 
  const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int* mask_offset)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int bdim = blockDim.x;

  const int tgt_batch_id = (bid + mask_offset[bid]) / seq_len;
  const int tgt_seq_id = (bid + mask_offset[bid]) % seq_len;
  const int tgt_head_id = tid / size_per_head;
  const int tgt_hidden_id = tid % size_per_head;

  const int src_id = bid * bdim + tid;
  const int tgt_id = tgt_batch_id * head_num * seq_len * size_per_head + \
                    tgt_head_id * seq_len * size_per_head + \
                    tgt_seq_id * size_per_head + \
                    tgt_hidden_id;
  
  q_buf_[tgt_id] = Q[src_id] + bias_Q[tid];
  k_buf_[tgt_id] = K[src_id] + bias_K[tid];
  v_buf_[tgt_id] = V[src_id] + bias_V[tid];
}

template <typename T>
__global__
void softmax_kernel(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, const int seq_len, 
  const T scalar)
{
    int batch_id = blockIdx.x / head_num;
    int qk_offset = blockIdx.x * seq_len * seq_len;
    int mask_offset = batch_id * seq_len * seq_len;

    __shared__ float s_sum, s_max;

    for(int i = 0; i < seq_len; ++i)
    {
      float qk = threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
      float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;
      
      mask_val = (1.0f - mask_val) * -10000.0f;

      float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scalar + mask_val): -1e20f;

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
  const int seq_len, const float scalar)
{
    int batch_id = blockIdx.x / head_num / seq_len;
    int seq_id = blockIdx.x % seq_len;
    int qk_offset = blockIdx.x * seq_len;
    int mask_offset = batch_id * seq_len * seq_len + seq_id * seq_len;

    __shared__ float s_sum, s_max;

    float qk = threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
    float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;
      
    mask_val = (1.0f - mask_val) * -10000.0f;

    float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scalar + mask_val) : -1e20f;
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

//grid = (seq_len, batch_size, head_num)
//block.x = max(32, (seq_len + 31)/32*32)
template <typename T>
__global__
void softmax_kernel_v3(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, const int seq_len, const T scalar)
{

    float tmp = -1e20f;
    int qk_offset;
    __shared__ float s_mean, s_max;
    if (threadIdx.x < seq_len){
        qk_offset = ((blockIdx.y*head_num + blockIdx.z)*seq_len + blockIdx.x) *seq_len + threadIdx.x;
        int mask_offset = (blockIdx.y * seq_len + blockIdx.x) * seq_len + threadIdx.x;

        float qk = static_cast<float>(qk_buf_[qk_offset]);
        float mask_val = static_cast<float>(__ldg(&attr_mask[mask_offset]));

        mask_val = (1.0f - mask_val) * -10000.0f;

        tmp = qk * static_cast<float>(scalar) + mask_val;
    }

    float max_val = blockReduceMax<float>(tmp);
    if (threadIdx.x == 0){
        s_max = max_val;
    }
    __syncthreads();
    
    float qk_tmp = threadIdx.x < seq_len ? __expf(tmp - s_max) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);
    if (threadIdx.x == 0){
        s_mean = sum_val + 1e-6f;
        s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();
    
    if(threadIdx.x < seq_len)
      qk_buf_[qk_offset] = (T)(qk_tmp * s_mean);
}  


//grid = (seq_len, batch_size, head_num)
//block.x = max(32, (seq_len/2 + 31)/32*32)
//seq_len % 2 == 0
template <>
__global__
void softmax_kernel_v3(half* qk_buf_, const half* attr_mask, 
                      const int batch_size, const int head_num, 
                      const int seq_len, const half scalar)
{
    half2* qk_buf_half2Ptr = (half2*) qk_buf_;
    const half2* attr_mask_half2Ptr = (const half2*) attr_mask;

    int qk_offset;
    int threadIdx2 = threadIdx.x << 1;
    __shared__ float s_mean, s_max;
    half2 tmp = __float2half2_rn(0.0f);

    float max_val = -1e20f;
    half2 qk;
    if (threadIdx2 < seq_len){ 
        qk_offset = ((((blockIdx.y*head_num + blockIdx.z)*seq_len + blockIdx.x) *seq_len) >> 1) + threadIdx.x;
        int mask_offset = (((blockIdx.y * seq_len + blockIdx.x) * seq_len) >> 1) + threadIdx.x;

        qk = qk_buf_half2Ptr[qk_offset];
        half2 mask_val = __ldg(&attr_mask_half2Ptr[mask_offset]);
        half2 mask_val_tmp = __hmul2(__hsub2(__float2half2_rn(1.0f), mask_val), __float2half2_rn(-10000.0f));
        tmp = __hadd2(__hmul2(__half2half2(scalar), qk), mask_val_tmp);
        max_val = fmax((float)tmp.x, (float)tmp.y);
    }
    
    max_val = blockDim.x <= 32 ? warpReduceMax(max_val) : blockReduceMax<float>(max_val);

    if (threadIdx.x == 0){
        s_max = max_val;
    }
    __syncthreads();
    
    if (threadIdx2 < seq_len){
        tmp = h2exp(__hsub2(tmp, __float2half2_rn(s_max)));
    }
    float sum_val = blockDim.x <= 32 ? warpReduceSum((float)(tmp.x + tmp.y)) : blockReduceSum<float>((float)(tmp.x + tmp.y));

    if (threadIdx.x == 0){
        s_mean = sum_val + 1e-6f;
        s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();

    if(threadIdx2 < seq_len){
      qk = __hmul2(tmp, __float2half2_rn(s_mean));
      qk_buf_half2Ptr[qk_offset] = qk;
    }
}

//grid = (seq_len, batch_size, head_num)
//block.x = max(32, (seq_len + 31)/32*32)
//for seq_len not larger than 32
template <typename T>
__global__
void softmax_kernel_v3_LE32(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, const int seq_len, const T scalar)
{

    int qk_offset;
    __shared__ float s_mean, s_max;
    float tmp = -1e20f;
    if (threadIdx.x < seq_len){
        qk_offset = ((blockIdx.y*head_num + blockIdx.z)*seq_len + blockIdx.x) *seq_len + threadIdx.x;
        int mask_offset = (blockIdx.y * seq_len + blockIdx.x) * seq_len + threadIdx.x;

        float qk = static_cast<float>(qk_buf_[qk_offset]);
        float mask_val = static_cast<float>(__ldg(&attr_mask[mask_offset]));

        mask_val = (1.0f - mask_val) * -10000.0f;

        tmp = static_cast<float>(qk) * static_cast<float>(scalar) + mask_val;
    }
    float max_val = warpReduceMax<float>(tmp);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    tmp = threadIdx.x < seq_len ? __expf(tmp - s_max) : 0.0f;
    float sum_val = warpReduceSum<float>(tmp);

    if (threadIdx.x == 0){
      s_mean = sum_val + 1e-6f;
      s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();

    if(threadIdx.x < seq_len)
      qk_buf_[qk_offset] = (T)(tmp * s_mean);
}

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
void transpose(half* src, half* dst,
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

/*
 3.2.2 transpose_rebuild_padding
关于这个函数名的问题笔者前面已经吐槽过了，到此为止，这并不影响实际应用，不然也没法通过测试上线。。。
 这个函数内部实现了两个操作：transpose、删除填充值，
 下面来看一下代码。

 函数体内部有两行关于 bid 和 tid 的注释，这个是源码作者的注释，应该是写混了，不用去看，请听笔者一一解释。
首先这个函数的应用场景是在  softmax QK^T/ 根号dk  之后，也就是说这里的源矩阵是转置之后的矩阵，
 形状为 [batch_size, head_num, seq_len, size_per_head]，
 计算目的有两个：transpose、和删除填充值。
 gird_size 设置为 valid_word_num，block_size 设置为 head_num * size_per_head，一个 thread 处理一个元素。
 函数内部通过偏移量确定源矩阵的 batch_id 和 seq_id，
 有了索引后，再按照矩阵维度顺序把形如 [batch_size, head_num, seq_len, size_per_head] 的源矩阵，
 转换为形如 [valid_word_num, head_num, size_per_head] 的无填充矩阵。
 */
template<typename T>
__global__
void transpose_rebuild_padding(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head,
  const int* mask_offset)
{
  // TODO: optimize this kernel? 
  // do remove_sequence_length_padding
  const int tid = threadIdx.x; // batch * seq_len or valid_word_num
  const int bid = blockIdx.x; // head_num * size_per_head

  const int src_batch_id = (bid + mask_offset[bid]) / seq_len;
  const int src_seq_id = (bid + mask_offset[bid]) % seq_len;

  const int dst_seq_id = bid;

  const int head_id = tid / size_per_head;
  const int hidden_id = tid % size_per_head;
  dst[dst_seq_id * head_num * size_per_head + tid] = src[ src_batch_id * head_num * seq_len * size_per_head +
    head_id * seq_len * size_per_head + src_seq_id * size_per_head + hidden_id];
}

template<typename T>
__global__ void rebuild_sequence_length_padding(const T* src, T* tgt,
                                            const int* mask_offset,
                                            const int n)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int tgt_seq_id = bid + mask_offset[bid];
  const int src_seq_id = bid;

  for(int i = tid; i < n; i += blockDim.x)
  {
    tgt[tgt_seq_id * n + i] = src[src_seq_id * n + i];
  }
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
      const DataType_ scalar)
{
    const int k = head_num * size_per_head;

    dim3 grid;
    dim3 block;
    
    if(OpType_ == OperationType::FP32)
    {
      if(param_.sequence_id_offset == nullptr || param_.valid_word_num == batch_size * seq_len)
      {
        const int m = batch_size * seq_len;
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
        add_QKV_bias_rebuild_padding<DataType_><<<param_.valid_word_num, k, 0, stream>>>(Q, bias_Q, K, bias_K, 
          V, bias_V, q_buf_, k_buf_, v_buf_, 
          batch_size, seq_len, head_num, size_per_head, param_.sequence_id_offset);
      }
    }
    else
    {
      if(param_.sequence_id_offset == nullptr || param_.valid_word_num == batch_size * seq_len)
      {
        const int word_per_block = 1;
        grid.x = batch_size * seq_len / word_per_block;
        block.x = head_num * size_per_head * word_per_block / 2;
  
        assert(block.x <= 1024);
  
        add_QKV_bias<DataType_><<<grid, block, 0, stream>>>(Q, bias_Q, K, bias_K, V, bias_V, q_buf_, k_buf_, 
          v_buf_, batch_size, seq_len, head_num, size_per_head / 2, word_per_block);
      }
      else
      {
        add_QKV_bias_rebuild_padding<half2><<<param_.valid_word_num, k / 2, 0, stream>>>((half2*)Q, (const half2*)bias_Q, 
          (half2*)K, (const half2*)bias_K, (half2*)V, (const half2*)bias_V, 
          (half2*)q_buf_, (half2*)k_buf_, (half2*)v_buf_,
          batch_size, seq_len, head_num, size_per_head / 2, param_.sequence_id_offset);
      }
    }

    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());

    DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;
    
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

    //deal with odd seq_len
    if (seq_len % 2 != 0){
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
        softmax_kernel_v2<DataType_><<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scalar);
      }
      else
      {
        grid.x = batch_size * head_num;
        softmax_kernel<DataType_><<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scalar);
      }
    }
    //deal with even seq_len 
    else{
      grid.x = seq_len;
      grid.y = batch_size;
      grid.z = head_num;
      if (seq_len <= 32){
        block.x = 32;
        softmax_kernel_v3_LE32<DataType_><<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scalar);
      }
      else{
        if (OpType_ == OperationType::FP16){
          block.x = (seq_len/2 + 31)/32*32;
          softmax_kernel_v3<<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scalar);
        }
        else{
          block.x = (seq_len + 31)/32*32;
          softmax_kernel_v3<DataType_><<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scalar);
        }
      }
      grid.x = grid.y = grid.z = 1;
    }

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
    if(OpType_ == OperationType::FP16)
    {
      if(param_.sequence_id_offset == nullptr || param_.valid_word_num == batch_size * seq_len)
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
        transpose_rebuild_padding<half2><<<param_.valid_word_num, k / 2, 0, stream>>>(
          (half2*)transpose_dst_, (half2*)dst, 
          batch_size, seq_len, head_num, size_per_head / 2, param_.sequence_id_offset);
      }
    }
    else
    {
      if(param_.sequence_id_offset == nullptr || param_.valid_word_num == batch_size * seq_len)
      {
        const int seq_per_block = 1;
        grid.x = batch_size * head_num * seq_len / seq_per_block;
        block.x = seq_per_block * size_per_head;
        transpose<DataType_><<<grid, block, 0, stream>>>(transpose_dst_, dst, 
            batch_size, seq_len, head_num, size_per_head);
      }
      else
      {
        transpose_rebuild_padding<DataType_><<<param_.valid_word_num, k, 0, stream>>>(transpose_dst_, dst, 
            batch_size, seq_len, head_num, size_per_head, param_.sequence_id_offset);
      }
    }
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
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
      const float scalar);

template void OpenMultiHeadAttention<OperationType::FP16>::multiHeadAttr_nofuse_kernelLauncher(
      cudaStream_t stream,
      cublasHandle_t handle,
      half* Q,
      const half* bias_Q,
      half* K,
      const half* bias_K,
      half* V,
      const half* bias_V,
      const half* attr_mask,
      half* dst,
      const int batch_size,
      const int seq_len,
      const int head_num,
      const int size_per_head,
      const half scalar);
}//namespace cuda
}//namespace fastertransformer
