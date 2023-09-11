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
 * BERT Encoder transformer
 **/

#pragma once

#include <cuda_runtime.h>
#include "fastertransformer/allocator.h"
#include "fastertransformer/cuda/cuda_kernels.h"
#include "fastertransformer/cuda/open_attention.h"
#include "fastertransformer/encoder_transformer.h"
namespace fastertransformer{

template<typename T>
class EncoderInitParam
{
  public:
    const T* from_tensor;
    const T* to_tensor;
    const T* attr_kernel_Q;
    const T* attr_kernel_K;
    const T* attr_kernel_V;
    const T* attr_bias_Q;
    const T* attr_bias_K;
    const T* attr_bias_V;
    const T* attr_mask;
    const T* attr_output_kernel;
    const T* attr_output_bias;
    const T* attr_output_layernorm_gamma;
    const T* attr_output_layernorm_beta;
    const T* inter_kernel;
    const T* inter_bias;
    const T* output_kernel;
    const T* output_bias;
    const T* output_layernorm_gamma;
    const T* output_layernorm_beta;
    T* transformer_out;
    cublasHandle_t cublas_handle;
    cudaStream_t stream;
};

template<OperationType OpType_, template<OperationType> class MultiHeadAttention_>
class BertEncoderTransformerTraits;

template<template<OperationType> class MultiHeadAttention_>
class BertEncoderTransformerTraits<OperationType::FP32, MultiHeadAttention_>
{
  public:
    typedef float DataType;
    static const OperationType OpType = OperationType::FP32;
    typedef MultiHeadAttention_<OpType> MultiHeadAttention;
    static cudaDataType_t const computeType = CUDA_R_32F;
    static cudaDataType_t const AType = CUDA_R_32F;
    static cudaDataType_t const BType = CUDA_R_32F;
    static cudaDataType_t const CType = CUDA_R_32F;
    //add FP32 Traits here
};

template<template<OperationType> class MultiHeadAttention_>
class BertEncoderTransformerTraits<OperationType::HALF, MultiHeadAttention_>
{
  public:
    typedef __half DataType;
    static const OperationType OpType = OperationType::HALF;
    typedef MultiHeadAttention_<OpType> MultiHeadAttention;
    static cudaDataType_t const computeType = CUDA_R_16F;
    static cudaDataType_t const AType = CUDA_R_16F;
    static cudaDataType_t const BType = CUDA_R_16F;
    static cudaDataType_t const CType = CUDA_R_16F;
    //add HALF Traits here
};


template<class Traits_>
class BertEncoderTransformer:IEncoderTransformer<Traits_::OpType>
{
  const IAllocator& allocator_;
  typename Traits_::MultiHeadAttention *attention_;
  typedef typename Traits_::DataType DataType_;
  EncoderInitParam<DataType_> param_;

  const cudaDataType_t computeType_ = Traits_::computeType;
  const cudaDataType_t AType_ = Traits_::AType;
  const cudaDataType_t BType_ = Traits_::BType;
  const cudaDataType_t CType_ = Traits_::CType;
  int cublasAlgo_[3];

  DataType_* buf_;
  DataType_* attr_out_buf_;
  DataType_* attr_matmul_buf_;
  DataType_* inter_matmul_buf_;

  int batch_size_;
  int from_seq_len_;
  int to_seq_len_;
  int head_num_;
  int size_per_head_;
  public:
  BertEncoderTransformer(const IAllocator& allocator, int batch_size, int from_seq_len, 
      int to_seq_len, int head_num, int size_per_head): 
    allocator_(allocator), batch_size_(batch_size), from_seq_len_(from_seq_len),
    to_seq_len_(to_seq_len), head_num_(head_num), size_per_head_(size_per_head){
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif

    int m = batch_size_ * from_seq_len_;
    int k = head_num_ * size_per_head_;
    int n = k;

    int buf_size = m * n;

    try
    {
      buf_ = reinterpret_cast<DataType_*>(allocator_.malloc(sizeof(DataType_) * buf_size * 6));
      if(buf_ == nullptr)
        throw std::runtime_error(std::string("Tensorflow Allocator failed to allocate internal buffer."));

      attr_out_buf_ = buf_;
      attr_matmul_buf_ = attr_out_buf_ + buf_size;
      inter_matmul_buf_ = attr_matmul_buf_ + buf_size;

      attention_ = new typename Traits_::MultiHeadAttention(allocator_, batch_size_, from_seq_len_, to_seq_len_, head_num_, size_per_head_);
      FILE* fd = fopen("gemm_config.in", "r");
      int err = 0;
      if(fd == NULL)
        printf("gemm_config.in is not found\n");
      else
      {
        err = fscanf(fd, "%d%d%d%*d%*d", &cublasAlgo_[0], &cublasAlgo_[1], &cublasAlgo_[2]);
        fclose(fd);
      }
      if(err != 3)
      {
	 printf("loading GEMM algorithms error, using default GEMM algorithms!\n");
         if(Traits_::OpType == OperationType::FP32)
         {
            cublasAlgo_[0] = -1;
            cublasAlgo_[1] = -1;
            cublasAlgo_[2] = -1;
         }
         else
         {
            cublasAlgo_[0] = 99;
            cublasAlgo_[1] = 99;
            cublasAlgo_[2] = 99;
         }
      }
    }
    catch(std::runtime_error& error)
    {
      throw error;
    }
  }
  /**
   * Initialize the parameters in class
   * We will keep the Ctor empty to ensure the sub classes follow the same init routine.
   * Please be aware that no dynamic memory allocation should be placed
   **/
  void initialize(EncoderInitParam<DataType_> param)
  {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    param_ = param;
    cuda::MultiHeadInitParam<DataType_> multi_head_init_param;

    multi_head_init_param.from_tensor = param.from_tensor;
    multi_head_init_param.to_tensor = param.to_tensor;
    multi_head_init_param.attr_kernel_Q = param.attr_kernel_Q;
    multi_head_init_param.attr_kernel_K = param.attr_kernel_K;
    multi_head_init_param.attr_kernel_V = param.attr_kernel_V;
    multi_head_init_param.attr_bias_Q = param.attr_bias_Q;
    multi_head_init_param.attr_bias_K = param.attr_bias_K;
    multi_head_init_param.attr_bias_V = param.attr_bias_V;
    multi_head_init_param.attr_mask = param.attr_mask;
    multi_head_init_param.stream = param.stream;
    multi_head_init_param.cublas_handle = param.cublas_handle;
    multi_head_init_param.attr_out = attr_out_buf_;

    attention_->initialize(multi_head_init_param);
  }

  /**
   * do forward 
   **/
  void forward() override
  {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    try{
      attention_->forward();

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif
/*
 5 BertEncoderTransformer
BertEncoderTransformer 类中有两个重要的成员方法：构造函数、forward 方法。
 其中构造函数内主要进行一些参数初始化功能，设备内存的申请和初始化也在该函数内进行。
 forward 方法内主要是核心逻辑的实现。

5.1 attention forward
根据调用链可知，BertEncoderTransformer->forward() 中第一步就是 attention_->forward()，
 其中 attention_ 对象在构造函数中被定义，attention_->forward() 执行的就是第 4 节的内容。

5.2 对 attention out 做线性变换
根据流程图和调用链可知，这一步是对多头注意力的输出 tensor 做一层线性变换，右乘一个参数矩阵，
 其实就是一个不加激活函数的 Dense 层，分为矩阵乘法和 add bias 两个操作步骤，这里调用了 cuBLAS API 实现矩阵乘法。
 * */
      DataType_ alpha = (DataType_)1.0f;
      DataType_ beta = (DataType_)0.0f;
      int m = batch_size_ * from_seq_len_;
      int k = head_num_ * size_per_head_;
      int n = k;

      check_cuda_error(cublasGemmEx(param_.cublas_handle, 
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k, 
        &alpha, 
        param_.attr_output_kernel, AType_, n, 
        attr_out_buf_, BType_, k, 
        &beta, 
        attr_matmul_buf_, CType_, n, 
        computeType_, 
        static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

      add_bias_input_layernorm_kernelLauncher<DataType_>(attr_matmul_buf_, 
        param_.from_tensor, param_.attr_output_bias, param_.attr_output_layernorm_gamma,
        param_.attr_output_layernorm_beta, m, n, param_.stream);

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif
/*
5.4 FeedForward 结构
根据 Transformer 模型结构，多头注意力之后为了增强表达能力，加了一个 FeedForward 层，该结构内部就是两个 Dense 层，
 第一层 Dense 中使用了激活函数，第二层没有激活函数。
 所以 FeedForward 层中包含了 5 个操作：矩阵乘法、add bias、activation、矩阵乘法、add bias。

5.4.1 attention out * inter kernel
FeedForward 层第一次线性变换会扩展 tensor 的最后一个维度的长度，源码中将 latent_dim（也就是 n）扩展为原来的 4 倍，
 所以这里的 inter kernel 的形状为 [latent_dim, 4 * latent_dim]，
 矩阵运算后的输出 tensor 形状为 [batch_size, seq_len, 4 * latent_dim]。
 * */
      n *= 4;
      check_cuda_error(cublasGemmEx(param_.cublas_handle, 
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k, 
        &alpha, 
        param_.inter_kernel, AType_, n, 
        attr_matmul_buf_, BType_, k, 
        &beta, 
        inter_matmul_buf_, CType_, n, 
        computeType_, 
        static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));

      add_bias_act_kernelLauncher<DataType_>(inter_matmul_buf_, param_.inter_bias, m, n, param_.stream);

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif
/*
5.4.3 inter out * out kernel
FeedForward 层第二次线性变换将 tensor 的最后一个维度的长度转换为原始大小，
 源码中将 n 重新赋值为 latent_dim，
 所以这里的 out kernel 的形状为 [4 * latent_dim, latent_dim]，
 矩阵运算后的输出 tensor 形状为 [batch_size, seq_len, latent_dim]。
 * */
      n = k;
      k *= 4;
      check_cuda_error(cublasGemmEx(param_.cublas_handle, 
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k, 
        &alpha, 
        param_.output_kernel, AType_, n, 
        inter_matmul_buf_, BType_, k, 
        &beta, 
        param_.transformer_out, CType_, n, 
        computeType_, 
        static_cast<cublasGemmAlgo_t>(cublasAlgo_[2])));
      add_bias_input_layernorm_kernelLauncher<DataType_>(param_.transformer_out, attr_matmul_buf_, param_.output_bias, 
          param_.output_layernorm_gamma,
          param_.output_layernorm_beta,
          m, n, param_.stream);
#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif
    }
    catch(std::runtime_error& error)
    {
      throw error;
    }
  }

  void trt_initialize(DataType_* from_tensor, DataType_* to_tensor, DataType_* attr_mask, DataType_* out, cudaStream_t stream, cublasHandle_t cublas_handle)
  {
    param_.from_tensor = from_tensor;
    param_.to_tensor = to_tensor;
    param_.stream = stream;
    param_.transformer_out = out;
    param_.cublas_handle = cublas_handle;
    attention_->trt_initialize(from_tensor, to_tensor, attr_mask, stream, param_.cublas_handle);
  }

  ~BertEncoderTransformer()
  {
    delete attention_;
    allocator_.free(buf_);
  }
};

}
