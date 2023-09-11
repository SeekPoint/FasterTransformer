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
#pragma once
#include <assert.h>
#include <array>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include "fastertransformer/arguments.h"
#include "fastertransformer/cuda/cuda_kernels.h"
#include <float.h>

namespace fastertransformer{

#define DO_SPLIT_SMALL_TOP_K_SOFTMAX
static const int SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE = 256;
static const int SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS = 128;
static const int MAX_K = 4;

/*
4.7.2.1 求TopK
在 v2.0 版本的 beam search 中也有求 TopK 的操作，不过当时那个计算思路就很粗糙，
 简单粗暴，总共分为两个 kernel，
 在第一个 kernel 里面，先是用块内规约求出当前线程对应值的最大值，把最大值存起来，
 然后变量赋值为极小值，然后线程内部直接循环 K 次，最后获得了 grid_size 个 TopK，
 然后再第二个 kernel 中把这个范围再缩小到 TopK。
 可以看到这是一种 native 的求 TopK 思路，在 v2.1 版本，求 TopK 的思路有所优化。
TopK 问题是一个经典算法问题，通常我们通过维护一个小根堆，堆里存了 K 个数据，
 每次新数据跟堆顶数据比较，大于堆顶元素就替换掉堆顶元素，然后重新建堆，遍历完所有元素后，堆中元素就是 TopK。
 这里源码中也使用了这个思路，但是并没有使用堆结构，
 而是定义了一个结构体 TopK，应该是作者嫌麻烦，
 因为 K 实在太小，就不折腾了，我们来看一下这个结构体。

 可以看到，结构体中有两个长度为 MAX_K 的数组变量，p 用来存索引，u 用来存值，一一对应并按值降序排列。
 为啥弄两个数组？是因为这里我们还需要元素的位置，也就是 word_id，这两个数组同步更新。
 除了成员变量以外还有两个成员函数，一个是初始化函数 init 主要用来初始化 p 和 u，另一个是 insert 函数用来“插入元素”和“建堆”。
 insert 函数中首先比较最后一个元素和新插入元素，满足以下任意条件后，将用新插入的元素替换掉 TopK 中最后一个元素。

        插入元素大于最后一个元素
        最后一个元素是初始化的标识，也就是数组没有满
        插入元素等于最后一个元素，但是插入元素的索引更小

 插入元素后，还得“建堆”保证堆顶元素最小，前面说过这里直接用排序代替“建堆”，
 所以源码就提供了一个冒泡排序，排序完成后，数组中的元素恢复降序排列。
 * */
template<typename T, int MAX_K>
struct TopK
{
    int p[MAX_K];
    T u[MAX_K];

    __device__ __forceinline__ void insert(T elem, int elem_id)
    {
        // 把插入元素跟最后一个元素比较，如果插入元素更大，则替换掉最后一个元素
        if (elem > u[MAX_K-1] || (p[MAX_K-1] == -1) || ((elem == u[MAX_K-1]) && (elem_id < p[MAX_K-1])))
        //if (elem > u[MAX_K-1] || ((elem == u[MAX_K-1]) && (elem_id < p[MAX_K-1])))
        {
            u[MAX_K-1] = elem;
            p[MAX_K-1] = elem_id;
        }

        // 冒泡排序，把 TopK 中的元素进行排序
        for(int k = MAX_K - 2; k >= 0; --k)
        {
            if ((u[k+1] > u[k]) || (p[k] == -1) || ((u[k+1] == u[k])&&(p[k+1] < p[k])))
            //if ((u[k+1] > u[k]) || ((u[k+1] == u[k])&&(p[k+1] < p[k])))
            {
                T u2 = u[k];
                int p2 = p[k];
                u[k] = u[k+1];
                p[k] = p[k+1];
                u[k+1] = u2;
                p[k+1] = p2;
            }
        }
    }

    __device__ __forceinline__ void init()
    {
      #pragma unroll
      for(int i = 0; i < MAX_K; i++)
      {
        p[i] = -1;
        u[i] = -FLT_MAX;
      }
    }
};

/*
可以看到，reduce_topk_op 是通过遍历一个 TopK 变量 b 的元素，
 不断 insert 到另一个 TopK 变量 a 的拷贝 res 中实现的合并工作。
有了操作函数以后，直接调用 cub 库的块内规约 API 就完成了块内规约，
 获取了整个 block 内的全局 TopK total。
 当 thread_id == 0 时，
 把这 k 个元素对应的 logit 和 word_id 写入 topk_tmp_val_buf 和 topk_tmp_id_buf 中。
 这里还有个 diversity_rate 参数，这应该是一个修正系数，
 但是笔者发现源码中实际设置为 0.0f 并没有启用。
 * */
template<typename T, int MAX_K>
__device__ __forceinline__ TopK<T, MAX_K> reduce_topk_op(const TopK<T, MAX_K>& a, const TopK<T, MAX_K>& b)
{
    TopK<T, MAX_K> res = a;
    for(int i = 0; i < MAX_K; ++i)
        res.insert(b.u[i], b.p[i]);
    return res;
}

template<typename T>
struct TopK_2
{
    int p = -1;
    T u = -FLT_MAX;

    __device__ __forceinline__ void insert(T elem, int elem_id)
    {
        if(elem > u)
        {
            u = elem;
            p = elem_id;
        }
    }

    __device__ __forceinline__ void init()
    {
        u = -FLT_MAX;
        p = -1;
    }
};

template<typename T>
__device__ __forceinline__ TopK_2<T> reduce_topk_op_2(const TopK_2<T>& a, const TopK_2<T>& b)
{
    return a.u > b.u ? a : b;
}

template <typename T>
void topK_kernelLauncher(T* log_probs,
                        T* temp_log_probs,
                        int* topk_tmp_id_buf,
                        T* topk_tmp_val_buf,
                        int* ids,
                        DecodingBeamsearchArguments args,
                        cudaStream_t stream);

template <typename T>
void topK_softMax(const T* log_probs, 
                  const float* bias, 
                  const bool* finished, 
                  T* cum_log_probs, 
                  int* ids, 
                  void * tmp_storage,
                  DecodingBeamsearchArguments args,
                  cudaStream_t stream);

/* *************************** end of BeamSearch kernel *********************************** */

/* ********************************** Sampling kernel *********************************** */

template <typename T>
void topK_sampling_kernel_kernelLauncher(T* log_probs,
                                        int* topk_tmp_id_buf,
                                        T* topk_tmp_val_buf,
                                        int* ids,
                                        int* sequence_length,
                                        bool* finished_buf,
                                        int random_num,
                                        DecodingSamplingArguments args,
                                        cudaStream_t stream);

template<typename T>
void topP_sampling_kernel_kernelLauncher(const T* log_probs,
                                        const int* id_vals,
                                        T* sorted_log_probs,
                                        int* sorted_id_vals, 
                                        int* topp_offset_buf,
                                        void* temp_storage,
                                        bool* finished_buf,
                                        int step,
                                        DecodingSamplingArguments args,
                                        int* output_ids, 
                                        int* sequence_length, 
                                        cudaStream_t stream);

/* *************************** end of Sampling kernel *********************************** */

}//namespace fastertransformer
