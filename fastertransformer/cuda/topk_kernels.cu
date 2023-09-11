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

#include "fastertransformer/cuda/topk_kernels.cuh"
#include "cub/cub.cuh"

namespace fastertransformer
{
/*
 * TopK 结构介绍完之后，
 * 下面就是如何使用 TopK 结构完成对 logits 的求 TopK 操作。
 * 源码中使用 beam_topK_kernel 核函数来求 TopK，
 * grid_size 设置为 batch_size，block_size 设置为 256，
 * 也就是说一个 block 内要处理 vocab_size 个元素，从中选出 TopK，
 * 每个线程处理 vocab_size / 256 个元素，步长为 256。
 *
 *
 核函数内部首先使用 cub 库进行了块内规约前的准备，这个我们暂且不去看，
 之后内部定义了一个寄存器变量 partial，partial 存储了当前线程处理元素的 TopK，相当于当前线程下的小根堆，
 随后对 partial 进行初始化，这块其实可以直接调用成员函数 init 的，
 但是作者估计忘记还有这个函数了就又手写了一遍。
 然后就是对当前线程待处理的元素进行遍历，让 partial 来 insert 待处理元素，
 全部 insert 一遍后的 partial 其实就存储了当前线程处理的所有元素的 TopK。
 但是我们的目标是要获取整个 block 内的全局 TopK，所以我们还需要进行一次“大合并”，把所有的 TopK 合并成一个，
 这实际相当于一次块内规约操作，只是我们还需要定义一个操作函数，
 显然这个操作函数的输入是两个 TopK 类型的变量，
 输出是 TopK 类型，其计算逻辑就是把两个 TopK 合并成 1 个 TopK。
 源码提供了一个 reduce_topk_op 函数来完成这个任务。
 * */
template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__
void beam_topK_kernel(const T* log_probs, 
                        int* topk_tmp_id_buf,
                        T* topk_tmp_val_buf,
                        const int vocab_size,
                        T diversity_rate)
{
    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    TopK<T, MAX_K> partial;
    
    #pragma unroll
    for(int i = 0; i < MAX_K; ++i)
    {
        partial.p[i] = -1;
        partial.u[i] = -FLT_MAX;
    }

    #pragma unroll
    for(int elem_id = thread_id; elem_id < vocab_size; elem_id += THREADBLOCK_SIZE)
    {
        int index = elem_id + block_id * vocab_size;        
        partial.insert(log_probs[index], index);
    }

    TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

    if (thread_id == 0)
    {
        int index = block_id * MAX_K;
        
        #pragma unroll
        for(int i = 0; i < MAX_K; ++i)
        {
            topk_tmp_id_buf[index + i] = total.p[i];
            topk_tmp_val_buf[index + i] = total.u[i] + diversity_rate * (T)i;
        }
    }
}

template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__
void batch_topK_kernel(int* topk_tmp_id_buf,
                        T* topk_tmp_val_buf,
                        int* id_buf)
{
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    TopK<T, MAX_K> partial;
    if (thread_id == 0)
    {
        for(int i = 0; i < MAX_K; ++i)
        {
            partial.p[i] = -1;
            partial.u[i] = -FLT_MAX;
        }

        int index = block_id * MAX_K * MAX_K;
        for(int i = 0; i < MAX_K * MAX_K; i++)
        {
            partial.insert( (T)topk_tmp_val_buf[index + i], topk_tmp_id_buf[index + i]);
        }

        index = block_id * MAX_K;
        for(int i = 0; i < MAX_K; i++)
        {
            id_buf[index + i] = partial.p[i];
        }
    }
}

template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__
void batch_topK_kernel_v2(int* topk_tmp_id_buf,
                        T* topk_tmp_val_buf,
                        int* id_buf)
{
    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    TopK<T, MAX_K> partial;
    #pragma unroll
    for(int i = 0; i < MAX_K; ++i)
    {
        partial.p[i] = -1;
        partial.u[i] = -FLT_MAX;
    }

    int ite = MAX_K * MAX_K / THREADBLOCK_SIZE;
    #pragma unroll
    for(int i = 0; i < ite; i++)
    {
        int index = bid * MAX_K * MAX_K + i * THREADBLOCK_SIZE + tid;
        partial.insert( (T)topk_tmp_val_buf[index], topk_tmp_id_buf[index]);
    }

    TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

    if(tid == 0)
    {
        #pragma unroll
        for(int i = 0; i < MAX_K; i++)
            id_buf[bid * MAX_K + i] = total.p[i];
    }
}
/*
5.3 topK kernel 优化
关于 Top-k 采样解码前面已经介绍，这里说的 topK 特指 beam search 过程中的求 topK 操作，

 在 v2.0 版本中，固定设置 block_size 为 1024，
 通过第一个 topK kernel 把 ids 的形状缩小到 [batch_size, grid_size_1st, beam_width]，

 再经过第二个 topK kernel 求出最终每个样本的 topK。
 其中关于 batch_size 维度的每个样本的计算过程是通过循环实现的，并行程度不高。

 在 v2.1 版本，作者更新了 topK kernel，
 依然通过两个 kernel （topk_stage_1_opt3 和 topk_stage_2_opt3）完成 topK 计算。

在 topk_stage_1_opt3 中把 gird_size 设置为 batch_size * K * BLOCKS_PER_BEAM_，
 也就是说对于每一行 vocab_size 个元素，要使用 BLOCKS_PER_BEAM_ 个 block 参与计算。

 * @brief
 * // grid_size = batch_size * K * BLOCKS_PER_BEAM_
 * @tparam T
 * @tparam BLOCK_SIZE_
 * @tparam BLOCKS_PER_BEAM_
 * @param log_probs                 [batch_size, beam_width, vocab_size]
 * @param tmp_log_probs             [batch_size, beam_width, vocab_size]
 * @param topk_tmp_id_buf           [batch_size, beam_width, BLOCKS_PER_BEAM_, K]
 * @param topk_tmp_val_buf          [batch_size, beam_width, BLOCKS_PER_BEAM_, K]
 * @param k                          beam_width
 * @param vocab_size
 * @return __global__

 核函数内引入了一个新的数据结构 TopK_2，这个数据结构中有两个成员属性，p 和 u，分别代表存储了概率值和其对应的 word_id；
 有两个成员方法，init 和 insert，分别进行初始化和更新，insert 方法非常简单，就是单纯的把更大的值和索引更新到对象中。
 接下来，我们看一下 topk_stage_1_opt3 函数。

首先计算了当前线程对应 log_probs 的各种索引，这个根据线程网格不难理解，根据索引将 log_probs 的值更新到 tmp_log_probs 中，
 注意这里每个线程处理元素的步长为 BLOCK_SIZE_ * BLOCKS_PER_BEAM_。

随后对 K 进行循环，在循环中首先对该线程处理的所有元素进行遍历，不断将数据 insert 到 partial 中，
 这样就得到了每个线程处理元素的最大值，然后再对 partial 进行块内规约，得到每个线程块内的最大值 total。
 在 tid == 0 的线程内把 total 更新到 topk_tmp_id_buf，再把 tmp_log_probs 中的值置为极小值，
 循环 K 次上述过程就得到每个线程块内的 topK，最终一行元素被处理成了 BLOCKS_PER_BEAM_ 个 topK，
 topk_tmp_val_buf 的形状为 [batch_size, beam_width, BLOCKS_PER_BEAM_, K]。
 在第二个 kernel 中我们需要将其缩小到 [batch_size, K]，下面来看一下代码。
 * */
template<typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage_1_opt3(
    const T* __restrict log_probs,
    T* tmp_log_probs,
    int* topk_tmp_id_buf,
    T* topk_tmp_val_buf,
    const int k,
    const int vocab_size
)
{
    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int row_id = bid / BLOCKS_PER_BEAM_; // row id for log_probs
    const int block_lane = bid % BLOCKS_PER_BEAM_; // block id for a beam 
    const int tmp_log_buf_index = row_id * vocab_size; 
    const int tmp_topk_buf_index = row_id * BLOCKS_PER_BEAM_ * k + block_lane * k;
    TopK_2<T> partial;

    for(int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size; elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
    {
        int index = elem_id + tmp_log_buf_index;
        tmp_log_probs[index] = log_probs[index]; 
    }


    for(int ite = 0; ite < k; ite++)
    {
        partial.init();
        #pragma unroll
        for(int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size; elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
        {
            int index = elem_id + tmp_log_buf_index;
            partial.insert(tmp_log_probs[index], index);
        }

        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

        if (tid == 0)
        {
            const int index = tmp_topk_buf_index + ite;
            topk_tmp_id_buf[index] = total.p;
            topk_tmp_val_buf[index] = total.u;
            tmp_log_probs[total.p] = -FLT_MAX;
        }
        __syncthreads();
    }
}

/**
 * @brief
 * // grid_size = batch_size
 * @tparam T
 * @tparam BLOCK_SIZE_
 * @tparam BLOCKS_PER_BEAM_
 * @param topk_tmp_id_buf               [batch_size, beam_width, BLOCKS_PER_BEAM_, K]
 * @param topk_tmp_val_buf
 * @param ids
 * @param k
 * @return __global__
topk_stage_2_opt3 的 grid_size 直接设置为 batch_size，block_size 设置为 BLOCK_SIZE_，
 也就是说，我们在一个 block 内求出 topK 就可以完成计算任务。

 计算思路和 topk_stage_2_opt3 大致相同，都是每次求最大值后把原值置小，循环 K 次即可。
 定义了一个共享内存变量 s_id 用来存储 topK，最终在 tid < k 的线程分别把 s_id 更新到 ids 完成计算。
总的来说，更新后的 topK kernel 计算思路更加清晰，便于理解，
 是一个较好的思路，但是笔者还是更推荐使用 Top-k 采样解码中的思路来计算 topK 问题，
 猜测这两种解码方式的代码不是同一个作者编写的，否则完全可以复用代码。
 */
template<typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage_2_opt3(
    const int* __restrict topk_tmp_id_buf,
    T* topk_tmp_val_buf,
    int* ids,
    const int k)
{
    const int size = k * k * BLOCKS_PER_BEAM_; 
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;

    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    extern __shared__ char array[];
    T *s_val = topk_tmp_val_buf + batch_id * size;
    int *s_id = (int*)(array);
    
    TopK_2<T> partial;

    for(int ite = 0; ite < k; ite++)
    {
        partial.init();
        #pragma unroll
        for(int i = tid; i < size; i+= BLOCK_SIZE_)
        {
            partial.insert(s_val[i], i);
        }
    
        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);
    
        if(tid == 0) 
        {
            s_id[ite] = total.p;
            s_val[total.p] = -FLT_MAX;
        }
        __syncthreads();
    }
    if(tid < k) ids[batch_id * k + tid] = topk_tmp_id_buf[batch_id * size + s_id[tid]];
}

template<typename T, int BLOCK_SIZE, int BLOCKS_PER_BEAM>
__global__ void topk_stage_1_opt2_general(
    const T* __restrict log_probs,
    T* tmp_log_probs,
    int* topk_tmp_id_buf,
    T* topk_tmp_val_buf,
    const int k,
    const int vocab_size
)
{
    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int row_id = bid / BLOCKS_PER_BEAM; // row id for log_probs
    const int block_lane = bid % BLOCKS_PER_BEAM; // block id for a beam 
    const int tmp_log_buf_index = row_id * vocab_size; 
    const int tmp_topk_buf_index = row_id * BLOCKS_PER_BEAM * k + block_lane * k;
    TopK_2<T> partial;

    for(int elem_id = tid + block_lane * BLOCK_SIZE; elem_id < vocab_size; elem_id += BLOCK_SIZE * BLOCKS_PER_BEAM)
    {
        int index = elem_id + tmp_log_buf_index;
        tmp_log_probs[index] = log_probs[index]; 
    }


    for(int ite = 0; ite < k; ite++)
    {
        partial.init();
        #pragma unroll
        for(int elem_id = tid + block_lane * BLOCK_SIZE; elem_id < vocab_size; elem_id += BLOCK_SIZE * BLOCKS_PER_BEAM)
        {
            int index = elem_id + tmp_log_buf_index;
            partial.insert(tmp_log_probs[index], index);
        }

        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

        if (tid == 0)
        {
            const int index = tmp_topk_buf_index + ite;
            topk_tmp_id_buf[index] = total.p;
            topk_tmp_val_buf[index] = total.u;
            tmp_log_probs[total.p] = -FLT_MAX;
        }
        __syncthreads();
    }
}

template<typename T, int BLOCK_SIZE, int BLOCKS_PER_BEAM>
__global__ void topk_stage_2_opt2_general(
    const int* __restrict topk_tmp_id_buf,
    T* topk_tmp_val_buf,
    int* ids,
    const int k)
{
    const int size = k * k * BLOCKS_PER_BEAM; 
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;

    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    extern __shared__ char array[];
    T *s_val = topk_tmp_val_buf + batch_id * size;
    int *s_id = (int*)(array);
    
    TopK_2<T> partial;

    for(int ite = 0; ite < k; ite++)
    {
        partial.init();
        #pragma unroll
        for(int i = tid; i < size; i+= BLOCK_SIZE)
        {
            partial.insert(s_val[i], i);
        }
    
        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);
    
        if(tid == 0) 
        {
            s_id[ite] = total.p;
            s_val[total.p] = -FLT_MAX;
        }
        __syncthreads();
    }
    if(tid < k) ids[batch_id * k + tid] = topk_tmp_id_buf[batch_id * size + s_id[tid]];
}

#define CASE_K_DIV(K,BLOCK_SIZE_1, BLOCK_SIZE_2) \
  case K: \
    beam_topK_kernel<T, K, BLOCK_SIZE_2><<<batch_size * beam_width, BLOCK_SIZE_2, 0, stream>>>(log_probs, \
        topk_tmp_id_buf, topk_tmp_val_buf, vocab_size, diversity_rate); \
    if (K < 10) \
      batch_topK_kernel<T, K, BLOCK_SIZE_1><<<batch_size, BLOCK_SIZE_1, 0, stream>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids); \
    else \
      batch_topK_kernel_v2<T, K, 32><<<batch_size, 32, 0, stream>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids); \
  break; \

#define CASE_K(K,BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_) \
  case K: \
    topk_stage_1_opt3<float, BLOCK_SIZE_1_, BLOCKS_PER_BEAM_><<<batch_size * K * BLOCKS_PER_BEAM_, BLOCK_SIZE_1_, 0, stream>>>( \
        log_probs, \
        temp_log_probs, \
        topk_tmp_id_buf, \
        topk_tmp_val_buf, \
        beam_width, vocab_size); \
    topk_stage_2_opt3<float, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_><<<batch_size, BLOCK_SIZE_2_, K * sizeof(int), stream>>>( \
        topk_tmp_id_buf, \
        topk_tmp_val_buf, \
        ids, \
        beam_width); \
  break; \

template <typename T>
void topK_kernelLauncher(T* log_probs,
                        T* temp_log_probs,
                        int* topk_tmp_id_buf,
                        T* topk_tmp_val_buf,
                        int* ids,
                        DecodingBeamsearchArguments args,
                        cudaStream_t stream)
{
    const int batch_size = args.batch_size_;
    const int beam_width = args.beam_width_;
    const int vocab_size = args.vocab_size_;
    const T diversity_rate = args.beam_search_diversity_rate_;
    if(diversity_rate == 0.0f)
    {
        switch(beam_width)
        {
            CASE_K(1,128,128,8);
            CASE_K(4,128,128,8);
            CASE_K(10,128,128,8);
            CASE_K(16,128,128,5);
            CASE_K(32,256,128,1);
            CASE_K(64,256,256,1);
            default:
                topk_stage_1_opt2_general<T, 128, 1><<<batch_size * beam_width * 1, 128, 0, stream>>>(
                    log_probs,
                    temp_log_probs,
                    topk_tmp_id_buf,
                    topk_tmp_val_buf,
                    beam_width, vocab_size);
                topk_stage_2_opt2_general<T, 128, 1><<<batch_size, 128, beam_width*beam_width*1*sizeof(float) + beam_width * sizeof(int), stream>>>(
                    topk_tmp_id_buf,
                    topk_tmp_val_buf,
                    ids,
                    beam_width);
                break;
        }
    }
    else
    {
        switch(beam_width)
        {
            CASE_K_DIV(1,256,256);
            CASE_K_DIV(4,256,256);
            CASE_K_DIV(16,256,64);
            CASE_K_DIV(64,256,64);
            default:
                printf("[ERROR] Topk kernel does not support beamwidth = %d \n", beam_width);
                exit(0);
                break;
        }
    }

}
#undef CASE_K
#undef CASE_K_DIV

template void topK_kernelLauncher<float>(float* log_probs,
                                          float* temp_log_probs,
                                          int* topk_tmp_id_buf,
                                          float* topk_tmp_val_buf,
                                          int* ids,
                                          DecodingBeamsearchArguments args,
                                          cudaStream_t stream);
/*
4.7.2.2 采样
前面介绍过采样原理，
 获取 TopK 之后，计算每个 word 的概率，然后在 TopK 中归一化，最后根据归一化后的概率采样。
 其实就是先 Softmax 后采样，我们来看一下源码。

 核函数中 grid_size 和 block_size 分别设置为 batch_size 和 candidate_num，当前线程就只处理对应一个元素，
 先根据索引从 topk_tmp_val_buf 中获取 TopK 中的最大值，然后让当前元素减去最大值然后求指数，再存入 topk_tmp_val_buf。
 在 0 号线程内循环求规约和，得到 sum，这时候其实已经可以开始采样了，没有必要非得归一化。
 源码中调用 cuda 随机数生成库的 API 从均匀分布中随机一个 0~1 之间的数再乘以 sum，得到一个 0~sum 之间的数 rand_num，
 要知道 TopK 中各元素是降序排列的，我可以把他当成 k 个相互连接的组合线段记作St（其中每个子线段记作Si），
 把 rand_num 当成一根长度为 rand_num 的线段记作Sr，并将其与St的最左侧对齐，
 那么Sr的右端点落在St的哪个子线段中就认为采样选中了哪个 word，
 笔者给出如下示意图。
   2105.png
随后根据采样选中的 word_id 对 sequence_length 和 finished_buf 进行更新，
 至此当前 step 的采样解码就完成了。
 * */
// Sampling kernels
template<typename T>
__global__ void sampling(int* topk_tmp_id_buf, 
                        T* topk_tmp_val_buf, 
                        int* ids, 
                        int* sequence_length, 
                        bool* finished_buf,
                        const int candidate_num, 
                        int random_num,
                        const int end_id,
                        const int vocab_size)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    __shared__ T sum;
    __shared__ T rand_num;

    if(tid < candidate_num)
    {
        T max_val = topk_tmp_val_buf[bid * candidate_num];
        topk_tmp_val_buf[bid * candidate_num + tid] = __expf(topk_tmp_val_buf[bid * candidate_num + tid] - max_val);
    }
    
    if(tid == 0)
    {
        sum = 0.0f;
        for(int i = 0; i < candidate_num; i++)
        {
            sum = sum + topk_tmp_val_buf[bid * candidate_num + i];
        }
        
        curandState_t local_state;
        curand_init((T)random_num, bid, 0, &local_state);
        rand_num = (T)curand_uniform(&local_state) * sum;

        ids[bid] = topk_tmp_id_buf[bid * candidate_num + candidate_num - 1] % vocab_size;
        for(int i = 0; i < candidate_num; i++)
        {
            rand_num = rand_num - topk_tmp_val_buf[bid * candidate_num + i];
            if(rand_num <= 0.0f){
                ids[bid] = topk_tmp_id_buf[bid * candidate_num + i] % vocab_size;
                break;
            }
        }

        sequence_length[bid] = finished_buf[bid] ? sequence_length[bid] : sequence_length[bid] + 1;
        finished_buf[bid] = ids[bid] == end_id ? 1 : 0;
    }
}

/*
4.7.2 topK_sampling_kernel_kernelLauncher
根据 topK_sampling_kernel_kernelLauncher 函数逻辑可以看出，
 采样过程由两个核函数完成：beam_topK_kernel 和 sampling。
 函数内部首先判断了 candidate_num 的值，貌似目前只支持 1、2、4 三种情况，这里源码为什么要用宏的模式，
 因为编译期要对模板进行实例化，要求 K(candidate_num) 在编译期就得确定，
 而源码中的 candidate_num 显然是一个运行期才确定的参数，所以只好牺牲编译期，
 多实例化几个模板（如 1、2、4，分别对应 1 个函数），
 等到运行期的时候匹配真实的 candidate_num，去执行对应的模板函数。
 * */
#define CASE_K(K) \
  case K : \
    beam_topK_kernel<T, K, block_size><<<batch_size, block_size, 0, stream>>>(log_probs, \
        topk_tmp_id_buf, topk_tmp_val_buf, vocab_size, 0.0f); \
  break; \

template <typename T>
void topK_sampling_kernel_kernelLauncher(T* log_probs,
                                        int* topk_tmp_id_buf,
                                        T* topk_tmp_val_buf,
                                        int* ids,
                                        int* sequence_length,
                                        bool* finished_buf,
                                        int random_num,
                                        DecodingSamplingArguments args,
                                        cudaStream_t stream)
{
    const int batch_size = args.batch_size_;
    const int vocab_size = args.vocab_size_;
    const int candidate_num = args.candidate_num_;
    const int end_id = args.end_id_;
    const int block_size = 256;
    switch(candidate_num)
    {
        CASE_K(1);
        CASE_K(2);
        CASE_K(4);
        default:
            printf("[ERROR] Topk kernel does not support candidate_num = %d \n", candidate_num);
            exit(0);
            break;
    }
    sampling<T> <<< batch_size, candidate_num, 0, stream>>> (topk_tmp_id_buf, topk_tmp_val_buf, 
                                                            ids, sequence_length, finished_buf,
                                                            candidate_num, random_num, end_id, vocab_size);
}

template void topK_sampling_kernel_kernelLauncher(float* log_probs,
                                                int* topk_tmp_id_buf,
                                                float* topk_tmp_val_buf,
                                                int* ids,
                                                int* sequence_length,
                                                bool* finished_buf,
                                                int random_num,
                                                DecodingSamplingArguments args,
                                                cudaStream_t stream);

__global__ void init_topp_id_val(int* topp_id_val_buf, 
                                int* topp_offset_buf,
                                const int batch_size,
                                const int vocab_size)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if(bid == 0)
    {
        for(int i = tid; i < batch_size + 1; i+= blockDim.x)
        {
            topp_offset_buf[i] = i * vocab_size;
        }
    }

    while(tid < vocab_size)
    {
        topp_id_val_buf[bid * vocab_size + tid] = tid;
        tid += blockDim.x;
    }
}


void init_topp_id_val_kernel_kernelLauncher(int* topp_id_val_buf,
                                            int* topp_offset_buf,
                                            const int batch_size,
                                            const int vocab_size,
                                            cudaStream_t stream)
{
    init_topp_id_val<<<batch_size, 512, 0, stream>>>(topp_id_val_buf,
                                                    topp_offset_buf,
                                                    batch_size,
                                                    vocab_size);
}

/*4.8.2.2 采样
根据采样原理，拿到排序结果后，我们需要根据 p 值进行候选集的确定，然后在候选集的内部进行采样。
源码中提供了核函数 top_p_sampling 进行采样工作，
 grid_size 设置为 1，block_size 设置为 bacth_size，
 block 内完成计算，每个线程承担一个样本的计算任务。


 采样过程和前面 Top-k 的过程大同小异，
 有一点区别就是，不用真的先确定候选集再进行采样，可以直接一步进行。
 先使用 cuda 随机数生成库的 API 从均匀分布中随机一个 0~1 之间的数再乘以 p 值（probability_threshold_），
 这其实就相当于把采样的概率点缩放到了 p 值范围内，

 然后遍历 sorted_log_probs 判断采样点落在哪个区间，就选中了哪个 word，示意图如下：  2106.png

采样完成后把采样结果更新到 ids，然后对 sequence_length 和 finished_buf 进行更新，
 至此，当前 step 的 Top-p 采样解码就完成了。
 */
// Sampling kernels
template<typename T>
__global__ void top_p_sampling(T* sorted_log_probs, 
                                int* sorted_id_vals,
                                int* ids,
                                int* sequence_length,
                                bool* finished_buf,
                                const int vocab_size,
                                const int random_num,
                                const float prob_threshold, 
                                const int end_id)
{
    int tid = threadIdx.x;
    curandState_t local_state;
    curand_init((T)random_num, tid, 0, &local_state);
    T rand_num = (T)curand_uniform(&local_state) * prob_threshold;
    ids[tid] = sorted_id_vals[vocab_size - 1];

    for(int i = tid * vocab_size; i < tid * vocab_size + vocab_size; i++)
    {
        rand_num = rand_num - sorted_log_probs[i];
        if(rand_num <= 0)
        {
            ids[tid] = sorted_id_vals[i];
            break;
        }
    }

    sequence_length[tid] = finished_buf[tid] ? sequence_length[tid] : sequence_length[tid] + 1;
    finished_buf[tid] = ids[tid] == end_id ? 1 : 0;
}

template <typename T>
__global__ void sort_kernel(const T* log_probs, 
                            const int* id_vals,
                            T* sorted_log_probs,
                            int* sorted_id_vals,
                            const int vocab_size)
{
    typedef cub::BlockRadixSort<T, 256, 32, int> BlockRadixSort;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;
    // Obtain a segment of consecutive items that are blocked across threads
    T thread_keys[32];
    int thread_values[32];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    for(int i = 0; i < 32; i++)
    {
        int index = tid + 256 * i + bid * vocab_size;
        thread_keys[i] = log_probs[index];
        thread_values[i] = id_vals[index];
    }
    BlockRadixSort(temp_storage).SortDescending(thread_keys, thread_values);

    for(int i = 0; i < 32; i++)
    {
        int index = tid + 256 * i + bid * vocab_size;
        sorted_log_probs[index] = thread_keys[i];
        sorted_id_vals[index] = thread_values[i];
    }
}

/*
4.8 Top-p 采样解码
Top-p 采样与 Top_k 采样有所不同，不再从固定 k 个候选词中采样，而是根据生成概率从高到低在词表上选择累积概率恰好超过p的候选 word 作为采样集合，
 从这个集合中采样。所以采样前必须先计算每个 word 对应的概率并进行排序，即要计算 Softmax，再按概率值排序。

4.8.1 update_logits_without_log
源码中使用 update_logits_kernel_without_log 核函数来计算 Softmax，顺带加一个上一步没进行的 add bias 操作。
 这个核函数比较简单是个老生常谈的 Softmax kernel，只需要注意一点，
 计算完 softmax 后不要取对数即可，具体计算逻辑笔者就不啰嗦了，读者有兴趣可以看笔者前面的文章。

4.8.2 topP_sampling_kernel_kernelLauncher
Softmax 后拿到词表内每个 word 的概率，在进行采样前还要进行排序。

4.8.2.1 排序
这里排序是个大工程，因为 vocab_size 通常会很大，源码中使用了 cub 库中的 API 进行排序。


下面我们对 cub::DeviceSegmentedRadixSort::SortPairsDescending 函数的主要参数进行介绍：

    d_temp_storage：设备可以访问的临时内存，当设置为 NULL 时，所需的分配大小将写入 temp_storage_bytes，并且不执行任何工作。
    所以在真正执行函数前，我们需要先传一下 NULL 获取 temp_storage_bytes 然后再开始真正的执行排序

    temp_storage_bytes：临时内存的大小

    d_keys_in：指向排序过程中的比较依据，也就是说排序是根据这个指针指向的数据的来进行的，这里我们将它设置为概率值 log_probs

    d_keys_out：排序后的输出，这里我们用 sorted_log_probs 来接收

    d_values_in：与 key 一一对应，这里我们把他设置为概率值对应的索引 id_vals，其实就是 word_id

    d_values_out：排序后的输出，这里我们用 sorted_id_vals 来接收

    num_items：待排序的元素数目，这里应该是 batch_size * vocab_size

    num_segments：待排序的批次，也就是分为多少个组，这里是对每个样本单独排序，所以取 batch_size

    d_begin_offsets：每个分组的起始索引，为了方便 end_offset 的设置，这个变量对应的元素数量通常是 num_segments + 1，
    前面 num_segments 个元素都是分组的起始索引，最后一个元素设为 num_items，这里我们设置为 topp_offset_buf，前面已经完成初始化

    d_end_offsets：每个分组的结束索引，注意这里是“顾头不顾尾”的模式，
    所以直接可以设置为 d_begin_offsets + 1，这里我们设置为 topp_offset_buf + 1

 参数意义介绍完毕后，其实函数的作用也就清晰了，就是分组降序排序，每一组对应 batch 内的一个样本，也就是 vocab_size 个元素，
 batch 内每个样本下排序后的待采样 word 的概率值 sorted_log_probs和 sorted_id_vals。
 * */
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
                                        cudaStream_t stream)
{
    // sort_kernel<<<batch_size, 256, 0, stream>>>(log_probs, 
    //                                             id_vals,
    //                                             sorted_log_probs,
    //                                             sorted_id_vals,
    //                                             vocab_size);
    cub::DeviceSegmentedRadixSort::SortPairsDescending(temp_storage, 
                                                        args.temp_storage_size_,
                                                        log_probs, 
                                                        sorted_log_probs,
                                                        id_vals, 
                                                        sorted_id_vals, 
                                                        args.vocab_size_ * args.batch_size_,
                                                        args.batch_size_, 
                                                        topp_offset_buf, topp_offset_buf + 1);
                                                        
    
    top_p_sampling<<<1, args.batch_size_, 0, stream>>>(sorted_log_probs, 
                                                        sorted_id_vals,
                                                        output_ids + (step - 1) * args.batch_size_,
                                                        sequence_length,
                                                        finished_buf,
                                                        args.vocab_size_, 
                                                        step,
                                                        args.probability_threshold_,
                                                        args.end_id_);
}

template void topP_sampling_kernel_kernelLauncher(const float* log_probs,
                                        const int* id_vals,
                                        float* sorted_log_probs,
                                        int* sorted_id_vals, 
                                        int* topp_offset_buf,
                                        void* temp_storage,
                                        bool* finished_buf,
                                        int step,
                                        DecodingSamplingArguments args,
                                        int* output_ids, 
                                        int* sequence_length, 
                                        cudaStream_t stream);

} // end of namespace fastertransformer