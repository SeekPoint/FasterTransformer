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
 * BeamSearch OpenNMT
 **/

#pragma once

#include <cuda_runtime.h>
#include "fastertransformer/allocator.h"
#include "fastertransformer/cuda/cuda_kernels.h"
#include "fastertransformer/cuda/open_attention.h"
#include "fastertransformer/cuda/decoding_kernel_check.h"

namespace fastertransformer
{
/*
4.10 BeamSearch_OpenNMT
在 Decoding 过程中，模型的输出是一个 step 一个 step 依次获得的，而且前面 step 的结果还会影响后面 step 的结果。
 也就是说，每一个 step，模型给出的都是基于历史生成结果的条件概率。为了生成完整的序列，需要一个额外动作来融合模型多个 step 的输出，
 而且使得最终得到的序列的每一步条件概率连乘起来最大。

在生成任务中，每一个 step 可能的输出种类称为字典大小（vocab_size），进行 T 步随机的生成可能获得的结果总共有vocab_size^T种。
拿中文文本生成来说，vocab_size 的值大约是 5000-6000，即常用汉字的个数。
 在如此大的基数下，遍历整个生成空间寻找最佳序列是不现实的。

基于上述背景，我们首先想到的策略是逐帧取最大值，也就是贪心搜索，即每一个时间步都取出一个条件概率最大的输出，
再将从开始到当前步的结果作为输入去获得下一个时间步的输出，直到模型给出生成结束的标志。
优点很明显，这样做将原来指数级别的求解空间直接压缩到了与长度线性相关的大小。
缺点同样很明显，丢弃了绝大多数的可能解，这种关注当下的策略无法保证最终得到的序列概率是最优的。

beam search（集束搜索）是对贪心搜索一个改进。思路也很简单，在每一个 step，不再只保留当前分数最高的 1 个输出，
 而是保留 beam_width 个。当 beam_width = 1 时集束搜索就退化成了贪心搜索。

下图是一个实际的例子，每个时间步有 ABCDE 共 5 种可能的输出，
 即，图中的 beam_width = 2，也就是说每个 step 都会保留到当前步为止条件概率最优的 2 个序列。

         211.png

可以发现，beam search 在每一步需要考察的候选人数量是贪心搜索的 beam_width 倍，
 因此是一种牺牲时间换效果的折中方法。
 源码中 BeamSearch_OpenNMT 函数内部分别调用了 4 个函数，下面进行逐一讲解。


 * @brief beam search
 *
 * @tparam T
 * @param log_probs           logits: [batch_size, beam_width, vocab_size]
 * @param cum_log_probs               [batch_size, beam_width]
 * @param finished                    [batch_size, beam_width]
 * @param key_cache               2 * [batch_size, beam_width, seq_len, hidden_units]
 * @param value_cache
 * @param parent_ids
 * @param sequence_length
 * @param word_ids                    [batch_size, beam_width]
 * @param ids
 * @param output_ids
 * @param batch_size
 * @param beam_width
 * @param vocab_size
 * @param hidden_dim
 * @param step
 * @param cache_size
 * @param decoder_layers
 * @param stream
 * @param end_id
 * @param finished_count

 * */
template <typename T>
void BeamSearch_OpenNMT(
    float *log_probs, float *cum_log_probs, bool *finished,
    T **key_cache, T **value_cache,
    int *parent_ids,
    int *sequence_length,
    int *word_ids,
    int *ids,
    int *output_ids,
    const int batch_size, const int beam_width,
    const int vocab_size, const int hidden_dim, const int step,
    const int cache_size, const int decoder_layers, cudaStream_t stream,
    const int end_id, 
    int *finished_count)
{
#ifdef NDEBUG
  /* adding cum_log_probs to log_probs */
  broadcast_kernelLauncher(log_probs, cum_log_probs, batch_size, beam_width, vocab_size, stream);
#else
  broadcast_kernelLauncher(log_probs, cum_log_probs, batch_size, beam_width, vocab_size, stream);
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());

  /*
    User can check the broadcast_kernel by broadcast_kernel_check.
    broadcast_kernel_check will compare the results of GPU and CPU.
    Note that broadcast_kernel_check contains broadcast_kernelLauncher and uses do not need to call it again. 
  */
  // broadcast_kernel_check(log_probs, cum_log_probs, batch_size, beam_width, vocab_size, stream);
#endif

#ifdef NDEBUG
  /*Use two round kernels to pick the topK values for each batch */
  topK(log_probs, ids, batch_size, beam_width, vocab_size, stream);
#else
  topK(log_probs, ids, batch_size, beam_width, vocab_size, stream);
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());

  /*
    User can check the topK by topK_check.
    topK_check will compare the results of GPU and CPU.
    Note that topK_check contains topK and uses do not need to call it again. 
  */
  // topK_kernel_check(log_probs, ids, batch_size, beam_width, vocab_size, stream);
#endif

#ifdef NDEBUG
  update(log_probs, cum_log_probs, ids, finished, 
        parent_ids, sequence_length, word_ids, output_ids,
        batch_size, beam_width, vocab_size, stream, 
        end_id, finished_count);
#else
  update(log_probs, cum_log_probs, ids, finished, 
        parent_ids, sequence_length, word_ids, output_ids,
        batch_size, beam_width, vocab_size, stream, 
        end_id, finished_count);
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());

  /*
    User can check the update by update_kernel_check.
    update_kernel_check will compare the results of GPU and CPU.
    Note that update_kernel_check contains update and uses do not need to call it again. 
  */
  // update_kernel_check(log_probs, cum_log_probs, ids, finished, parent_ids, sequence_length, word_ids, output_ids,
  //                     batch_size, beam_width, vocab_size, stream, end_id, finished_count);
#endif

#ifdef NDEBUG
  update_KV_cache<T>(key_cache, value_cache, parent_ids, batch_size, 
                    beam_width, hidden_dim, step, cache_size, 
                    decoder_layers, stream);
#else
  update_KV_cache<T>(key_cache, value_cache, parent_ids, batch_size, 
                    beam_width, hidden_dim, step, cache_size, 
                    decoder_layers, stream);
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());

  /*
    User can check the update_KV_cache by update_KV_cache_kernel_check.
    update_KV_cache_kernel_check will compare the results of GPU and CPU.
    Note that update_KV_cache_kernel_check contains update_KV_cache and uses do not need to call it again. 
  */
  // update_KV_cache_kernel_check(key_cache, value_cache, parent_ids, batch_size, beam_width, hidden_dim, step, cache_size, decoder_layers, stream);
#endif
}

} // namespace fastertransformer
