#pragma once
#include <ATen/ATen.h>
#include <memory>
#include <string>

namespace pp {
// Native kernels read packed historical K/V in-place. Only tile partials and
// the current query's BF16 scratch exist outside the INT8 persistent history.
class Int8KvKernels {
 public:
  Int8KvKernels();
  ~Int8KvKernels();
  void prepare(at::Device device,int64_t head_width);
  bool ready() const noexcept;
  at::Tensor attention(const at::Tensor& query,const at::Tensor& keys,
    const at::Tensor& values,const at::Tensor& key_scale,const at::Tensor& value_scale,
    const at::Tensor& scratch_key,const at::Tensor& scratch_value,
    const at::Tensor& valid,const at::Tensor& lengths,int64_t bucket_end) const;
  void validate(at::Device device) const;
  void commit(const at::Tensor& packed,const at::Tensor& scales,const at::Tensor& scratch,const at::Tensor& lengths)const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
void quantize_kv_chunk(const at::Tensor& source,const at::Tensor& packed,const at::Tensor& scales);
at::Tensor dequantize_kv_chunk(const at::Tensor& packed,const at::Tensor& scales,at::ScalarType dtype);
}
