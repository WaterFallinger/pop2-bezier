#pragma once
#include <ATen/ATen.h>
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "kv_int8.h"

namespace pp {
using StateDict = std::unordered_map<std::string, at::Tensor>;
constexpr int64_t kContext = 1024;
constexpr int64_t kAttributeVocab = 7054;
constexpr int64_t kPrimitiveVocab = 7069;
constexpr int64_t kTextVocab = 1500;
constexpr std::array<int64_t,9> kOffsets{0,1024,2048,2318,4366,6414,6670,6798,6926};
constexpr std::array<int64_t,9> kBins{1024,1024,270,2048,2048,256,128,128,128};
constexpr std::array<int64_t,9> kPartial{7054,7058,7059,7060,7061,7062,7063,7064,7065};

enum class AttentionMode { Math = 0, Efficient = 1 };
struct Scope {
  int64_t offset = 0;
  int64_t count = kAttributeVocab;
  bool include_termination = true;
  static Scope field(int64_t field);
  static Scope next();
  static Scope all();
};
struct Prediction {
  at::Tensor attributes;   // F32 [physical_rows, projected_count]
  at::Tensor termination;  // F32 [physical_rows,2], undefined for field-only
  int64_t attribute_offset = 0;
};
struct Prefix {
  std::vector<int64_t> text_ids;
  std::vector<std::array<int64_t,9>> records;
  std::vector<uint8_t> text_valid; // empty means all supplied text positions valid
  int64_t position_offset = 0;     // compact empty text retains aligned RoPE positions
};
struct Cache {
  std::vector<at::Tensor> keys, values; // BF16/F32 [physical,H,capacity,D]
  bool int8=false;
  std::vector<at::Tensor> key_scales,value_scales; // FP32 [physical,H,capacity]
  std::vector<at::Tensor> scratch_keys,scratch_values; // BF16 [physical,H,1,D]
  at::Tensor packed_int8,packed_scales,packed_scratch; // cross-layer fused commit
  std::shared_ptr<Int8KvKernels> int8_kernels;
  at::Tensor valid;                    // Bool [physical,capacity]
  at::Tensor positions;                // Long [capacity], fixed 0..capacity-1
  at::Tensor lengths;                  // Long [physical], next physical write slots
  at::Tensor absolute_positions;       // Long [physical], next RoPE positions
  int64_t batch = 0;
  int64_t capacity = 0;
};
struct Architecture {
  int64_t layers=0, width=0, heads=0, head_width=0, ffn_hidden=0;
};

class Model {
 public:
  Model(const StateDict& state, at::Device device, at::ScalarType dtype=at::kBFloat16);
  const Architecture& architecture() const noexcept { return architecture_; }
  at::Device device() const noexcept { return device_; }
  at::ScalarType dtype() const noexcept { return dtype_; }
  void prepare_int8_kv();
  void configure_int8_kv(bool enabled);
  bool int8_kv_enabled()const noexcept{return int8_kv_enabled_;}
  bool int8_kv_available()const noexcept{return int8_kernels_->ready();}
  Cache make_cache(int64_t physical_rows, int64_t capacity=kContext) const;
  Prediction prefill_rows(Cache& cache, const std::vector<Prefix>& prefixes,
      Scope scope=Scope::next(), int64_t chunk_size=64,
      AttentionMode mode=AttentionMode::Efficient) const;
  Prediction append_prefix(Cache& cache,int64_t retained,const Prefix& suffix,
      int64_t chunk_size=64,AttentionMode mode=AttentionMode::Efficient)const;
  // Scratch-only Q1 query. All indexing and valid masks remain on GPU and
  // shapes remain fixed within a bucket. No GPU scalar is read by the host.
  Prediction decode(const at::Tensor& records, Cache& cache, Scope scope,
      int64_t bucket_end, AttentionMode mode=AttentionMode::Efficient) const;
  static void advance_cache(Cache& cache);

 private:
  struct Block {
    at::Tensor norm1, qkv, q_norm, k_norm, projection, norm2, gate_up, down;
  };
  Architecture architecture_;
  at::Device device_;
  at::ScalarType dtype_;
  at::Tensor embedding_, text_embedding_, final_norm_, termination_weight_, termination_bias_;
  at::Tensor inverse_frequency_;
  std::vector<Block> blocks_;
  std::shared_ptr<Int8KvKernels> int8_kernels_=std::make_shared<Int8KvKernels>();
  bool int8_kv_enabled_=false;
  at::Tensor primitive_embedding(const at::Tensor& records) const;
  at::Tensor rms(const at::Tensor& x, const at::Tensor& weight) const;
  std::pair<at::Tensor,at::Tensor> rope_factors(const at::Tensor& absolute) const;
  std::pair<at::Tensor,at::Tensor> rope(const at::Tensor& q,
      const at::Tensor& k, const at::Tensor& cosine, const at::Tensor& sine) const;
  at::Tensor attention(const at::Tensor& q, const at::Tensor& k,
      const at::Tensor& v, const at::Tensor& bias, AttentionMode mode) const;
  at::Tensor feed_forward(const at::Tensor& x, const Block& block) const;
  Prediction project(const at::Tensor& hidden, Scope scope) const;
  at::Tensor prefill_row(Cache& cache, int64_t row, const Prefix& prefix,
      int64_t chunk_size, AttentionMode mode,int64_t retained=0) const;
  void validate_cache(const Cache& cache) const;
};
} // namespace pp
