#pragma once
#include "painter_model.h"
#include <cstdint>
#include <string>

namespace pp {
struct LoadedCheckpoint {
  StateDict state;
  std::string state_key;
  uint64_t source_bytes=0;
};
// The compact metadata index is made by the already used restricted native Rust
// PTH parser. LibTorch's JIT unpickler intentionally discards OrderedDict and
// cannot read Python's model_state_dict directly. No tensor-file conversion or
// Python runtime is involved; storages are read directly from the original ZIP.
LoadedCheckpoint load_checkpoint(const std::string& model_utf8,
                                 const std::string& index_utf8);
}
