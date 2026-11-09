#include "checkpoint_loader.h"
#include <caffe2/serialize/inline_container.h>
#include <c10/core/InferenceMode.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <set>

namespace pp {
namespace {
uint32_t u32(std::istream& in) {
  unsigned char b[4];in.read(reinterpret_cast<char*>(b),4);
  TORCH_CHECK(in.good(),"truncated weight index");
  return uint32_t(b[0])|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24);
}
uint64_t u64(std::istream& in) {
  const auto lo=u32(in),hi=u32(in);return uint64_t(lo)|(uint64_t(hi)<<32);
}
std::string string(std::istream& in) {
  const auto n=u32(in);TORCH_CHECK(n>0&&n<=65536,"invalid weight-index string length");
  std::string value(n,'\0');in.read(value.data(),n);TORCH_CHECK(in.good(),"truncated weight-index string");
  TORCH_CHECK(value.find('\0')==std::string::npos,"NUL in weight-index string");return value;
}
at::ScalarType type(uint32_t code) {
  switch(code){case 0:return at::kBFloat16;case 1:return at::kFloat;case 2:return at::kHalf;
    case 3:return at::kDouble;case 4:return at::kLong;case 5:return at::kByte;default:TORCH_CHECK(false,"unsupported weight-index dtype");}
}
std::string normalize_name(std::string name) {
  for(int n=0;n<8;++n){
    if(name.rfind("module.",0)==0)name.erase(0,7);
    else if(name.rfind("_orig_mod.",0)==0)name.erase(0,10);
    else return name;
  }
  TORCH_CHECK(false,"excessive wrapper prefix in weight name");
}
}

LoadedCheckpoint load_checkpoint(const std::string& model_utf8,const std::string& index_utf8) {
  c10::InferenceMode inference;
  TORCH_CHECK(!model_utf8.empty()&&!index_utf8.empty(),"model and native weight index paths are required");
  const auto path=std::filesystem::u8path(model_utf8);
  const auto actual_bytes=std::filesystem::file_size(path);
  std::ifstream index(std::filesystem::u8path(index_utf8),std::ios::binary);
  TORCH_CHECK(index.good(),"cannot open native weight metadata index");
  char magic[8];index.read(magic,8);
  TORCH_CHECK(index.good()&&std::string(magic,8)=="PPATENI1","weight metadata index magic differs");
  TORCH_CHECK(u32(index)==1,"unsupported weight metadata index version");
  const auto count=u32(index);const auto expected_bytes=u64(index);
  TORCH_CHECK(count>0&&count<=100000,"invalid tensor count in weight index");
  TORCH_CHECK(actual_bytes==expected_bytes,"selected model file size differs from its native metadata index");
  std::ifstream model(path,std::ios::binary);
  TORCH_CHECK(model.good(),"cannot open selected original weight file");
  caffe2::serialize::PyTorchStreamReader reader(&model);
  if(reader.hasRecord("byteorder")){
    auto [raw,bytes]=reader.getRecord("byteorder");
    TORCH_CHECK(std::string(static_cast<const char*>(raw.get()),bytes)=="little","only little-endian PyTorch storages are supported");
  }
  LoadedCheckpoint result;result.source_bytes=actual_bytes;result.state_key="native_restricted_pth_index";
  for(uint32_t tensor=0;tensor<count;++tensor){
    auto name=normalize_name(string(index));auto record=string(index);
    const auto dtype=type(u32(index));const auto rank=u32(index);
    const auto storage_elements=u64(index);const auto byte_offset=u64(index);
    TORCH_CHECK(rank<=16&&storage_elements<=uint64_t(std::numeric_limits<int64_t>::max()),"invalid weight-index geometry");
    std::vector<int64_t> sizes(rank),strides(rank);
    for(auto& n:sizes){const auto v=u64(index);TORCH_CHECK(v<=uint64_t(std::numeric_limits<int64_t>::max()),"weight dimension overflows");n=int64_t(v);}
    for(auto& n:strides){const auto v=u64(index);TORCH_CHECK(v<=uint64_t(std::numeric_limits<int64_t>::max()),"weight stride overflows");n=int64_t(v);}
    TORCH_CHECK(!record.empty()&&record.find("..") == std::string::npos&&record.front()!='/'&&record.find('\\')==std::string::npos,"invalid ZIP storage record");
    if(!reader.hasRecord(record)){
      const auto slash=record.find('/');TORCH_CHECK(slash!=std::string::npos,"storage record missing from original archive");
      record.erase(0,slash+1);
    }
    TORCH_CHECK(reader.hasRecord(record),"storage record missing: ",record);
    auto [data,bytes]=reader.getRecord(record);
    const auto element_size=c10::elementSize(dtype);
    TORCH_CHECK(storage_elements<=std::numeric_limits<uint64_t>::max()/element_size
      && bytes==storage_elements*element_size&&byte_offset%element_size==0,"storage/index size mismatch: ",name);
    uint64_t last=byte_offset/element_size;
    for(uint32_t i=0;i<rank;++i){
      TORCH_CHECK(sizes[i]>0,"zero-sized model parameter is unsupported");
      TORCH_CHECK(strides[i]==0||uint64_t(sizes[i]-1)<=std::numeric_limits<uint64_t>::max()/uint64_t(strides[i]),"storage span multiplication overflow");
      const uint64_t extent=uint64_t(sizes[i]-1)*uint64_t(strides[i]);
      TORCH_CHECK(last<=std::numeric_limits<uint64_t>::max()-extent,"storage span overflow");last+=extent;
    }
    TORCH_CHECK(last<storage_elements,"tensor view extends past its original storage: ",name);
    auto owner=std::make_shared<c10::DataPtr>(std::move(data));
    auto storage=at::from_blob(owner->get(),{int64_t(storage_elements)},
      [owner](void*){},at::TensorOptions().device(at::kCPU).dtype(dtype));
    auto value=storage.as_strided(sizes,strides,int64_t(byte_offset/element_size));
    TORCH_CHECK(result.state.emplace(name,std::move(value)).second,"duplicate normalized model parameter: ",name);
  }
  TORCH_CHECK(index.peek()==std::char_traits<char>::eof(),"unexpected trailing bytes in weight index");
  TORCH_CHECK(result.state.count("wte.weight")&&result.state.count("text_wte.weight"),"not the primitive painter checkpoint");
  return result;
}
}
