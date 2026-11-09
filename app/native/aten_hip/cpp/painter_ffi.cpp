#include "painter_ffi.h"
#include "painter_engine.h"
#include <c10/core/InferenceMode.h>
#include <cstring>
#include <atomic>
#include <memory>
#include <string>

struct PpAtenModel { std::shared_ptr<pp::Engine> engine; };
struct PpAtenCache { pp::CacheSnapshot snapshot; };
namespace {
thread_local std::string last_error;
std::atomic<bool> process_capture_state_uncertain{false};
void allocation_ready(){
  TORCH_CHECK(!process_capture_state_uncertain.load(std::memory_order_acquire),
    "EXECUTION_UNCERTAIN: CAPTURE_STATE_UNCERTAIN: 请关闭并重新启动应用；仅重新加载权重无法修复已失败的框架捕获状态。已保留画布，未继续执行模型。");
}
int32_t error_code(const std::exception& error){
  last_error=error.what();if(last_error.size()>6000)last_error.resize(6000);
  if(last_error.find("CAPTURE_STATE_UNCERTAIN")!=std::string::npos)
    process_capture_state_uncertain.store(true,std::memory_order_release);
  if(pp::execution_uncertain(last_error))return 3;
  if(last_error.find("MODEL_NONFINITE")!=std::string::npos)return 2;
  return 1;
}
template<class Operation>int32_t call(Operation&& operation){
  last_error.clear();try{operation();return 0;}catch(const std::exception&error){return error_code(error);}
  catch(...){last_error="EXECUTION_UNCERTAIN: unrecognized native exception";return 3;}
}
}
extern "C" {
uint32_t pp_aten_abi_version(void){return 1;}
const char* pp_aten_last_error(void){return last_error.c_str();}
int32_t pp_aten_load(const char* path,const PpAtenLoadOptions* options,PpAtenModel** out){
  if(out)*out=nullptr;
  return call([&]{TORCH_CHECK(path&&options&&out&&options->struct_size>=sizeof(PpAtenLoadOptions)
      &&options->index_utf8,"invalid native load options or missing native weight metadata index");
    allocation_ready();
    auto model=std::make_unique<PpAtenModel>();
    model->engine=std::make_shared<pp::Engine>(path,options->index_utf8,options->device_ordinal,options->prefer_efficient_attention!=0);
    *out=model.release();});
}
const char* pp_aten_model_info(const PpAtenModel* model){
  return model&&model->engine?model->engine->info().c_str():nullptr;
}
int32_t pp_aten_configure(PpAtenModel* model,uint32_t efficient_prefill){
  return call([&]{allocation_ready();TORCH_CHECK(model&&model->engine,"missing native model");model->engine->configure(efficient_prefill!=0);});
}
int32_t pp_aten_begin_request(PpAtenModel* model){
  return call([&]{allocation_ready();TORCH_CHECK(model&&model->engine,"missing native model");model->engine->begin_request();});
}
int32_t pp_aten_configure_kv(PpAtenModel* model,uint32_t preference){
  return call([&]{allocation_ready();TORCH_CHECK(model&&model->engine,"missing native model");model->engine->configure_kv(preference);});
}
int32_t pp_aten_prepare(PpAtenModel* model){
  return call([&]{allocation_ready();TORCH_CHECK(model&&model->engine,"missing native model");model->engine->prepare_optional();});
}
int32_t pp_aten_prepare_graph(PpAtenModel* model){
  return call([&]{allocation_ready();TORCH_CHECK(model&&model->engine,"missing native model");model->engine->prepare_graph();});
}
int32_t pp_aten_probe_auto(PpAtenModel* model,const PpAtenAutoProbeOptions* options){
  return call([&]{allocation_ready();TORCH_CHECK(model&&model->engine&&options,"missing native model or adaptive options");model->engine->probe_auto(*options);});
}
const char* pp_aten_probe_info(const PpAtenModel* model){return model&&model->engine?model->engine->probe_info().c_str():nullptr;}
void pp_aten_model_release(PpAtenModel* model){delete model;}
int32_t pp_aten_prefill(PpAtenModel* model,const uint32_t* records,size_t record_count,
    const uint32_t* text,size_t text_count,const uint8_t* mask,size_t offset,
    PpAtenCache** out,float* terms){
  if(out)*out=nullptr;
  return call([&]{TORCH_CHECK(model&&model->engine&&records&&text&&out&&terms
      &&record_count>0&&record_count<1024&&text_count>0&&text_count<=224&&offset<1024,"invalid native prefill inputs");
    allocation_ready();
    pp::Prefix p;p.position_offset=int64_t(offset);p.text_ids.assign(text,text+text_count);
    if(mask)p.text_valid.assign(mask,mask+text_count);
    p.records.resize(record_count);for(size_t row=0;row<record_count;++row)for(size_t f=0;f<9;++f)p.records[row][f]=records[row*9+f];
    auto cache=std::make_unique<PpAtenCache>();cache->snapshot=model->engine->prefill(p,terms);*out=cache.release();});
}
int32_t pp_aten_cache_clone(const PpAtenCache* cache,PpAtenCache** out){
  if(out)*out=nullptr;return call([&]{TORCH_CHECK(cache&&cache->snapshot.owner&&out,"invalid cache clone input");
    auto copy=std::make_unique<PpAtenCache>();copy->snapshot=cache->snapshot;*out=copy.release();});
}
int32_t pp_aten_extend(PpAtenModel* model,const PpAtenCache* source,size_t retained,
    const uint32_t* records,size_t count,PpAtenCache** out,float* terms){
  if(out)*out=nullptr;return call([&]{TORCH_CHECK(model&&model->engine&&source&&records&&out&&terms
    &&retained<=size_t(pp::kContext)&&count>0&&count<=size_t(pp::kContext),"invalid incremental prefill input");
    allocation_ready();std::vector<std::array<int64_t,9>> suffix(count);
    for(size_t row=0;row<count;++row)for(size_t field=0;field<9;++field)suffix[row][field]=records[row*9+field];
    auto cache=std::make_unique<PpAtenCache>();cache->snapshot=model->engine->extend(source->snapshot,int64_t(retained),suffix,terms);*out=cache.release();});
}
int32_t pp_aten_cache_info(const PpAtenCache* cache,size_t* length,size_t* absolute){
  return call([&]{TORCH_CHECK(cache&&cache->snapshot.owner&&length&&absolute,"invalid cache metadata input");
    TORCH_CHECK(!cache->snapshot.owner->poisoned,"EXECUTION_UNCERTAIN: cache has failed state");
    *length=size_t(cache->snapshot.length);*absolute=size_t(cache->snapshot.absolute);});
}
void pp_aten_cache_release(PpAtenCache* cache){delete cache;}
int32_t pp_aten_frame(PpAtenModel* model,const PpAtenCache* const* caches,
    const PpAtenFrameOptions* options,const uint32_t* bounds,const float* uniforms,
    uint32_t* tokens,float* terms,uint32_t* valid,PpAtenCache** out){
  return call([&]{TORCH_CHECK(model&&model->engine&&caches&&options&&bounds&&uniforms&&tokens&&terms&&valid&&out
      &&options->struct_size>=sizeof(PpAtenFrameOptions),"invalid native frame pointers/options");
    const auto geometry=pp::row_geometry(options->logical_rows,options->guidance);
    const auto physical=geometry.physical;
    for(size_t row=0;row<physical;++row)out[row]=nullptr;
    allocation_ready();
    std::vector<pp::CacheSnapshot> input;input.reserve(physical);
    for(size_t row=0;row<physical;++row){TORCH_CHECK(caches[row],"missing physical cache handle");input.push_back(caches[row]->snapshot);}
    std::vector<std::unique_ptr<PpAtenCache>> owned;owned.reserve(physical);
    for(size_t row=0;row<physical;++row)owned.push_back(std::make_unique<PpAtenCache>());
    const auto result=model->engine->frame(input,*options,bounds,uniforms);
    for(size_t row=0;row<physical;++row)owned[row]->snapshot=result.caches[row];
    std::memcpy(tokens,result.tokens.data(),result.tokens.size()*sizeof(uint32_t));
    std::memcpy(terms,result.terms.data(),result.terms.size()*sizeof(float));
    std::memcpy(valid,result.valid.data(),result.valid.size()*sizeof(uint32_t));
    for(size_t row=0;row<physical;++row)out[row]=owned[row].release();});
}
const char* pp_aten_diagnostics(const PpAtenModel* model){return model&&model->engine?model->engine->diagnostics().c_str():nullptr;}
}
