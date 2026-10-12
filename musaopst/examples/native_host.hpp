// SPDX-License-Identifier: MulanPSL-2.0
// Original reusable MUSA host helper. User sources implement prepare/launch hooks.
#pragma once
#include "native_json.hpp"
#include <musa_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <functional>
namespace opst {
struct Unsupported : std::runtime_error { using std::runtime_error::runtime_error; };
inline void check(musaError_t status, const char* operation) {
    if (status != musaSuccess) throw std::runtime_error(std::string(operation)+": "+musaGetErrorString(status));
}
#define OPST_MUSA(expr) ::opst::check((expr), #expr)
inline size_t itemsize(const std::string& dtype) {
    if (dtype == "bool" || dtype == "int8" || dtype == "uint8") return 1;
    if (dtype == "float16" || dtype == "bfloat16" || dtype == "int16" || dtype == "uint16") return 2;
    if (dtype == "float32" || dtype == "int32" || dtype == "uint32") return 4;
    if (dtype == "float64" || dtype == "int64" || dtype == "uint64") return 8;
    throw Unsupported("Native helper does not encode dtype "+dtype);
}
inline std::vector<int64_t> integers(const Json& value) {
    if (value.kind != Json::Array) throw std::runtime_error("Expected shape/stride array");
    std::vector<int64_t> result;
    for (const auto& x : value.items) { auto n = x.integer(); if (n < 0) throw Unsupported("Negative shape/stride not supported"); result.push_back(n); }
    return result;
}
inline std::string read_text(const std::string& name) {
    std::ifstream input(name, std::ios::binary); if (!input) throw std::runtime_error("Cannot read "+name);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
inline void write_text(const std::string& name, const std::string& data) {
    std::ofstream out(name); if (!out || !(out << data)) throw std::runtime_error("Cannot write "+name);
}
struct Layout { int rank; int64_t shape[8], strides[8], offset, n; };
struct Tensor {
    std::string dtype;
    std::vector<int64_t> shape, strides;
    int64_t offset = 0;
    size_t bytes = 0;
    std::vector<uint8_t> host;
    void* device = nullptr;
    bool owns_device = true;
    int alias_input = -1;
    Tensor() = default;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    ~Tensor() { if (device && owns_device) musaFree(device); }
    int64_t count() const {
        int64_t count = 1;
        for (auto n : shape) { if (n && count > INT64_MAX/n) throw std::runtime_error("Shape product overflow"); count *= n; }
        return count;
    }
    Layout layout() const {
        if (shape.size() > 8) throw Unsupported("Bundled native kernels support rank <= 8");
        Layout value{}; value.rank = int(shape.size()); value.offset = offset; value.n = count();
        for (size_t i=0;i<shape.size();++i) { value.shape[i]=shape[i]; value.strides[i]=strides[i]; } return value;
    }
    void validate_span() const {
        if (shape.size() != strides.size() || offset < 0) throw std::runtime_error("Invalid tensor layout");
        uint64_t span = uint64_t(offset);
        if (count()) {
            ++span;
            for (size_t i=0;i<shape.size();++i) { uint64_t dim=uint64_t(std::max<int64_t>(shape[i]-1,0)), step=uint64_t(strides[i]);
                if (dim && step > (UINT64_MAX-span)/dim) throw std::runtime_error("Storage span overflow"); span += dim*step; }
        }
        if (span > bytes/itemsize(dtype)) throw std::runtime_error("Tensor view exceeds physical storage");
    }
    void upload() { validate_span(); if (!device) OPST_MUSA(musaMalloc(&device, std::max<size_t>(bytes,1))); if (bytes) OPST_MUSA(musaMemcpy(device,host.data(),bytes,musaMemcpyHostToDevice)); }
    void download() { host.resize(bytes); if (bytes) OPST_MUSA(musaMemcpy(host.data(),device,bytes,musaMemcpyDeviceToHost)); }
    Json metadata(const std::string& file) const {
        Json value=Json::object(); value["file"]=file; value["dtype"]=dtype; value["shape"]=integer_array(shape); value["strides"]=integer_array(strides);
        value["storage_offset"]=offset; value["storage_bytes"]=bytes; value["byte_order"]="little"; value["alias_inputs"]=Json::array();
        if (alias_input>=0) value["alias_inputs"].items.emplace_back(alias_input);
        bool contiguous=true; int64_t stride=1; for (int i=int(shape.size())-1;i>=0;--i) { if(shape[i]>1 && strides[i]!=stride) contiguous=false; stride*=std::max<int64_t>(shape[i],1); }
        value["contiguous"]=contiguous; return value;
    }
};
using TensorPtr = std::unique_ptr<Tensor>;
struct Context {
    Json request;
    std::vector<TensorPtr> inputs, outputs;
    musaStream_t stream = nullptr;
    bool owns_stream = false, mutates_inputs = false;
    ~Context() { if (owns_stream && stream) musaStreamDestroy(stream); }
    const Json& attrs() const { return request.at("attrs"); }
    Tensor& output(const std::vector<int64_t>& shape, const std::string& dtype) {
        auto tensor=std::make_unique<Tensor>(); tensor->shape=shape; tensor->dtype=dtype;
        int64_t stride=1; tensor->strides.resize(shape.size());
        for (int i=int(shape.size())-1;i>=0;--i) { tensor->strides[i]=stride; if (shape[i] && stride>INT64_MAX/std::max<int64_t>(shape[i],1)) throw std::runtime_error("Output size overflow"); stride*=std::max<int64_t>(shape[i],1); }
        if (uint64_t(tensor->count()) > SIZE_MAX/itemsize(dtype)) throw std::runtime_error("Output bytes overflow");
        tensor->bytes=std::max<size_t>(size_t(tensor->count())*itemsize(dtype),itemsize(dtype)); tensor->host.resize(tensor->bytes,0);
        tensor->upload(); outputs.push_back(std::move(tensor)); return *outputs.back();
    }
    Tensor& alias_output(size_t index) {
        Tensor& input=*inputs.at(index); auto tensor=std::make_unique<Tensor>(); tensor->shape=input.shape; tensor->strides=input.strides; tensor->dtype=input.dtype;
        tensor->offset=input.offset; tensor->bytes=input.bytes; tensor->device=input.device; tensor->owns_device=false; tensor->alias_input=int(index);
        outputs.push_back(std::move(tensor)); mutates_inputs=true; return *outputs.back();
    }
    void reset() { for(auto& tensor:inputs) tensor->upload(); OPST_MUSA(musaStreamSynchronize(stream)); }
};
inline int run_native(int argc, char** argv, const std::function<void(Context&)>& prepare,
                      const std::function<void(Context&)>& launch) {
    std::string result_path;
    try {
        if(argc!=5 || std::string(argv[1])!="--request" || std::string(argv[3])!="--result") throw std::runtime_error("Expected --request request.json --result result.json");
        result_path=argv[4]; Context ctx; ctx.request=parse_json(read_text(argv[2]));
        if(ctx.request.at("protocol").str()!="musaopst_native_v2") throw std::runtime_error("Protocol mismatch");
        const uint16_t endian=1; if(*reinterpret_cast<const uint8_t*>(&endian)!=1) throw Unsupported("Little-endian host required");
        int device=int(ctx.request.at("device").integer()); OPST_MUSA(musaSetDevice(device));
        const Json& timing=ctx.request.at("timing"); auto stream_mode=timing.text("stream","current");
        if(stream_mode=="independent") { OPST_MUSA(musaStreamCreate(&ctx.stream)); ctx.owns_stream=true; }
        for(const auto& description:ctx.request.at("inputs").items) {
            auto tensor=std::make_unique<Tensor>(); tensor->dtype=description.at("dtype").str(); tensor->shape=integers(description.at("shape")); tensor->strides=integers(description.at("strides"));
            tensor->offset=description.integer("storage_offset",0); auto nbytes=description.at("storage_bytes").integer(); if(nbytes<0) throw std::runtime_error("Negative storage bytes"); tensor->bytes=size_t(nbytes);
            auto data=read_text(description.at("file").str()); if(data.size()!=tensor->bytes) throw std::runtime_error("Input file byte count mismatch");
            tensor->host.assign(data.begin(),data.end()); tensor->upload(); ctx.inputs.push_back(std::move(tensor));
        }
        prepare(ctx); launch(ctx); OPST_MUSA(musaGetLastError()); OPST_MUSA(musaStreamSynchronize(ctx.stream));
        // Detect mutations in the actual physical buffers, including padding.
        for(auto& input:ctx.inputs) { std::vector<uint8_t> actual(input->bytes); if(input->bytes) OPST_MUSA(musaMemcpy(actual.data(),input->device,input->bytes,musaMemcpyDeviceToHost)); if(actual!=input->host) ctx.mutates_inputs=true; }
        auto mode=timing.text("mode","batch_event"), cache=timing.text("cache","warm");
        int64_t samples=timing.integer("samples",0), iterations=mode=="batch_event"?timing.integer("iterations",1):1, warmup=timing.integer("warmup",0);
        if(samples<0 || warmup<0 || iterations<1) throw std::runtime_error("Invalid timing counts");
        if(samples && ctx.mutates_inputs && (mode=="batch_event" || !timing.boolean("allow_inplace",false))) throw Unsupported("Input-mutating native timing requires single_event/wall_time and allow_inplace");
        std::unique_ptr<Tensor> interference;
        if(samples && cache=="interference") { auto bytes=timing.integer("cache_bytes",0); if(bytes<=0) throw std::runtime_error("cache_bytes must be positive"); interference=std::make_unique<Tensor>(); interference->dtype="uint8"; interference->shape={bytes}; interference->strides={1}; interference->bytes=size_t(bytes); interference->host.resize(bytes); interference->upload(); }
        for(int64_t i=0;i<warmup;++i) { if(ctx.mutates_inputs) ctx.reset(); launch(ctx); }
        OPST_MUSA(musaGetLastError()); OPST_MUSA(musaStreamSynchronize(ctx.stream));
        Json raw=Json::array(), averages=Json::array(); musaEvent_t start=nullptr,stop=nullptr;
        if(samples && mode!="wall_time") { OPST_MUSA(musaEventCreate(&start)); OPST_MUSA(musaEventCreate(&stop)); }
        for(int64_t sample=0;sample<samples;++sample) {
            if(ctx.mutates_inputs) ctx.reset();
            if(interference) { OPST_MUSA(musaMemsetAsync(interference->device,0,interference->bytes,ctx.stream)); OPST_MUSA(musaStreamSynchronize(ctx.stream)); }
            double elapsed;
            if(mode=="wall_time") { OPST_MUSA(musaStreamSynchronize(ctx.stream)); auto begin=std::chrono::steady_clock::now(); launch(ctx); OPST_MUSA(musaGetLastError()); OPST_MUSA(musaStreamSynchronize(ctx.stream)); elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count(); }
            else { OPST_MUSA(musaEventRecord(start,ctx.stream)); for(int64_t i=0;i<iterations;++i) launch(ctx); OPST_MUSA(musaEventRecord(stop,ctx.stream)); OPST_MUSA(musaGetLastError()); OPST_MUSA(musaEventSynchronize(stop)); float ms=0; OPST_MUSA(musaEventElapsedTime(&ms,start,stop)); elapsed=ms; }
            raw.items.emplace_back(elapsed); averages.items.emplace_back(elapsed/iterations);
        }
        if(start) OPST_MUSA(musaEventDestroy(start)); if(stop) OPST_MUSA(musaEventDestroy(stop)); OPST_MUSA(musaDeviceSynchronize());
        Json result=Json::object(); result["protocol"]="musaopst_native_v2"; result["status"]="PASS"; result["device_execution"]="musa"; result["device_index"]=device; result["outputs"]=Json::array();
        for(size_t i=0;i<ctx.outputs.size();++i) { auto& tensor=*ctx.outputs[i]; tensor.download(); auto file=ctx.request.at("output_prefix").str()+"_output_"+std::to_string(i)+".bin";
            std::ofstream out(file,std::ios::binary); if(!out || (tensor.bytes && !out.write(reinterpret_cast<char*>(tensor.host.data()),tensor.bytes))) throw std::runtime_error("Cannot write output file"); out.close(); result["outputs"].items.push_back(tensor.metadata(file)); }
        result["raw_samples_ms"]=averages; result["raw_batch_elapsed_ms"]=raw; result["timing_policy"]=Json::object();
        result["timing_policy"]["input_mutation_detected"]=ctx.mutates_inputs; result["timing_policy"]["input_reset"]=ctx.mutates_inputs?"before each sample, excluded from timer":"not needed";
        result["stream"]=Json::object(); result["stream"]["policy"]=stream_mode; result["stream"]["handle"]=std::to_string(reinterpret_cast<uintptr_t>(ctx.stream));
        musaDeviceProp property{}; OPST_MUSA(musaGetDeviceProperties(&property,device)); int driver=0,runtime=0; OPST_MUSA(musaDriverGetVersion(&driver)); OPST_MUSA(musaRuntimeGetVersion(&runtime)); size_t free=0,total=0; OPST_MUSA(musaMemGetInfo(&free,&total));
        result["runtime"]=Json::object(); result["runtime"]["device_name"]=property.name; result["runtime"]["driver_api_version"]=driver; result["runtime"]["runtime_api_version"]=runtime;
        result["runtime"]["device_capability"]=integer_array({property.major,property.minor});
        result["runtime"]["memory"]=Json::object(); result["runtime"]["memory"]["free_bytes"]=free; result["runtime"]["memory"]["total_bytes"]=total;
        // Explicit successful frees are part of execution success.
        for(auto& tensor:ctx.outputs) if(tensor->device && tensor->owns_device) { OPST_MUSA(musaFree(tensor->device)); tensor->device=nullptr; }
        for(auto& tensor:ctx.inputs) if(tensor->device) { OPST_MUSA(musaFree(tensor->device)); tensor->device=nullptr; }
        if(interference && interference->device) { OPST_MUSA(musaFree(interference->device)); interference->device=nullptr; }
        if(ctx.owns_stream) { OPST_MUSA(musaStreamDestroy(ctx.stream)); ctx.stream=nullptr; ctx.owns_stream=false; }
        write_text(result_path,result.dump()); return 0;
    } catch(const std::exception& error) {
        Json result=Json::object(); result["protocol"]="musaopst_native_v2"; result["status"]=dynamic_cast<const Unsupported*>(&error)?"UNSUPPORTED":"EXECUTION_FAILED"; result["message"]=error.what();
        if(!result_path.empty()) try { write_text(result_path,result.dump()); } catch(...) {}
        std::cerr << error.what() << '\n'; return 1;
    }
}
} // namespace opst
