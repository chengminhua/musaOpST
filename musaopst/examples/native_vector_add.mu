// SPDX-License-Identifier: MulanPSL-2.0
// Original musaopst example; see project LICENSE.
// Protocol vector_add_f32_v1: argv A.bin B.bin OUT.bin N DEVICE WARMUP SAMPLES ITERATIONS.
// Files contain little-endian IEEE-754 float32. stdout is exactly one JSON object.
#include <musa_runtime.h>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(musaError_t status, const char* expr) {
    if (status != musaSuccess) throw std::runtime_error(std::string(expr) + ": " + musaGetErrorString(status));
}
#define CHECK(expr) check((expr), #expr)
static std::string quoted(const std::string& value) {
    std::ostringstream out; out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else out << c;
    }
    out << '"'; return out.str();
}
static std::vector<float> read_input(const char* path, std::size_t n) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != std::streamoff(n * sizeof(float)))
        throw std::runtime_error("Input file byte length mismatch");
    std::vector<float> data(n); file.seekg(0);
    if (n && !file.read(reinterpret_cast<char*>(data.data()), n * sizeof(float)))
        throw std::runtime_error("Input file read failed");
    return data;
}
__global__ void add(const float* a, const float* b, float* out, std::size_t n) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
int main(int argc, char** argv) {
    try {
        if (argc != 9) throw std::runtime_error("Usage: native_vector_add A.bin B.bin OUT.bin N DEVICE WARMUP SAMPLES ITERATIONS");
        const std::size_t n = std::stoull(argv[4]);
        const int device = std::stoi(argv[5]);
        const unsigned warmup = std::stoul(argv[6]), samples = std::stoul(argv[7]), iterations = std::stoul(argv[8]);
        if (!iterations || n > 2147483647ULL * 256 || n > std::numeric_limits<std::size_t>::max() / 4)
            throw std::runtime_error("Invalid size/iteration count");
        const uint16_t endian_probe = 1;
        if (*reinterpret_cast<const unsigned char*>(&endian_probe) != 1)
            throw std::runtime_error("Protocol requires a little-endian host");
        auto a = read_input(argv[1], n), b = read_input(argv[2], n);
        std::vector<float> out(n);
        CHECK(musaSetDevice(device));
        musaDeviceProp property{}; CHECK(musaGetDeviceProperties(&property, device));
        if (property.maxThreadsPerBlock < 256) throw std::runtime_error("Device does not support a 256-thread block");
        int runtime = 0, driver = 0;
        CHECK(musaRuntimeGetVersion(&runtime)); CHECK(musaDriverGetVersion(&driver));
        std::size_t free_bytes = 0, total_bytes = 0;
        CHECK(musaMemGetInfo(&free_bytes, &total_bytes));
        float *da = nullptr, *db = nullptr, *dc = nullptr;
        if (n) {
            CHECK(musaMalloc(reinterpret_cast<void**>(&da), n * sizeof(float)));
            CHECK(musaMalloc(reinterpret_cast<void**>(&db), n * sizeof(float)));
            CHECK(musaMalloc(reinterpret_cast<void**>(&dc), n * sizeof(float)));
            CHECK(musaMemcpy(da, a.data(), n * sizeof(float), musaMemcpyHostToDevice));
            CHECK(musaMemcpy(db, b.data(), n * sizeof(float), musaMemcpyHostToDevice));
            CHECK(musaMemset(dc, 0xff, n * sizeof(float)));
        }
        auto launch = [&]() {
            if (n) add<<<(n + 255) / 256, 256, 0, 0>>>(da, db, dc, n);
        };
        launch(); CHECK(musaGetLastError()); CHECK(musaDeviceSynchronize());
        for (unsigned i = 0; i < warmup; ++i) launch();
        CHECK(musaGetLastError()); CHECK(musaDeviceSynchronize());
        std::vector<double> timings;
        if (samples) {
            musaEvent_t start, stop; CHECK(musaEventCreate(&start)); CHECK(musaEventCreate(&stop));
            for (unsigned sample = 0; sample < samples; ++sample) {
                CHECK(musaEventRecord(start, 0));
                for (unsigned i = 0; i < iterations; ++i) launch();
                CHECK(musaEventRecord(stop, 0)); CHECK(musaGetLastError());
                CHECK(musaEventSynchronize(stop));
                float elapsed = 0; CHECK(musaEventElapsedTime(&elapsed, start, stop));
                timings.push_back(elapsed);
            }
            CHECK(musaEventDestroy(stop)); CHECK(musaEventDestroy(start));
        }
        CHECK(musaDeviceSynchronize());
        // This is the final timed output, not a recomputed CPU or GPU substitute.
        if (n) CHECK(musaMemcpy(out.data(), dc, n * sizeof(float), musaMemcpyDeviceToHost));
        std::ofstream result(argv[3], std::ios::binary | std::ios::trunc);
        if (!result || (n && !result.write(reinterpret_cast<const char*>(out.data()), n * sizeof(float))))
            throw std::runtime_error("Output file write failed");
        result.close();
        if (n) { CHECK(musaFree(dc)); CHECK(musaFree(db)); CHECK(musaFree(da)); }
        std::cout << std::setprecision(12)
                  << "{\"protocol\":\"vector_add_f32_v1\",\"device_execution\":\"musa\",\"device_name\":"
                  << quoted(property.name) << ",\"capability\":" << quoted(std::to_string(property.major) + "." + std::to_string(property.minor))
                  << ",\"runtime_api_version\":" << runtime << ",\"driver_api_version\":" << driver
                  << ",\"memory\":{\"free_bytes_before_allocation\":" << free_bytes << ",\"total_bytes\":" << total_bytes << "}"
                  << ",\"output\":{\"shape\":[" << n << "],\"dtype\":\"float32\"},\"empty_no_kernel_launch\":" << (n ? "false" : "true")
                  << ",\"raw_samples_ms\":[";
        for (std::size_t i = 0; i < timings.size(); ++i) { if (i) std::cout << ','; std::cout << timings[i] / iterations; }
        std::cout << "],\"raw_batch_elapsed_ms\":[";
        for (std::size_t i = 0; i < timings.size(); ++i) { if (i) std::cout << ','; std::cout << timings[i]; }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "NATIVE_ERROR: " << error.what() << '\n';
        return 1;
    }
}
