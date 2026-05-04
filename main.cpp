#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cstring>
#include <vector>

namespace {

constexpr int ALPHA = 12;
constexpr int Q = 8;
constexpr int S = ALPHA / Q;
constexpr int ASIZE = 1 << ALPHA;
constexpr int TABLE_MASK = ASIZE - 1;
constexpr int Q2 = Q + Q;
constexpr int END_FIRST_QGRAM = Q - 1;
constexpr int END_SECOND_QGRAM = Q2 - 1;

inline uint32_t link_hash(uint32_t h) {
    return 1u << (h & 0x1Fu);
}

inline uint32_t chain_hash8(const std::vector<unsigned char>& x, int p) {
    uint32_t h = static_cast<uint32_t>(x[p]);
    h = (h << S) + static_cast<uint32_t>(x[p - 1]);
    h = (h << S) + static_cast<uint32_t>(x[p - 2]);
    h = (h << S) + static_cast<uint32_t>(x[p - 3]);
    h = (h << S) + static_cast<uint32_t>(x[p - 4]);
    h = (h << S) + static_cast<uint32_t>(x[p - 5]);
    h = (h << S) + static_cast<uint32_t>(x[p - 6]);
    h = (h << S) + static_cast<uint32_t>(x[p - 7]);
    return h;
}

uint32_t preprocessing_hc8(const std::vector<unsigned char>& pattern, std::vector<uint32_t>& F) {
    const int m = static_cast<int>(pattern.size());
    std::fill(F.begin(), F.end(), 0u);

    uint32_t H = 0;
    const int last_chain = (m < Q2) ? (m - END_FIRST_QGRAM) : Q;

    for (int chain_no = last_chain; chain_no >= 1; --chain_no) {
        H = chain_hash8(pattern, m - chain_no);

        for (int chain_pos = m - chain_no - Q; chain_pos >= END_FIRST_QGRAM; chain_pos -= Q) {
            const uint32_t H_last = H;
            H = chain_hash8(pattern, chain_pos);
            F[H_last & TABLE_MASK] |= link_hash(H);
        }
    }

    const int stop = std::min(m, END_SECOND_QGRAM);
    for (int chain_pos = END_FIRST_QGRAM; chain_pos < stop; ++chain_pos) {
        const uint32_t h = chain_hash8(pattern, chain_pos);
        if (!F[h & TABLE_MASK]) {
            F[h & TABLE_MASK] = link_hash(~h);
        }
    }

    return H;
}

std::vector<unsigned char> read_binary_file(const std::string& file_path) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Falha ao abrir arquivo: " + file_path);
    }
    in.seekg(0, std::ios::end);
    const std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);

    if (size <= 0) {
        throw std::runtime_error("Arquivo vazio ou inválido: " + file_path);
    }

    std::vector<unsigned char> data(static_cast<size_t>(size));
    in.read(reinterpret_cast<char*>(data.data()), size);
    if (!in) {
        throw std::runtime_error("Erro de leitura binária: " + file_path);
    }

    return data;
}

std::string read_text_file(const std::string& file_path) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Falha ao abrir kernel: " + file_path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

double event_millis(cl_event evt) {
    cl_ulong start = 0;
    cl_ulong end = 0;
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
    return static_cast<double>(end - start) / 1e6;
}

std::string cl_error_to_string(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        default: return "CL_ERROR_" + std::to_string(err);
    }
}

void check_cl(cl_int err, const char* where) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string(where) + " falhou com " + cl_error_to_string(err));
    }
}

struct OclContext {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    ~OclContext() {
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

OclContext create_opencl(const std::string& kernel_path) {
    OclContext ocl;

    cl_uint num_platforms = 0;
    check_cl(clGetPlatformIDs(0, nullptr, &num_platforms), "clGetPlatformIDs(count)");
    if (num_platforms == 0) {
        throw std::runtime_error("Nenhuma plataforma OpenCL encontrada.");
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    check_cl(clGetPlatformIDs(num_platforms, platforms.data(), nullptr), "clGetPlatformIDs(list)");

    auto platform_is_amd = [](cl_platform_id p) {
        size_t size = 0;
        clGetPlatformInfo(p, CL_PLATFORM_VENDOR, 0, nullptr, &size);
        std::string vendor(size, '\0');
        clGetPlatformInfo(p, CL_PLATFORM_VENDOR, size, vendor.data(), nullptr);
        return vendor.find("AMD") != std::string::npos || vendor.find("Advanced Micro Devices") != std::string::npos;
    };

    ocl.platform = platforms[0];
    for (cl_platform_id p : platforms) {
        if (platform_is_amd(p)) {
            ocl.platform = p;
            break;
        }
    }

    cl_uint num_devices = 0;
    cl_int err = clGetDeviceIDs(ocl.platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    check_cl(err, "clGetDeviceIDs(count)");
    if (num_devices == 0) {
        throw std::runtime_error("Nenhuma GPU OpenCL encontrada na plataforma selecionada.");
    }

    std::vector<cl_device_id> devices(num_devices);
    check_cl(clGetDeviceIDs(ocl.platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr), "clGetDeviceIDs(list)");
    ocl.device = devices[0];

    ocl.context = clCreateContext(nullptr, 1, &ocl.device, nullptr, nullptr, &err);
    check_cl(err, "clCreateContext");

    ocl.queue = clCreateCommandQueue(ocl.context, ocl.device, CL_QUEUE_PROFILING_ENABLE, &err);
    check_cl(err, "clCreateCommandQueue");

    const std::string source = read_text_file(kernel_path);
    const char* src = source.c_str();
    const size_t src_len = source.size();

    ocl.program = clCreateProgramWithSource(ocl.context, 1, &src, &src_len, &err);
    check_cl(err, "clCreateProgramWithSource");

    err = clBuildProgram(ocl.program, 1, &ocl.device, "-cl-std=CL1.2", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(ocl.program, ocl.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(ocl.program, ocl.device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        throw std::runtime_error("Falha no build do kernel:\n" + log);
    }

    ocl.kernel = clCreateKernel(ocl.program, "hc8_search_kernel", &err);
    check_cl(err, "clCreateKernel");

    return ocl;
}

struct RunResult {
    std::vector<int> indices;
    uint32_t total_found = 0;
    bool overflow = false;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double gpu_wall_ms = 0.0;
};

struct CpuRunResult {
    std::vector<int> indices;
    uint32_t total_found = 0;
    bool overflow = false;
    double preprocessing_ms = 0.0;
    double search_ms = 0.0;
};

CpuRunResult run_hc8_cpu(const std::vector<unsigned char>& text,
                         const std::vector<unsigned char>& pattern,
                         int max_results) {
    if (max_results <= 0) {
        throw std::runtime_error("max_results deve ser > 0.");
    }
    if (pattern.size() < static_cast<size_t>(Q)) {
        throw std::runtime_error("Para hc8, o padrão precisa ter tamanho >= 8.");
    }
    if (text.size() < pattern.size()) {
        throw std::runtime_error("Texto menor que o padrão.");
    }

    const int n = static_cast<int>(text.size());
    const int m = static_cast<int>(pattern.size());
    const int MQ1 = m - Q + 1;

    std::vector<uint32_t> F(ASIZE, 0u);

    const auto t0 = std::chrono::high_resolution_clock::now();
    const uint32_t Hm = preprocessing_hc8(pattern, F);
    const auto t1 = std::chrono::high_resolution_clock::now();

    CpuRunResult out;
    out.preprocessing_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    uint32_t total = 0;
    bool overflow = false;
    std::vector<int> indices;
    indices.reserve(static_cast<size_t>(std::min(max_results, 1'000'000)));

    const auto s0 = std::chrono::high_resolution_clock::now();

    int pos = m - 1;
    while (pos < n) {
        uint32_t H = chain_hash8(text, pos);
        uint32_t V = F[H & TABLE_MASK];

        if (V) {
            const int end_second_qgram_pos = pos - m + Q2;

            while (pos >= end_second_qgram_pos) {
                pos -= Q;
                H = chain_hash8(text, pos);
                if (!(V & link_hash(H))) {
                    goto shift_cpu;
                }
                V = F[H & TABLE_MASK];
            }

            pos = end_second_qgram_pos - Q;
            const int match_start = pos - END_FIRST_QGRAM;

            if (H == Hm && std::memcmp(text.data() + match_start, pattern.data(), static_cast<size_t>(m)) == 0) {
                if (indices.size() < static_cast<size_t>(max_results)) {
                    indices.push_back(match_start);
                } else {
                    overflow = true;
                }
                ++total;
            }
        }

        shift_cpu:
        pos += MQ1;
    }

    const auto s1 = std::chrono::high_resolution_clock::now();
    out.search_ms = std::chrono::duration<double, std::milli>(s1 - s0).count();
    out.total_found = total;
    out.overflow = overflow || (total > static_cast<uint32_t>(max_results));
    out.indices = std::move(indices);
    std::sort(out.indices.begin(), out.indices.end());

    return out;
}

RunResult run_hc8_gpu(const std::vector<unsigned char>& text,
                      const std::vector<unsigned char>& pattern,
                      int chunk_size,
                      int max_results,
                      const std::string& kernel_path) {
    if (chunk_size <= 0) {
        throw std::runtime_error("chunk_size deve ser > 0.");
    }
    if (max_results <= 0) {
        throw std::runtime_error("max_results deve ser > 0.");
    }
    if (pattern.size() < static_cast<size_t>(Q)) {
        throw std::runtime_error("Para hc8, o padrão precisa ter tamanho >= 8.");
    }
    if (text.size() < pattern.size()) {
        throw std::runtime_error("Texto menor que o padrão.");
    }

    const int n = static_cast<int>(text.size());
    const int m = static_cast<int>(pattern.size());

    std::vector<uint32_t> F(ASIZE, 0u);
    const auto t0 = std::chrono::high_resolution_clock::now();
    const uint32_t Hm = preprocessing_hc8(pattern, F);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double preprocessing_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const auto wall_start = std::chrono::high_resolution_clock::now();
    OclContext ocl = create_opencl(kernel_path);

    cl_int err = CL_SUCCESS;
    cl_mem text_buf = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, text.size() * sizeof(unsigned char), nullptr, &err);
    check_cl(err, "clCreateBuffer(text)");

    cl_mem pattern_buf = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, pattern.size() * sizeof(unsigned char), nullptr, &err);
    check_cl(err, "clCreateBuffer(pattern)");

    cl_mem F_buf = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, F.size() * sizeof(uint32_t), nullptr, &err);
    check_cl(err, "clCreateBuffer(F)");

    cl_mem results_buf = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, static_cast<size_t>(max_results) * sizeof(cl_int), nullptr, &err);
    check_cl(err, "clCreateBuffer(results)");

    cl_mem count_buf = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
    check_cl(err, "clCreateBuffer(count)");

    cl_mem overflow_buf = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
    check_cl(err, "clCreateBuffer(overflow)");

    cl_uint zero = 0;
    cl_event evt_w_text = nullptr, evt_w_pattern = nullptr, evt_w_F = nullptr, evt_w_count = nullptr, evt_w_over = nullptr;

    check_cl(clEnqueueWriteBuffer(ocl.queue, text_buf, CL_FALSE, 0, text.size() * sizeof(unsigned char), text.data(), 0, nullptr, &evt_w_text), "clEnqueueWriteBuffer(text)");
    check_cl(clEnqueueWriteBuffer(ocl.queue, pattern_buf, CL_FALSE, 0, pattern.size() * sizeof(unsigned char), pattern.data(), 0, nullptr, &evt_w_pattern), "clEnqueueWriteBuffer(pattern)");
    check_cl(clEnqueueWriteBuffer(ocl.queue, F_buf, CL_FALSE, 0, F.size() * sizeof(uint32_t), F.data(), 0, nullptr, &evt_w_F), "clEnqueueWriteBuffer(F)");
    check_cl(clEnqueueWriteBuffer(ocl.queue, count_buf, CL_FALSE, 0, sizeof(cl_uint), &zero, 0, nullptr, &evt_w_count), "clEnqueueWriteBuffer(count=0)");
    check_cl(clEnqueueWriteBuffer(ocl.queue, overflow_buf, CL_FALSE, 0, sizeof(cl_uint), &zero, 0, nullptr, &evt_w_over), "clEnqueueWriteBuffer(overflow=0)");

    const cl_event h2d_wait_list[] = {evt_w_text, evt_w_pattern, evt_w_F, evt_w_count, evt_w_over};
    check_cl(clWaitForEvents(5, h2d_wait_list), "clWaitForEvents(H2D)");

    RunResult out;
    out.h2d_ms = event_millis(evt_w_text) + event_millis(evt_w_pattern) + event_millis(evt_w_F) + event_millis(evt_w_count) + event_millis(evt_w_over);

    clReleaseEvent(evt_w_text);
    clReleaseEvent(evt_w_pattern);
    clReleaseEvent(evt_w_F);
    clReleaseEvent(evt_w_count);
    clReleaseEvent(evt_w_over);

    const int num_chunks = (n + chunk_size - 1) / chunk_size;
    const size_t local_size = 256;
    const size_t global_size = ((static_cast<size_t>(num_chunks) + local_size - 1) / local_size) * local_size;

    int arg = 0;
    check_cl(clSetKernelArg(ocl.kernel, arg++, sizeof(cl_mem), &text_buf), "clSetKernelArg(text)");
    check_cl(clSetKernelArg(ocl.kernel, arg++, sizeof(cl_int), &n), "clSetKernelArg(n)");
    check_cl(clSetKernelArg(ocl.kernel, arg++, sizeof(cl_mem), &pattern_buf), "clSetKernelArg(pattern)");
    check_cl(clSetKernelArg(ocl.kernel, arg++, sizeof(cl_int), &m), "clSetKernelArg(m)");
    check_cl(clSetKernelArg(ocl.kernel, arg++, sizeof(cl_mem), &F_buf), "clSetKernelArg(F)");
    check_cl(clSetKernelArg(ocl.kernel, arg++, sizeof(cl_uint), &Hm), "clSetKernelArg(Hm)");
    check_cl(clSetKernelArg(ocl.kernel, arg++, sizeof(cl_int), &chunk_size), "clSetKernelArg(chunk_size)");
    check_cl(clSetKernelArg(ocl.kernel, arg++, sizeof(cl_int), &num_chunks), "clSetKernelArg(num_chunks)");
    check_cl(clSetKernelArg(ocl.kernel, arg++, sizeof(cl_int), &max_results), "clSetKernelArg(max_results)");
    check_cl(clSetKernelArg(ocl.kernel, arg++, sizeof(cl_mem), &results_buf), "clSetKernelArg(results)");
    check_cl(clSetKernelArg(ocl.kernel, arg++, sizeof(cl_mem), &count_buf), "clSetKernelArg(count)");
    check_cl(clSetKernelArg(ocl.kernel, arg++, sizeof(cl_mem), &overflow_buf), "clSetKernelArg(overflow)");
    check_cl(clSetKernelArg(ocl.kernel, arg++, ASIZE * sizeof(cl_uint), nullptr), "clSetKernelArg(F_local)");

    cl_event evt_kernel = nullptr;
    check_cl(clEnqueueNDRangeKernel(ocl.queue, ocl.kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, &evt_kernel), "clEnqueueNDRangeKernel");
    check_cl(clWaitForEvents(1, &evt_kernel), "clWaitForEvents(kernel)");
    out.kernel_ms = event_millis(evt_kernel);
    clReleaseEvent(evt_kernel);

    cl_uint total_found = 0;
    cl_uint overflow = 0;

    cl_event evt_r_count = nullptr, evt_r_over = nullptr;
    check_cl(clEnqueueReadBuffer(ocl.queue, count_buf, CL_FALSE, 0, sizeof(cl_uint), &total_found, 0, nullptr, &evt_r_count), "clEnqueueReadBuffer(count)");
    check_cl(clEnqueueReadBuffer(ocl.queue, overflow_buf, CL_FALSE, 0, sizeof(cl_uint), &overflow, 0, nullptr, &evt_r_over), "clEnqueueReadBuffer(overflow)");
    const cl_event d2h_meta_wait_list[] = {evt_r_count, evt_r_over};
    check_cl(clWaitForEvents(2, d2h_meta_wait_list), "clWaitForEvents(D2H-meta)");

    out.d2h_ms = event_millis(evt_r_count) + event_millis(evt_r_over);

    clReleaseEvent(evt_r_count);
    clReleaseEvent(evt_r_over);

    out.total_found = total_found;
    out.overflow = (overflow != 0) || (total_found > static_cast<cl_uint>(max_results));

    const size_t kept = std::min<size_t>(static_cast<size_t>(total_found), static_cast<size_t>(max_results));
    std::vector<int> tmp_indices(kept);
    if (kept > 0) {
        cl_event evt_r_results = nullptr;
        check_cl(clEnqueueReadBuffer(ocl.queue, results_buf, CL_FALSE, 0, kept * sizeof(cl_int), tmp_indices.data(), 0, nullptr, &evt_r_results), "clEnqueueReadBuffer(results-kept)");
        check_cl(clWaitForEvents(1, &evt_r_results), "clWaitForEvents(D2H-results)");
        out.d2h_ms += event_millis(evt_r_results);
        clReleaseEvent(evt_r_results);
    }

    const auto wall_end = std::chrono::high_resolution_clock::now();
    out.gpu_wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    out.indices.assign(tmp_indices.begin(), tmp_indices.end());
    std::sort(out.indices.begin(), out.indices.end());

    clReleaseMemObject(text_buf);
    clReleaseMemObject(pattern_buf);
    clReleaseMemObject(F_buf);
    clReleaseMemObject(results_buf);
    clReleaseMemObject(count_buf);
    clReleaseMemObject(overflow_buf);

    std::cout << "[CPU] preprocessing(ms): " << std::fixed << std::setprecision(3) << preprocessing_ms << "\n";

    return out;
}

std::vector<unsigned char> sample_pattern_from_text(const std::vector<unsigned char>& text, int length, int offset) {
    if (length < Q) {
        throw std::runtime_error("Tamanho de padrão inválido para hc8 (min=8).");
    }
    if (static_cast<int>(text.size()) < length) {
        throw std::runtime_error("Texto menor que o padrão solicitado.");
    }
    if (offset < 0 || (offset + length) > static_cast<int>(text.size())) {
        offset = static_cast<int>(text.size() / 3);
        if (offset + length > static_cast<int>(text.size())) {
            offset = static_cast<int>(text.size()) - length;
        }
    }

    return std::vector<unsigned char>(text.begin() + offset, text.begin() + offset + length);
}

void print_log(const std::string& tag,
               const std::string& text_file,
               int pattern_len,
               int chunk_size,
               int max_results,
               const RunResult& result) {
    const double gpu_total = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    std::cout << "\n=== " << tag << " ===\n";
    std::cout << "text: " << text_file << "\n";
    std::cout << "pattern_len: " << pattern_len << "\n";
    std::cout << "chunk_size: " << chunk_size << "\n";
    std::cout << "max_results: " << max_results << "\n";
    std::cout << "found_total: " << result.total_found << "\n";
    std::cout << "returned_indices: " << result.indices.size() << "\n";
    std::cout << "overflow: " << (result.overflow ? "true" : "false") << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "H2D(ms): " << result.h2d_ms << "\n";
    std::cout << "kernel(ms): " << result.kernel_ms << "\n";
    std::cout << "D2H(ms): " << result.d2h_ms << "\n";
    std::cout << "GPU_total(ms): " << gpu_total << "\n";
    std::cout << "GPU_wall(ms): " << result.gpu_wall_ms << "\n";

    const size_t preview = std::min<size_t>(10, result.indices.size());
    std::cout << "indices_preview(" << preview << "): ";
    for (size_t i = 0; i < preview; ++i) {
        std::cout << result.indices[i] << (i + 1 < preview ? ", " : "");
    }
    std::cout << "\n";
}

void print_cpu_log(const std::string& tag,
                   const std::string& text_file,
                   int pattern_len,
                   int max_results,
                   const CpuRunResult& result) {
    std::cout << "\n=== " << tag << " ===\n";
    std::cout << "text: " << text_file << "\n";
    std::cout << "pattern_len: " << pattern_len << "\n";
    std::cout << "max_results: " << max_results << "\n";
    std::cout << "found_total: " << result.total_found << "\n";
    std::cout << "returned_indices: " << result.indices.size() << "\n";
    std::cout << "overflow: " << (result.overflow ? "true" : "false") << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "preprocessing(ms): " << result.preprocessing_ms << "\n";
    std::cout << "search(ms): " << result.search_ms << "\n";

    const size_t preview = std::min<size_t>(10, result.indices.size());
    std::cout << "indices_preview(" << preview << "): ";
    for (size_t i = 0; i < preview; ++i) {
        std::cout << result.indices[i] << (i + 1 < preview ? ", " : "");
    }
    std::cout << "\n";
}

void compare_and_print(const CpuRunResult& cpu, const RunResult& gpu) {
    const bool same_total = cpu.total_found == gpu.total_found;
    const bool same_overflow = cpu.overflow == gpu.overflow;
    const bool same_indices = cpu.indices == gpu.indices;

    std::cout << "[COMPARE] same_total: " << (same_total ? "true" : "false") << "\n";
    std::cout << "[COMPARE] same_overflow: " << (same_overflow ? "true" : "false") << "\n";
    std::cout << "[COMPARE] same_indices: " << (same_indices ? "true" : "false") << "\n";

    const double gpu_total = gpu.h2d_ms + gpu.kernel_ms + gpu.d2h_ms;
    if (gpu_total > 0.0) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "[COMPARE] speedup(search_cpu / gpu_total): " << (cpu.search_ms / gpu_total) << "x\n";
    }
    if (gpu.gpu_wall_ms > 0.0) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "[COMPARE] speedup(search_cpu / gpu_wall): " << (cpu.search_ms / gpu.gpu_wall_ms) << "x\n";
    }
}

std::string sanitize_filename(std::string s) {
    for (char& c : s) {
        const bool ok =
            (c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') ||
            c == '-' || c == '_';
        if (!ok) c = '_';
    }
    if (s.empty()) s = "run";
    return s;
}

std::string now_compact_utc() {
    std::time_t t = std::time(nullptr);
    std::tm tm_utc{};
#ifdef _WIN32
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_utc, "%Y%m%d_%H%M%S");
    return oss.str();
}

void write_run_csv(const std::string& csv_dir,
                   const std::string& tag,
                   const std::string& text_file,
                   int pattern_len,
                   int chunk_size,
                   int max_results,
                   const RunResult& result) {
    std::filesystem::create_directories(csv_dir);

    const double gpu_total = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    const std::string file_name = sanitize_filename(tag) + "_" + now_compact_utc() + ".csv";
    const std::filesystem::path out_path = std::filesystem::path(csv_dir) / file_name;

    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Falha ao criar CSV: " + out_path.string());
    }

    out << "run_tag,text_file,pattern_len,chunk_size,max_results,found_total,returned_indices,overflow,h2d_ms,kernel_ms,d2h_ms,gpu_total_ms,gpu_wall_ms,index_order,index_value\n";
    out << std::fixed << std::setprecision(3);

    if (result.indices.empty()) {
        out << '"' << tag << "\"," << '"' << text_file << "\"," << pattern_len << ',' << chunk_size << ','
            << max_results << ',' << result.total_found << ',' << result.indices.size() << ','
            << (result.overflow ? 1 : 0) << ','
            << result.h2d_ms << ',' << result.kernel_ms << ',' << result.d2h_ms << ',' << gpu_total << ','
            << result.gpu_wall_ms << ",,\n";
    } else {
        for (size_t i = 0; i < result.indices.size(); ++i) {
            out << '"' << tag << "\"," << '"' << text_file << "\"," << pattern_len << ',' << chunk_size << ','
                << max_results << ',' << result.total_found << ',' << result.indices.size() << ','
                << (result.overflow ? 1 : 0) << ','
                << result.h2d_ms << ',' << result.kernel_ms << ',' << result.d2h_ms << ',' << gpu_total << ','
                << result.gpu_wall_ms << ','
                << i << ',' << result.indices[i] << '\n';
        }
    }

    std::cout << "CSV salvo em: " << out_path.string() << "\n";
}

void write_run_csv_cpu(const std::string& csv_dir,
                       const std::string& tag,
                       const std::string& text_file,
                       int pattern_len,
                       int max_results,
                       const CpuRunResult& result) {
    std::filesystem::create_directories(csv_dir);

    const std::string file_name = sanitize_filename(tag) + "_" + now_compact_utc() + ".csv";
    const std::filesystem::path out_path = std::filesystem::path(csv_dir) / file_name;

    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Falha ao criar CSV: " + out_path.string());
    }

    out << "run_tag,text_file,pattern_len,max_results,found_total,returned_indices,overflow,preprocessing_ms,search_ms,index_order,index_value\n";
    out << std::fixed << std::setprecision(3);

    if (result.indices.empty()) {
        out << '"' << tag << "\"," << '"' << text_file << "\"," << pattern_len << ','
            << max_results << ',' << result.total_found << ',' << result.indices.size() << ','
            << (result.overflow ? 1 : 0) << ',' << result.preprocessing_ms << ',' << result.search_ms << ",,\n";
    } else {
        for (size_t i = 0; i < result.indices.size(); ++i) {
            out << '"' << tag << "\"," << '"' << text_file << "\"," << pattern_len << ','
                << max_results << ',' << result.total_found << ',' << result.indices.size() << ','
                << (result.overflow ? 1 : 0) << ',' << result.preprocessing_ms << ',' << result.search_ms << ','
                << i << ',' << result.indices[i] << '\n';
        }
    }

    std::cout << "CSV salvo em: " << out_path.string() << "\n";
}

void run_benchmark_examples(const std::string& kernel_path,
                            int chunk_size,
                            int max_results,
                            const std::string& csv_dir,
                            bool export_csv) {
    struct Example {
        std::string tag;
        std::string text_file;
        int pattern_len;
        int offset;
    };

    // Exemplos baseados no conjunto SEA2024: pizza-dna, pizza-english, pizza-protein (100MB)
    const std::vector<Example> examples = {
        {"benchmark-example:dna", "bd/dna.100MB", 16, 1'000'000},
        {"benchmark-example:english", "bd/english.100MB", 64, 20'000'000},
        {"benchmark-example:protein", "bd/proteins.100MB", 32, 5'000'000},

        // Casos extras (english) para comparação CPU x GPU em mais cenários.
        {"benchmark-example:english-extra-8", "bd/english.100MB", 8, 500'000},
        {"benchmark-example:english-extra-16", "bd/english.100MB", 16, 2'000'000},
        {"benchmark-example:english-extra-32", "bd/english.100MB", 32, 10'000'000},
        {"benchmark-example:english-extra-128", "bd/english.100MB", 128, 30'000'000},
    };

    for (const auto& ex : examples) {
        const auto text = read_binary_file(ex.text_file);
        const auto pattern = sample_pattern_from_text(text, ex.pattern_len, ex.offset);

        const auto cpu = run_hc8_cpu(text, pattern, max_results);
        print_cpu_log(ex.tag + ":cpu", ex.text_file, ex.pattern_len, max_results, cpu);

        const auto gpu = run_hc8_gpu(text, pattern, chunk_size, max_results, kernel_path);
        print_log(ex.tag + ":gpu", ex.text_file, ex.pattern_len, chunk_size, max_results, gpu);

        compare_and_print(cpu, gpu);

        if (export_csv) {
            write_run_csv_cpu(csv_dir, ex.tag + "-cpu", ex.text_file, ex.pattern_len, max_results, cpu);
            write_run_csv(csv_dir, ex.tag + "-gpu", ex.text_file, ex.pattern_len, chunk_size, max_results, gpu);
        }
    }
}

void usage() {
    std::cout
        << "Uso:\n"
        << "  main.exe --run-examples [--chunk-size 262144] [--max-results 1000000] [--csv-dir results-csv]\n"
        << "  main.exe --text bd/dna.100MB --pattern ACGTACGT --chunk-size 262144 --max-results 1000000 [--csv-dir results-csv]  (roda CPU e GPU)\n"
        << "  main.exe --text bd/english.100MB --pattern-len 64 --pattern-offset 20000000 --chunk-size 262144 --max-results 1000000 [--csv-dir results-csv]  (roda CPU e GPU)\n"
        << "  main.exe --run-examples --no-csv\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string text_file;
        std::string pattern_arg;
        int pattern_len = -1;
        int pattern_offset = -1;
        int chunk_size = 256 * 1024;
        int max_results = 1'000'000;
        std::string csv_dir = "results-csv";
        bool export_csv = true;
        bool run_examples = false;

        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--text" && i + 1 < argc) {
                text_file = argv[++i];
            } else if (a == "--pattern" && i + 1 < argc) {
                pattern_arg = argv[++i];
            } else if (a == "--pattern-len" && i + 1 < argc) {
                pattern_len = std::stoi(argv[++i]);
            } else if (a == "--pattern-offset" && i + 1 < argc) {
                pattern_offset = std::stoi(argv[++i]);
            } else if (a == "--chunk-size" && i + 1 < argc) {
                chunk_size = std::stoi(argv[++i]);
            } else if (a == "--max-results" && i + 1 < argc) {
                max_results = std::stoi(argv[++i]);
            } else if (a == "--csv-dir" && i + 1 < argc) {
                csv_dir = argv[++i];
            } else if (a == "--no-csv") {
                export_csv = false;
            } else if (a == "--run-examples") {
                run_examples = true;
            } else if (a == "--help" || a == "-h") {
                usage();
                return 0;
            } else {
                std::cerr << "Argumento desconhecido: " << a << "\n";
                usage();
                return 1;
            }
        }

        const std::string kernel_path = "kernel.cl";

        if (run_examples || (text_file.empty() && pattern_arg.empty() && pattern_len < 0)) {
            run_benchmark_examples(kernel_path, chunk_size, max_results, csv_dir, export_csv);
            return 0;
        }

        if (text_file.empty()) {
            throw std::runtime_error("Informe --text ou use --run-examples.");
        }

        const auto text = read_binary_file(text_file);

        std::vector<unsigned char> pattern;
        if (!pattern_arg.empty()) {
            pattern.assign(pattern_arg.begin(), pattern_arg.end());
        } else {
            if (pattern_len < 0) {
                throw std::runtime_error("Informe --pattern ou --pattern-len.");
            }
            pattern = sample_pattern_from_text(text, pattern_len, pattern_offset);
        }

        const auto cpu = run_hc8_cpu(text, pattern, max_results);
        print_cpu_log("single-run:cpu", text_file, static_cast<int>(pattern.size()), max_results, cpu);

        const auto gpu = run_hc8_gpu(text, pattern, chunk_size, max_results, kernel_path);
        print_log("single-run:gpu", text_file, static_cast<int>(pattern.size()), chunk_size, max_results, gpu);

        compare_and_print(cpu, gpu);

        if (export_csv) {
            write_run_csv_cpu(csv_dir, "single-run-cpu", text_file, static_cast<int>(pattern.size()), max_results, cpu);
            write_run_csv(csv_dir, "single-run-gpu", text_file, static_cast<int>(pattern.size()), chunk_size, max_results, gpu);
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Erro: " << ex.what() << "\n";
        return 1;
    }
}
