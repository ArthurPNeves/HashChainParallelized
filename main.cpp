#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
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

// Estatística de benchmark: mediana e coeficiente de variação (CV%, = desvio/média*100).
// A mediana é robusta a outliers de escalonamento; o CV% mede a dispersão relativa.
struct Stats {
    double median = 0.0;
    double cv = 0.0;
};

Stats compute_stats(std::vector<double> samples) {
    Stats s;
    if (samples.empty()) return s;
    std::sort(samples.begin(), samples.end());
    const size_t n = samples.size();
    s.median = (n % 2 == 1) ? samples[n / 2]
                            : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
    double sum = 0.0;
    for (double v : samples) sum += v;
    const double mean = sum / static_cast<double>(n);
    if (mean > 0.0 && n > 1) {
        double acc = 0.0;
        for (double v : samples) acc += (v - mean) * (v - mean);
        const double stddev = std::sqrt(acc / static_cast<double>(n - 1));
        s.cv = (stddev / mean) * 100.0;
    }
    return s;
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

struct PinnedTextUpload {
    cl_mem device_buf = nullptr;
    double h2d_ms = 0.0;
};

PinnedTextUpload upload_text_pinned(OclContext& ocl, const std::vector<unsigned char>& text) {
    const size_t bytes = text.size() * sizeof(unsigned char);
    cl_int err = CL_SUCCESS;

    cl_mem staging_buf = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr, &err);
    check_cl(err, "clCreateBuffer(text-pinned-staging)");

    void* host_ptr = clEnqueueMapBuffer(ocl.queue, staging_buf, CL_TRUE,
                                        CL_MAP_WRITE_INVALIDATE_REGION,
                                        0, bytes, 0, nullptr, nullptr, &err);
    check_cl(err, "clEnqueueMapBuffer(text-pinned-staging)");

    std::memcpy(host_ptr, text.data(), bytes);

    cl_event evt_unmap = nullptr;
    check_cl(clEnqueueUnmapMemObject(ocl.queue, staging_buf, host_ptr, 0, nullptr, &evt_unmap),
             "clEnqueueUnmapMemObject(text-pinned-staging)");

    cl_mem device_buf = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    check_cl(err, "clCreateBuffer(text-device)");

    cl_event evt_copy = nullptr;
    const cl_event copy_wait[] = {evt_unmap};
    check_cl(clEnqueueCopyBuffer(ocl.queue, staging_buf, device_buf, 0, 0, bytes, 1, copy_wait, &evt_copy),
             "clEnqueueCopyBuffer(staging->device)");
    check_cl(clWaitForEvents(1, &evt_copy), "clWaitForEvents(text-pinned-copy)");

    PinnedTextUpload r;
    r.device_buf = device_buf;
    r.h2d_ms = event_millis(evt_unmap) + event_millis(evt_copy);

    clReleaseEvent(evt_unmap);
    clReleaseEvent(evt_copy);
    clReleaseMemObject(staging_buf);
    return r;
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

RunResult run_hc8_gpu_inner(OclContext& ocl,
                            cl_mem text_buf,
                            int n,
                            const std::vector<unsigned char>& pattern,
                            int chunk_size,
                            int max_results) {
    if (chunk_size <= 0) {
        throw std::runtime_error("chunk_size deve ser > 0.");
    }
    if (max_results <= 0) {
        throw std::runtime_error("max_results deve ser > 0.");
    }
    if (pattern.size() < static_cast<size_t>(Q)) {
        throw std::runtime_error("Para hc8, o padrão precisa ter tamanho >= 8.");
    }
    if (n < static_cast<int>(pattern.size())) {
        throw std::runtime_error("Texto menor que o padrão.");
    }

    const int m = static_cast<int>(pattern.size());

    std::vector<uint32_t> F(ASIZE, 0u);
    const auto t0 = std::chrono::high_resolution_clock::now();
    const uint32_t Hm = preprocessing_hc8(pattern, F);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double preprocessing_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const auto wall_start = std::chrono::high_resolution_clock::now();

    cl_int err = CL_SUCCESS;
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
    cl_event evt_w_pattern = nullptr, evt_w_F = nullptr, evt_w_count = nullptr, evt_w_over = nullptr;

    check_cl(clEnqueueWriteBuffer(ocl.queue, pattern_buf, CL_FALSE, 0, pattern.size() * sizeof(unsigned char), pattern.data(), 0, nullptr, &evt_w_pattern), "clEnqueueWriteBuffer(pattern)");
    check_cl(clEnqueueWriteBuffer(ocl.queue, F_buf, CL_FALSE, 0, F.size() * sizeof(uint32_t), F.data(), 0, nullptr, &evt_w_F), "clEnqueueWriteBuffer(F)");
    check_cl(clEnqueueWriteBuffer(ocl.queue, count_buf, CL_FALSE, 0, sizeof(cl_uint), &zero, 0, nullptr, &evt_w_count), "clEnqueueWriteBuffer(count=0)");
    check_cl(clEnqueueWriteBuffer(ocl.queue, overflow_buf, CL_FALSE, 0, sizeof(cl_uint), &zero, 0, nullptr, &evt_w_over), "clEnqueueWriteBuffer(overflow=0)");

    const cl_event h2d_wait_list[] = {evt_w_pattern, evt_w_F, evt_w_count, evt_w_over};
    check_cl(clWaitForEvents(4, h2d_wait_list), "clWaitForEvents(H2D)");

    RunResult out;
    out.h2d_ms = event_millis(evt_w_pattern) + event_millis(evt_w_F) + event_millis(evt_w_count) + event_millis(evt_w_over);

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

    clReleaseMemObject(pattern_buf);
    clReleaseMemObject(F_buf);
    clReleaseMemObject(results_buf);
    clReleaseMemObject(count_buf);
    clReleaseMemObject(overflow_buf);

    (void)preprocessing_ms;  // preprocessamento medido no host; não logado por repetição (evita ruído)

    return out;
}

RunResult run_hc8_gpu(const std::vector<unsigned char>& text,
                      const std::vector<unsigned char>& pattern,
                      int chunk_size,
                      int max_results,
                      const std::string& kernel_path) {
    const auto wall_setup_start = std::chrono::high_resolution_clock::now();
    OclContext ocl = create_opencl(kernel_path);

    PinnedTextUpload up = upload_text_pinned(ocl, text);
    cl_mem text_buf = up.device_buf;
    const double text_h2d_ms = up.h2d_ms;

    const auto wall_setup_end = std::chrono::high_resolution_clock::now();
    const double setup_wall_ms = std::chrono::duration<double, std::milli>(wall_setup_end - wall_setup_start).count();

    RunResult out = run_hc8_gpu_inner(ocl, text_buf, static_cast<int>(text.size()), pattern, chunk_size, max_results);
    out.h2d_ms += text_h2d_ms;
    out.gpu_wall_ms += setup_wall_ms;

    clReleaseMemObject(text_buf);
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

// Uma linha por (cenário, regime) do CSV de timings: fonte oficial da Tabela 1 e das figuras.
// Toda agregação estatística (mediana+CV%) é feita aqui no host; o Python apenas lê e plota.
struct TimingsRow {
    std::string run_tag;
    std::string text_file;
    std::string base;
    int pattern_len = 0;
    std::string regime;       // "cold" ou "warm"
    int repeats = 0;
    Stats cpu_search;         // tempo de busca sequencial (mediana+CV%)
    double cpu_preproc_ms = 0.0;
    double file_read_ms = 0.0;       // leitura de disco (só cold)
    Stats gpu_upload;         // RAM->VRAM: pin+memcpy+DMA (só cold)
    double h2d_ms = 0.0;             // transferência de pattern/F por consulta (pequena)
    Stats kernel;             // kernel de busca (mediana+CV%)
    double d2h_ms = 0.0;
    Stats gpu_query_wall;     // wall por consulta (exclui upload do texto e contexto)
    double cpu_total_ms = 0.0;
    double gpu_total_ms = 0.0;
    double speedup = 0.0;
    uint32_t found_total = 0;
    bool overflow = false;
    bool same_indices = true;
};

void write_timings_csv(const std::string& csv_dir, const std::vector<TimingsRow>& rows) {
    std::filesystem::create_directories(csv_dir);
    const std::string file_name = "timings_" + now_compact_utc() + ".csv";
    const std::filesystem::path out_path = std::filesystem::path(csv_dir) / file_name;

    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Falha ao criar CSV de timings: " + out_path.string());
    }

    out << "run_tag,text_file,base,pattern_len,regime,repeats,"
           "cpu_search_ms_median,cpu_search_ms_cv,cpu_preproc_ms,"
           "file_read_ms,gpu_upload_ms_median,gpu_upload_ms_cv,"
           "h2d_ms,kernel_ms_median,kernel_ms_cv,d2h_ms,gpu_query_wall_ms_median,gpu_query_wall_ms_cv,"
           "cpu_total_ms,gpu_total_ms,speedup,found_total,overflow,same_indices\n";
    out << std::fixed << std::setprecision(4);

    for (const auto& r : rows) {
        out << '"' << r.run_tag << "\",\"" << r.text_file << "\",\"" << r.base << "\","
            << r.pattern_len << ',' << r.regime << ',' << r.repeats << ','
            << r.cpu_search.median << ',' << r.cpu_search.cv << ',' << r.cpu_preproc_ms << ','
            << r.file_read_ms << ',' << r.gpu_upload.median << ',' << r.gpu_upload.cv << ','
            << r.h2d_ms << ',' << r.kernel.median << ',' << r.kernel.cv << ',' << r.d2h_ms << ','
            << r.gpu_query_wall.median << ',' << r.gpu_query_wall.cv << ','
            << r.cpu_total_ms << ',' << r.gpu_total_ms << ',' << r.speedup << ','
            << r.found_total << ',' << (r.overflow ? 1 : 0) << ',' << (r.same_indices ? 1 : 0) << '\n';
    }

    std::cout << "CSV de timings salvo em: " << out_path.string() << "\n";
}

void run_benchmark_examples(const std::string& kernel_path,
                            int chunk_size,
                            int max_results,
                            const std::string& csv_dir,
                            bool export_csv,
                            int repeat,
                            int warmup) {
    struct Example {
        std::string tag;
        std::string text_file;
        int pattern_len;
        int offset;
    };

    // Exemplos baseados no conjunto SEA2024: pizza-dna, pizza-english, pizza-protein (100MB).
    // As 3 primeiras entradas (uma por base) pagam o cold-upload do texto;
    // as demais reusam a mesma base já no device (warm cache).
    const std::vector<Example> examples = {
        {"benchmark-example:dna", "bd/dna.100MB", 16, 1'000'000},
        {"benchmark-example:english", "bd/english.100MB", 64, 20'000'000},
        {"benchmark-example:protein", "bd/proteins.100MB", 32, 5'000'000},

        // Sweep de m para DNA (warm-cache).
        {"benchmark-example:dna-extra-8", "bd/dna.100MB", 8, 1'500'000},
        {"benchmark-example:dna-extra-32", "bd/dna.100MB", 32, 2'000'000},
        {"benchmark-example:dna-extra-64", "bd/dna.100MB", 64, 3'000'000},
        {"benchmark-example:dna-extra-128", "bd/dna.100MB", 128, 4'000'000},

        // Sweep de m para English (warm-cache).
        {"benchmark-example:english-extra-8", "bd/english.100MB", 8, 500'000},
        {"benchmark-example:english-extra-16", "bd/english.100MB", 16, 2'000'000},
        {"benchmark-example:english-extra-32", "bd/english.100MB", 32, 10'000'000},
        {"benchmark-example:english-extra-128", "bd/english.100MB", 128, 30'000'000},

        // Sweep de m para Proteins (warm-cache).
        {"benchmark-example:protein-extra-8", "bd/proteins.100MB", 8, 1'000'000},
        {"benchmark-example:protein-extra-16", "bd/proteins.100MB", 16, 3'000'000},
        {"benchmark-example:protein-extra-64", "bd/proteins.100MB", 64, 6'000'000},
        {"benchmark-example:protein-extra-128", "bd/proteins.100MB", 128, 8'000'000},
    };

    const auto setup_t0 = std::chrono::high_resolution_clock::now();
    OclContext ocl = create_opencl(kernel_path);
    const auto setup_t1 = std::chrono::high_resolution_clock::now();
    const double setup_ms = std::chrono::duration<double, std::milli>(setup_t1 - setup_t0).count();
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[SETUP] OpenCL context+kernel build (uma vez): " << setup_ms << " ms\n";

    struct CachedText {
        std::vector<unsigned char> data;
        cl_mem buf = nullptr;
    };
    std::map<std::string, CachedText> text_cache;

    auto release_cache = [&]() {
        for (auto& kv : text_cache) {
            if (kv.second.buf) clReleaseMemObject(kv.second.buf);
            kv.second.buf = nullptr;
        }
    };

    std::vector<TimingsRow> timing_rows;

    auto base_of = [](const std::string& tf) -> std::string {
        if (tf.find("english") != std::string::npos) return "English";
        if (tf.find("dna") != std::string::npos) return "DNA";
        if (tf.find("protein") != std::string::npos) return "Proteins";
        return "Outros";
    };

    std::cout << "[CONFIG] repeticoes(warmup+amostras)=" << warmup << "+" << repeat << "\n";

    try {
        for (const auto& ex : examples) {
            // ---------------------------------------------------------------
            // Regime COLD: primeira aparição de cada texto (1x por base).
            // Custo inerente: leitura de disco (1x) + upload RAM->VRAM (medido N vezes).
            // ---------------------------------------------------------------
            auto cache_it = text_cache.find(ex.text_file);
            const bool first_load = (cache_it == text_cache.end());

            double file_read_ms = 0.0;
            Stats upload_stats;  // só preenchido no cold

            if (first_load) {
                const auto r0 = std::chrono::high_resolution_clock::now();
                std::vector<unsigned char> data = read_binary_file(ex.text_file);
                const auto r1 = std::chrono::high_resolution_clock::now();
                file_read_ms = std::chrono::duration<double, std::milli>(r1 - r0).count();

                std::vector<double> upload_samples;
                upload_samples.reserve(static_cast<size_t>(repeat));
                cl_mem last_buf = nullptr;
                for (int it = 0; it < warmup + repeat; ++it) {
                    const auto u0 = std::chrono::high_resolution_clock::now();
                    PinnedTextUpload up = upload_text_pinned(ocl, data);
                    const auto u1 = std::chrono::high_resolution_clock::now();
                    if (it >= warmup) {
                        upload_samples.push_back(std::chrono::duration<double, std::milli>(u1 - u0).count());
                    }
                    if (it + 1 < warmup + repeat) {
                        clReleaseMemObject(up.device_buf);  // descarta e remede
                    } else {
                        last_buf = up.device_buf;           // mantém o último em cache
                    }
                }
                upload_stats = compute_stats(upload_samples);

                CachedText c;
                c.data = std::move(data);
                c.buf = last_buf;
                cache_it = text_cache.emplace(ex.text_file, std::move(c)).first;
            }

            const std::vector<unsigned char>& text_ref = cache_it->second.data;
            const cl_mem text_buf = cache_it->second.buf;
            const auto pattern = sample_pattern_from_text(text_ref, ex.pattern_len, ex.offset);

            // ---------------------------------------------------------------
            // Regime WARM: repetições para todos os cenários (texto já em VRAM).
            // ---------------------------------------------------------------
            std::vector<double> cpu_search, gpu_wall, gpu_kernel, gpu_d2h, gpu_h2d;
            cpu_search.reserve(static_cast<size_t>(repeat));
            gpu_wall.reserve(static_cast<size_t>(repeat));
            double cpu_preproc_ms = 0.0;
            uint32_t found_total = 0;
            bool overflow_flag = false;
            bool same_idx = true;
            CpuRunResult last_cpu;
            RunResult last_gpu;

            for (int it = 0; it < warmup + repeat; ++it) {
                CpuRunResult cpu = run_hc8_cpu(text_ref, pattern, max_results);
                RunResult gpu = run_hc8_gpu_inner(ocl, text_buf, static_cast<int>(text_ref.size()),
                                                  pattern, chunk_size, max_results);

                const bool same_total = cpu.total_found == gpu.total_found;
                const bool same_over = cpu.overflow == gpu.overflow;
                const bool same_indices = cpu.indices == gpu.indices;
                if (!(same_total && same_over && same_indices)) {
                    same_idx = false;
                    std::cout << "[REGRESSION-FAIL] " << ex.tag << " m=" << ex.pattern_len
                              << " rep=" << it << " (total=" << same_total
                              << " over=" << same_over << " indices=" << same_indices << ")\n";
                }

                if (it >= warmup) {
                    cpu_search.push_back(cpu.search_ms);
                    gpu_wall.push_back(gpu.gpu_wall_ms);
                    gpu_kernel.push_back(gpu.kernel_ms);
                    gpu_d2h.push_back(gpu.d2h_ms);
                    gpu_h2d.push_back(gpu.h2d_ms);
                    cpu_preproc_ms = cpu.preprocessing_ms;
                    found_total = cpu.total_found;
                    overflow_flag = cpu.overflow || gpu.overflow;
                }

                last_cpu = std::move(cpu);
                last_gpu = std::move(gpu);
            }

            const Stats cpu_s = compute_stats(cpu_search);
            const Stats wall_s = compute_stats(gpu_wall);
            const Stats kern_s = compute_stats(gpu_kernel);
            const double d2h_med = compute_stats(gpu_d2h).median;
            const double h2d_med = compute_stats(gpu_h2d).median;
            const std::string base = base_of(ex.text_file);

            // Linha WARM (texto residente; comparação search-only vs wall por consulta).
            TimingsRow warm;
            warm.run_tag = ex.tag; warm.text_file = ex.text_file; warm.base = base;
            warm.pattern_len = ex.pattern_len; warm.regime = "warm"; warm.repeats = repeat;
            warm.cpu_search = cpu_s; warm.cpu_preproc_ms = cpu_preproc_ms;
            warm.h2d_ms = h2d_med; warm.kernel = kern_s; warm.d2h_ms = d2h_med;
            warm.gpu_query_wall = wall_s;
            warm.cpu_total_ms = cpu_s.median;
            warm.gpu_total_ms = wall_s.median;
            warm.speedup = (wall_s.median > 0.0) ? (cpu_s.median / wall_s.median) : 0.0;
            warm.found_total = found_total; warm.overflow = overflow_flag; warm.same_indices = same_idx;
            timing_rows.push_back(warm);

            // Linha COLD (1ª aparição): leitura de disco cobrada de AMBOS os lados (comparação justa).
            if (first_load) {
                TimingsRow cold;
                cold.run_tag = ex.tag; cold.text_file = ex.text_file; cold.base = base;
                cold.pattern_len = ex.pattern_len; cold.regime = "cold"; cold.repeats = repeat;
                cold.cpu_search = cpu_s; cold.cpu_preproc_ms = cpu_preproc_ms;
                cold.file_read_ms = file_read_ms; cold.gpu_upload = upload_stats;
                cold.h2d_ms = h2d_med; cold.kernel = kern_s; cold.d2h_ms = d2h_med;
                cold.gpu_query_wall = wall_s;
                cold.cpu_total_ms = file_read_ms + cpu_preproc_ms + cpu_s.median;
                cold.gpu_total_ms = file_read_ms + upload_stats.median + wall_s.median;
                cold.speedup = (cold.gpu_total_ms > 0.0) ? (cold.cpu_total_ms / cold.gpu_total_ms) : 0.0;
                cold.found_total = found_total; cold.overflow = overflow_flag; cold.same_indices = same_idx;
                timing_rows.push_back(cold);
            }

            print_cpu_log(ex.tag + ":cpu", ex.text_file, ex.pattern_len, max_results, last_cpu);
            print_log(ex.tag + ":gpu", ex.text_file, ex.pattern_len, chunk_size, max_results, last_gpu);
            std::cout << std::fixed << std::setprecision(3)
                      << "[STATS] " << ex.tag << " m=" << ex.pattern_len
                      << " | CPU search med=" << cpu_s.median << "ms (cv=" << cpu_s.cv << "%)"
                      << " | GPU wall med=" << wall_s.median << "ms (cv=" << wall_s.cv << "%)"
                      << " | speedup_warm=" << warm.speedup << "x";
            if (first_load) {
                std::cout << " | COLD speedup=" << timing_rows.back().speedup << "x"
                          << " (read=" << file_read_ms << "ms, upload med=" << upload_stats.median << "ms)";
            }
            std::cout << "\n";

            if (export_csv) {
                const std::string idx_dir = csv_dir + "/indices";
                write_run_csv_cpu(idx_dir, ex.tag + "-cpu", ex.text_file, ex.pattern_len, max_results, last_cpu);
                write_run_csv(idx_dir, ex.tag + "-gpu", ex.text_file, ex.pattern_len, chunk_size, max_results, last_gpu);
            }
        }
    } catch (...) {
        release_cache();
        throw;
    }

    if (export_csv) {
        write_timings_csv(csv_dir, timing_rows);
    }
    release_cache();
}

void usage() {
    std::cout
        << "Uso:\n"
        << "  main.exe --run-examples [--repeat 30] [--warmup 1] [--chunk-size 262144] [--max-results 1000000] [--csv-dir results-csv]\n"
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
        int repeat = 30;
        int warmup = 1;

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
            } else if (a == "--repeat" && i + 1 < argc) {
                repeat = std::stoi(argv[++i]);
            } else if (a == "--warmup" && i + 1 < argc) {
                warmup = std::stoi(argv[++i]);
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

        if (repeat < 1) repeat = 1;
        if (warmup < 0) warmup = 0;

        if (run_examples || (text_file.empty() && pattern_arg.empty() && pattern_len < 0)) {
            run_benchmark_examples(kernel_path, chunk_size, max_results, csv_dir, export_csv, repeat, warmup);
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

        std::cout << "[CONFIG] repeticoes(warmup+amostras)=" << warmup << "+" << repeat << "\n";
        std::vector<double> cpu_search_samples, gpu_wall_samples;
        cpu_search_samples.reserve(static_cast<size_t>(repeat));
        gpu_wall_samples.reserve(static_cast<size_t>(repeat));
        CpuRunResult cpu;
        RunResult gpu;
        for (int it = 0; it < warmup + repeat; ++it) {
            cpu = run_hc8_cpu(text, pattern, max_results);
            gpu = run_hc8_gpu(text, pattern, chunk_size, max_results, kernel_path);
            if (it >= warmup) {
                cpu_search_samples.push_back(cpu.search_ms);
                gpu_wall_samples.push_back(gpu.gpu_wall_ms);
            }
        }

        print_cpu_log("single-run:cpu", text_file, static_cast<int>(pattern.size()), max_results, cpu);
        print_log("single-run:gpu", text_file, static_cast<int>(pattern.size()), chunk_size, max_results, gpu);
        compare_and_print(cpu, gpu);

        const Stats css = compute_stats(cpu_search_samples);
        const Stats gws = compute_stats(gpu_wall_samples);
        std::cout << std::fixed << std::setprecision(3)
                  << "[STATS] single-run | CPU search med=" << css.median << "ms (cv=" << css.cv << "%)"
                  << " | GPU wall med=" << gws.median << "ms (cv=" << gws.cv << "%)"
                  << " | speedup=" << (gws.median > 0.0 ? css.median / gws.median : 0.0) << "x\n";

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
