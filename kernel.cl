#define ALPHA 12
#define Q 8
#define S (ALPHA / Q)
#define ASIZE (1 << ALPHA)
#define TABLE_MASK (ASIZE - 1)
#define Q2 (Q + Q)
#define END_FIRST_QGRAM (Q - 1)
#define LINK_HASH(H) (1u << ((H) & 0x1Fu))

inline uint chain_hash8_global(__global const uchar* x, int p) {
    uint h = (uint)x[p];
    h = (h << S) + (uint)x[p - 1];
    h = (h << S) + (uint)x[p - 2];
    h = (h << S) + (uint)x[p - 3];
    h = (h << S) + (uint)x[p - 4];
    h = (h << S) + (uint)x[p - 5];
    h = (h << S) + (uint)x[p - 6];
    h = (h << S) + (uint)x[p - 7];
    return h;
}

inline int verify_pattern(__global const uchar* text,
                          int start,
                          __global const uchar* pattern,
                          int m) {
    for (int i = 0; i < m; ++i) {
        if (text[start + i] != pattern[i]) {
            return 0;
        }
    }
    return 1;
}

__kernel void hc8_search_kernel(__global const uchar* text,
                                const int n,
                                __global const uchar* pattern,
                                const int m,
                                __global const uint* F_global,
                                const uint Hm,
                                const int chunk_size,
                                const int num_chunks,
                                const int max_results,
                                __global int* results,
                                __global uint* result_count,
                                __global uint* overflow_flag,
                                __local uint* F_local) {
    if (m < Q || n < m || chunk_size <= 0 || num_chunks <= 0 || max_results <= 0) {
        return;
    }

    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);

    for (int i = lid; i < ASIZE; i += lsize) {
        F_local[i] = F_global[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int gid = get_global_id(0);
    if (gid >= num_chunks) {
        return;
    }

    const int chunk_start = gid * chunk_size;
    if (chunk_start >= n) {
        return;
    }

    const int core_end = min(n, chunk_start + chunk_size);
    const int max_pos = min(n - 1, core_end + m - 2);
    const int MQ1 = m - Q + 1;

    int pos = chunk_start + m - 1;
    if (pos < END_FIRST_QGRAM || pos > max_pos) {
        return;
    }

    while (pos <= max_pos) {
        uint H = chain_hash8_global(text, pos);
        uint V = F_local[H & TABLE_MASK];

        if (V) {
            const int end_second_qgram_pos = pos - m + Q2;

            while (pos >= end_second_qgram_pos) {
                pos -= Q;
                H = chain_hash8_global(text, pos);

                if (!(V & LINK_HASH(H))) {
                    goto shift;
                }

                V = F_local[H & TABLE_MASK];
            }

            pos = end_second_qgram_pos - Q;
            const int match_start = pos - END_FIRST_QGRAM;

            if (H == Hm &&
                match_start >= chunk_start &&
                match_start < core_end &&
                match_start >= 0 &&
                (match_start + m) <= n &&
                verify_pattern(text, match_start, pattern, m)) {

                uint idx = atomic_inc((volatile __global uint*)result_count);
                if (idx < (uint)max_results) {
                    results[idx] = match_start;
                } else {
                    atomic_or((volatile __global uint*)overflow_flag, 1u);
                }
            }
        }

        shift:
        pos += MQ1;
    }
}
