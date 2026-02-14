#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <limits.h>

typedef uint16_t f16;

static inline f16 f32_to_f16(float val) {
    uint32_t x;
    memcpy(&x, &val, sizeof(uint32_t));

    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t exp = (x >> 23) & 0xFFu;
    uint32_t mant = x & 0x7FFFFFu;

    if (exp == 0xFFu) {
        if (mant != 0) {
            uint16_t payload = (uint16_t)(mant >> 13);
            if ((payload & 0x03FFu) == 0) payload = 1;
            return (f16)(sign | 0x7C00u | (payload & 0x03FFu) | 0x0200u);
        } else {
            return (f16)(sign | 0x7C00u);
        }
    }

    int32_t half_exp = (int32_t)exp - 127 + 15;

    if (half_exp >= 31) {
        return (f16)(sign | 0x7C00u);
    }

    if (half_exp <= 0) {
        if (half_exp < -10) {
            return (f16)sign;
        }
        mant |= 0x800000u;
        int32_t shift = 14 - half_exp;
        uint32_t half_mant = mant >> shift;
        uint32_t round_mask = 1u << (shift - 1);
        uint32_t round_bit = (mant & round_mask) != 0u;
        uint32_t remainder = mant & (round_mask - 1u);
        if (round_bit && (remainder != 0u || (half_mant & 1u))) {
            half_mant++;
        }
        return (f16)(sign | (uint16_t)half_mant);
    }

    uint32_t half_mant = mant >> 13;
    uint32_t round_bit = (mant >> 12) & 1u;
    uint32_t remainder = mant & 0xFFFu;
    if (round_bit && (remainder != 0u || (half_mant & 1u))) {
        half_mant++;
        if (half_mant == 0x400u) {
            half_mant = 0;
            half_exp++;
            if (half_exp >= 31) {
                return (f16)(sign | 0x7C00u);
            }
        }
    }

    return (f16)(sign | ((uint16_t)half_exp << 10) | (uint16_t)(half_mant & 0x3FFu));
}

static inline float f16_to_f32(f16 val) {
    uint32_t h = (uint32_t)val;
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;

    uint32_t x;

    if (exp == 0) {
        if (mant == 0) {
            x = sign;
        } else {
            int32_t e = -14;
            while ((mant & 0x400u) == 0u) {
                mant <<= 1;
                e--;
            }
            mant &= 0x3FFu;
            uint32_t fexp = (uint32_t)(e + 127);
            uint32_t fmant = mant << 13;
            x = sign | (fexp << 23) | fmant;
        }
    } else if (exp == 31) {
        if (mant == 0) {
            x = sign | 0x7F800000u;
        } else {
            uint32_t payload = mant << 13;
            payload |= 0x400000u;
            x = sign | 0x7F800000u | payload;
        }
    } else {
        uint32_t fexp = (exp - 15 + 127) & 0xFFu;
        uint32_t fmant = mant << 13;
        x = sign | (fexp << 23) | fmant;
    }

    float result;
    memcpy(&result, &x, sizeof(float));
    return result;
}

static void f32_array_to_f16(const float *src, f16 *dst, int64_t count) {
    for (int64_t i = 0; i < count; i++) {
        dst[i] = f32_to_f16(src[i]);
    }
}

static void f16_array_to_f32(const f16 *src, float *dst, int64_t count) {
    for (int64_t i = 0; i < count; i++) {
        dst[i] = f16_to_f32(src[i]);
    }
}

struct futhark_context_config {
    int device;
    int platform;
    size_t default_group_size;
    size_t default_num_groups;
    size_t default_tile_size;
    int profiling;
    int owns_config;
};

struct futhark_context {
    struct futhark_context_config cfg_copy;
    void *opencl_ctx;
    char *error;
};

struct futhark_f16_1d {
    f16 *data;
    int64_t shape[1];
};

struct futhark_f16_2d {
    f16 *data;
    int64_t shape[2];
};

struct futhark_f16_3d {
    f16 *data;
    int64_t shape[3];
};

struct futhark_f32_1d {
    float *data;
    int64_t shape[1];
};

struct futhark_f32_2d {
    float *data;
    int64_t shape[2];
};

struct futhark_f32_3d {
    float *data;
    int64_t shape[3];
};

struct futhark_u64_1d {
    uint64_t *data;
    int64_t shape[1];
};

struct futhark_i64_1d {
    int64_t *data;
    int64_t shape[1];
};

struct futhark_context_config *futhark_context_config_new(void) {
    struct futhark_context_config *cfg = (struct futhark_context_config*)malloc(sizeof(struct futhark_context_config));
    if (cfg) {
        cfg->device = 0;
        cfg->platform = 0;
        cfg->default_group_size = 256;
        cfg->default_num_groups = 128;
        cfg->default_tile_size = 16;
        cfg->profiling = 0;
        cfg->owns_config = 1;
    }
    return cfg;
}

void futhark_context_config_free(struct futhark_context_config *cfg) {
    if (!cfg) return;
    if (cfg->owns_config) {
        free(cfg);
    }
}

void futhark_context_config_set_device(struct futhark_context_config *cfg, int device) {
    if (cfg) cfg->device = device;
}

void futhark_context_config_set_platform(struct futhark_context_config *cfg, int platform) {
    if (cfg) cfg->platform = platform;
}

void futhark_context_config_set_default_group_size(struct futhark_context_config *cfg, int size) {
    if (cfg && size > 0) cfg->default_group_size = (size_t)size;
}

void futhark_context_config_set_default_num_groups(struct futhark_context_config *cfg, int num) {
    if (cfg && num > 0) cfg->default_num_groups = (size_t)num;
}

void futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg, int size) {
    if (cfg && size > 0) cfg->default_tile_size = (size_t)size;
}

struct futhark_context *futhark_context_new(struct futhark_context_config *cfg) {
    if (!cfg) return NULL;
    struct futhark_context *ctx = (struct futhark_context*)malloc(sizeof(struct futhark_context));
    if (ctx) {
        memcpy(&ctx->cfg_copy, cfg, sizeof(struct futhark_context_config));
        ctx->opencl_ctx = NULL;
        ctx->error = NULL;
    }
    return ctx;
}

void futhark_context_free(struct futhark_context *ctx) {
    if (ctx) {
        free(ctx->error);
        free(ctx);
    }
}

int futhark_context_sync(struct futhark_context *ctx) {
    (void)ctx;
    return 0;
}

char *futhark_context_get_error(struct futhark_context *ctx) {
    return ctx ? ctx->error : NULL;
}

static int check_size_overflow_2(int64_t a, int64_t b) {
    if (a < 0 || b < 0) return 1;
    if (a == 0 || b == 0) return 0;
    if (a > INT64_MAX / b) return 1;
    return 0;
}

static int check_size_overflow_3(int64_t a, int64_t b, int64_t c) {
    if (a < 0 || b < 0 || c < 0) return 1;
    if (a == 0 || b == 0 || c == 0) return 0;
    if (check_size_overflow_2(a, b)) return 1;
    int64_t ab = a * b;
    if (check_size_overflow_2(ab, c)) return 1;
    return 0;
}

static int size_for_elems_i64(int64_t elems, size_t elem_size, size_t *out_bytes) {
    if (elems < 0) return 1;
    if (elems == 0) {
        *out_bytes = 0;
        return 0;
    }
    if ((uint64_t)elems > (uint64_t)(SIZE_MAX / elem_size)) return 1;
    *out_bytes = (size_t)elems * elem_size;
    return 0;
}

struct futhark_f16_1d *futhark_new_f16_1d(struct futhark_context *ctx, const f16 *data, int64_t dim0) {
    (void)ctx;
    if (dim0 < 0) return NULL;
    struct futhark_f16_1d *arr = (struct futhark_f16_1d*)malloc(sizeof(struct futhark_f16_1d));
    if (!arr) return NULL;
    arr->shape[0] = dim0;
    if (dim0 == 0) {
        arr->data = NULL;
        return arr;
    }
    size_t bytes;
    if (size_for_elems_i64(dim0, sizeof(f16), &bytes)) {
        free(arr);
        return NULL;
    }
    arr->data = (f16*)malloc(bytes);
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    if (data) {
        memcpy(arr->data, data, bytes);
    } else {
        memset(arr->data, 0, bytes);
    }
    return arr;
}

struct futhark_f16_2d *futhark_new_f16_2d(struct futhark_context *ctx, const f16 *data, int64_t dim0, int64_t dim1) {
    (void)ctx;
    if (dim0 < 0 || dim1 < 0) return NULL;
    if (check_size_overflow_2(dim0, dim1)) return NULL;
    struct futhark_f16_2d *arr = (struct futhark_f16_2d*)malloc(sizeof(struct futhark_f16_2d));
    if (!arr) return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    int64_t total = dim0 * dim1;
    if (total == 0) {
        arr->data = NULL;
        return arr;
    }
    size_t bytes;
    if (size_for_elems_i64(total, sizeof(f16), &bytes)) {
        free(arr);
        return NULL;
    }
    arr->data = (f16*)malloc(bytes);
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    if (data) {
        memcpy(arr->data, data, bytes);
    } else {
        memset(arr->data, 0, bytes);
    }
    return arr;
}

struct futhark_f16_3d *futhark_new_f16_3d(struct futhark_context *ctx, const f16 *data, int64_t dim0, int64_t dim1, int64_t dim2) {
    (void)ctx;
    if (dim0 < 0 || dim1 < 0 || dim2 < 0) return NULL;
    if (check_size_overflow_3(dim0, dim1, dim2)) return NULL;
    struct futhark_f16_3d *arr = (struct futhark_f16_3d*)malloc(sizeof(struct futhark_f16_3d));
    if (!arr) return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    arr->shape[2] = dim2;
    int64_t total = dim0 * dim1 * dim2;
    if (total == 0) {
        arr->data = NULL;
        return arr;
    }
    size_t bytes;
    if (size_for_elems_i64(total, sizeof(f16), &bytes)) {
        free(arr);
        return NULL;
    }
    arr->data = (f16*)malloc(bytes);
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    if (data) {
        memcpy(arr->data, data, bytes);
    } else {
        memset(arr->data, 0, bytes);
    }
    return arr;
}

struct futhark_f16_2d *futhark_new_f16_2d_from_f32(struct futhark_context *ctx, const float *data, int64_t dim0, int64_t dim1) {
    (void)ctx;
    if (dim0 < 0 || dim1 < 0) return NULL;
    if (check_size_overflow_2(dim0, dim1)) return NULL;
    struct futhark_f16_2d *arr = (struct futhark_f16_2d*)malloc(sizeof(struct futhark_f16_2d));
    if (!arr) return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    int64_t count = dim0 * dim1;
    if (count == 0) {
        arr->data = NULL;
        return arr;
    }
    size_t bytes;
    if (size_for_elems_i64(count, sizeof(f16), &bytes)) {
        free(arr);
        return NULL;
    }
    arr->data = (f16*)malloc(bytes);
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    if (data) {
        f32_array_to_f16(data, arr->data, count);
    } else {
        memset(arr->data, 0, bytes);
    }
    return arr;
}

struct futhark_f16_3d *futhark_new_f16_3d_from_f32(struct futhark_context *ctx, const float *data, int64_t dim0, int64_t dim1, int64_t dim2) {
    (void)ctx;
    if (dim0 < 0 || dim1 < 0 || dim2 < 0) return NULL;
    if (check_size_overflow_3(dim0, dim1, dim2)) return NULL;
    struct futhark_f16_3d *arr = (struct futhark_f16_3d*)malloc(sizeof(struct futhark_f16_3d));
    if (!arr) return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    arr->shape[2] = dim2;
    int64_t count = dim0 * dim1 * dim2;
    if (count == 0) {
        arr->data = NULL;
        return arr;
    }
    size_t bytes;
    if (size_for_elems_i64(count, sizeof(f16), &bytes)) {
        free(arr);
        return NULL;
    }
    arr->data = (f16*)malloc(bytes);
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    if (data) {
        f32_array_to_f16(data, arr->data, count);
    } else {
        memset(arr->data, 0, bytes);
    }
    return arr;
}

int futhark_free_f16_1d(struct futhark_context *ctx, struct futhark_f16_1d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
    return 0;
}

int futhark_free_f16_2d(struct futhark_context *ctx, struct futhark_f16_2d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
    return 0;
}

int futhark_free_f16_3d(struct futhark_context *ctx, struct futhark_f16_3d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
    return 0;
}

int futhark_values_f16_1d(struct futhark_context *ctx, struct futhark_f16_1d *arr, f16 *data) {
    (void)ctx;
    if (!arr || !data) return 1;
    if (arr->shape[0] < 0) return 1;
    if (arr->shape[0] == 0) return 0;
    if (!arr->data) return 1;
    size_t bytes;
    if (size_for_elems_i64(arr->shape[0], sizeof(f16), &bytes)) return 1;
    memcpy(data, arr->data, bytes);
    return 0;
}

int futhark_values_f16_2d(struct futhark_context *ctx, struct futhark_f16_2d *arr, f16 *data) {
    (void)ctx;
    if (!arr || !data) return 1;
    if (arr->shape[0] < 0 || arr->shape[1] < 0) return 1;
    if (check_size_overflow_2(arr->shape[0], arr->shape[1])) return 1;
    int64_t total = arr->shape[0] * arr->shape[1];
    if (total == 0) return 0;
    if (!arr->data) return 1;
    size_t bytes;
    if (size_for_elems_i64(total, sizeof(f16), &bytes)) return 1;
    memcpy(data, arr->data, bytes);
    return 0;
}

int futhark_values_f16_3d(struct futhark_context *ctx, struct futhark_f16_3d *arr, f16 *data) {
    (void)ctx;
    if (!arr || !data) return 1;
    if (arr->shape[0] < 0 || arr->shape[1] < 0 || arr->shape[2] < 0) return 1;
    if (check_size_overflow_3(arr->shape[0], arr->shape[1], arr->shape[2])) return 1;
    int64_t total = arr->shape[0] * arr->shape[1] * arr->shape[2];
    if (total == 0) return 0;
    if (!arr->data) return 1;
    size_t bytes;
    if (size_for_elems_i64(total, sizeof(f16), &bytes)) return 1;
    memcpy(data, arr->data, bytes);
    return 0;
}

int futhark_values_f16_2d_to_f32(struct futhark_context *ctx, struct futhark_f16_2d *arr, float *data) {
    (void)ctx;
    if (!arr || !data) return 1;
    if (arr->shape[0] < 0 || arr->shape[1] < 0) return 1;
    if (check_size_overflow_2(arr->shape[0], arr->shape[1])) return 1;
    int64_t count = arr->shape[0] * arr->shape[1];
    if (count == 0) return 0;
    if (!arr->data) return 1;
    f16_array_to_f32(arr->data, data, count);
    return 0;
}

int futhark_values_f16_3d_to_f32(struct futhark_context *ctx, struct futhark_f16_3d *arr, float *data) {
    (void)ctx;
    if (!arr || !data) return 1;
    if (arr->shape[0] < 0 || arr->shape[1] < 0 || arr->shape[2] < 0) return 1;
    if (check_size_overflow_3(arr->shape[0], arr->shape[1], arr->shape[2])) return 1;
    int64_t count = arr->shape[0] * arr->shape[1] * arr->shape[2];
    if (count == 0) return 0;
    if (!arr->data) return 1;
    f16_array_to_f32(arr->data, data, count);
    return 0;
}

void *futhark_values_raw_f16_2d(struct futhark_context *ctx, struct futhark_f16_2d *arr) {
    (void)ctx;
    if (arr) {
        return (void*)arr->data;
    }
    return NULL;
}

int futhark_shape_f16_2d(struct futhark_context *ctx, struct futhark_f16_2d *arr, int64_t *dims) {
    (void)ctx;
    if (!arr || !dims) return 1;
    dims[0] = arr->shape[0];
    dims[1] = arr->shape[1];
    return 0;
}

struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx, const float *data, int64_t dim0) {
    (void)ctx;
    if (dim0 < 0) return NULL;
    struct futhark_f32_1d *arr = (struct futhark_f32_1d*)malloc(sizeof(struct futhark_f32_1d));
    if (!arr) return NULL;
    arr->shape[0] = dim0;
    if (dim0 == 0) {
        arr->data = NULL;
        return arr;
    }
    size_t bytes;
    if (size_for_elems_i64(dim0, sizeof(float), &bytes)) {
        free(arr);
        return NULL;
    }
    arr->data = (float*)malloc(bytes);
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    if (data) {
        memcpy(arr->data, data, bytes);
    } else {
        memset(arr->data, 0, bytes);
    }
    return arr;
}

struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx, const float *data, int64_t dim0, int64_t dim1) {
    (void)ctx;
    if (dim0 < 0 || dim1 < 0) return NULL;
    if (check_size_overflow_2(dim0, dim1)) return NULL;
    struct futhark_f32_2d *arr = (struct futhark_f32_2d*)malloc(sizeof(struct futhark_f32_2d));
    if (!arr) return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    int64_t total = dim0 * dim1;
    if (total == 0) {
        arr->data = NULL;
        return arr;
    }
    size_t bytes;
    if (size_for_elems_i64(total, sizeof(float), &bytes)) {
        free(arr);
        return NULL;
    }
    arr->data = (float*)malloc(bytes);
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    if (data) {
        memcpy(arr->data, data, bytes);
    } else {
        memset(arr->data, 0, bytes);
    }
    return arr;
}

struct futhark_f32_3d *futhark_new_f32_3d(struct futhark_context *ctx, const float *data, int64_t dim0, int64_t dim1, int64_t dim2) {
    (void)ctx;
    if (dim0 < 0 || dim1 < 0 || dim2 < 0) return NULL;
    if (check_size_overflow_3(dim0, dim1, dim2)) return NULL;
    struct futhark_f32_3d *arr = (struct futhark_f32_3d*)malloc(sizeof(struct futhark_f32_3d));
    if (!arr) return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    arr->shape[2] = dim2;
    int64_t total = dim0 * dim1 * dim2;
    if (total == 0) {
        arr->data = NULL;
        return arr;
    }
    size_t bytes;
    if (size_for_elems_i64(total, sizeof(float), &bytes)) {
        free(arr);
        return NULL;
    }
    arr->data = (float*)malloc(bytes);
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    if (data) {
        memcpy(arr->data, data, bytes);
    } else {
        memset(arr->data, 0, bytes);
    }
    return arr;
}

struct futhark_u64_1d *futhark_new_u64_1d(struct futhark_context *ctx, const uint64_t *data, int64_t dim0) {
    (void)ctx;
    if (dim0 < 0) return NULL;
    struct futhark_u64_1d *arr = (struct futhark_u64_1d*)malloc(sizeof(struct futhark_u64_1d));
    if (!arr) return NULL;
    arr->shape[0] = dim0;
    if (dim0 == 0) {
        arr->data = NULL;
        return arr;
    }
    size_t bytes;
    if (size_for_elems_i64(dim0, sizeof(uint64_t), &bytes)) {
        free(arr);
        return NULL;
    }
    arr->data = (uint64_t*)malloc(bytes);
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    if (data) {
        memcpy(arr->data, data, bytes);
    } else {
        memset(arr->data, 0, bytes);
    }
    return arr;
}

struct futhark_i64_1d *futhark_new_i64_1d(struct futhark_context *ctx, const int64_t *data, int64_t dim0) {
    (void)ctx;
    if (dim0 < 0) return NULL;
    struct futhark_i64_1d *arr = (struct futhark_i64_1d*)malloc(sizeof(struct futhark_i64_1d));
    if (!arr) return NULL;
    arr->shape[0] = dim0;
    if (dim0 == 0) {
        arr->data = NULL;
        return arr;
    }
    size_t bytes;
    if (size_for_elems_i64(dim0, sizeof(int64_t), &bytes)) {
        free(arr);
        return NULL;
    }
    arr->data = (int64_t*)malloc(bytes);
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    if (data) {
        memcpy(arr->data, data, bytes);
    } else {
        memset(arr->data, 0, bytes);
    }
    return arr;
}

void futhark_free_f32_1d(struct futhark_context *ctx, struct futhark_f32_1d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

void futhark_free_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

void futhark_free_f32_3d(struct futhark_context *ctx, struct futhark_f32_3d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

void futhark_free_u64_1d(struct futhark_context *ctx, struct futhark_u64_1d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

void futhark_free_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

int futhark_values_f32_1d(struct futhark_context *ctx, struct futhark_f32_1d *arr, float *data) {
    (void)ctx;
    if (!arr || !data) return 1;
    if (arr->shape[0] < 0) return 1;
    if (arr->shape[0] == 0) return 0;
    if (!arr->data) return 1;
    size_t bytes;
    if (size_for_elems_i64(arr->shape[0], sizeof(float), &bytes)) return 1;
    memcpy(data, arr->data, bytes);
    return 0;
}

int futhark_values_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr, float *data) {
    (void)ctx;
    if (!arr || !data) return 1;
    if (arr->shape[0] < 0 || arr->shape[1] < 0) return 1;
    if (check_size_overflow_2(arr->shape[0], arr->shape[1])) return 1;
    int64_t total = arr->shape[0] * arr->shape[1];
    if (total == 0) return 0;
    if (!arr->data) return 1;
    size_t bytes;
    if (size_for_elems_i64(total, sizeof(float), &bytes)) return 1;
    memcpy(data, arr->data, bytes);
    return 0;
}

int futhark_values_f32_3d(struct futhark_context *ctx, struct futhark_f32_3d *arr, float *data) {
    (void)ctx;
    if (!arr || !data) return 1;
    if (arr->shape[0] < 0 || arr->shape[1] < 0 || arr->shape[2] < 0) return 1;
    if (check_size_overflow_3(arr->shape[0], arr->shape[1], arr->shape[2])) return 1;
    int64_t total = arr->shape[0] * arr->shape[1] * arr->shape[2];
    if (total == 0) return 0;
    if (!arr->data) return 1;
    size_t bytes;
    if (size_for_elems_i64(total, sizeof(float), &bytes)) return 1;
    memcpy(data, arr->data, bytes);
    return 0;
}

int futhark_values_u64_1d(struct futhark_context *ctx, struct futhark_u64_1d *arr, uint64_t *data) {
    (void)ctx;
    if (!arr || !data) return 1;
    if (arr->shape[0] < 0) return 1;
    if (arr->shape[0] == 0) return 0;
    if (!arr->data) return 1;
    size_t bytes;
    if (size_for_elems_i64(arr->shape[0], sizeof(uint64_t), &bytes)) return 1;
    memcpy(data, arr->data, bytes);
    return 0;
}

int futhark_values_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr, int64_t *data) {
    (void)ctx;
    if (!arr || !data) return 1;
    if (arr->shape[0] < 0) return 1;
    if (arr->shape[0] == 0) return 0;
    if (!arr->data) return 1;
    size_t bytes;
    if (size_for_elems_i64(arr->shape[0], sizeof(int64_t), &bytes)) return 1;
    memcpy(data, arr->data, bytes);
    return 0;
}

int futhark_entry_matmul(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_f32_2d *a, const struct futhark_f32_2d *b) {
    (void)ctx;
    if (!a || !b || !out) return 1;
    if (a->shape[0] < 0 || a->shape[1] < 0 || b->shape[0] < 0 || b->shape[1] < 0) return 1;
    if (!a->data || !b->data) {
        if (a->shape[0] == 0 || a->shape[1] == 0 || b->shape[0] == 0 || b->shape[1] == 0) {
        } else {
            return 1;
        }
    }

    int64_t m = a->shape[0];
    int64_t k = a->shape[1];
    int64_t n = b->shape[1];

    if (k != b->shape[0]) return 1;
    if (check_size_overflow_2(m, n)) return 1;

    *out = (struct futhark_f32_2d*)malloc(sizeof(struct futhark_f32_2d));
    if (!*out) return 1;

    (*out)->shape[0] = m;
    (*out)->shape[1] = n;

    int64_t total = m * n;
    if (total == 0) {
        (*out)->data = NULL;
        return 0;
    }

    size_t bytes;
    if (size_for_elems_i64(total, sizeof(float), &bytes)) {
        free(*out);
        *out = NULL;
        return 1;
    }

    (*out)->data = (float*)calloc((size_t)total, sizeof(float));
    if (!(*out)->data) {
        free(*out);
        *out = NULL;
        return 1;
    }

    for (int64_t i = 0; i < m; i++) {
        for (int64_t kk = 0; kk < k; kk++) {
            float a_ik = a->data[i * k + kk];
            for (int64_t j = 0; j < n; j++) {
                (*out)->data[i * n + j] += a_ik * b->data[kk * n + j];
            }
        }
    }

    (void)bytes;
    return 0;
}

int futhark_entry_batch_matmul(struct futhark_context *ctx, struct futhark_f32_3d **out, const struct futhark_f32_3d *a, const struct futhark_f32_3d *c) {
    (void)ctx;
    if (!a || !c || !out) return 1;
    if (a->shape[0] < 0 || a->shape[1] < 0 || a->shape[2] < 0 || c->shape[0] < 0 || c->shape[1] < 0 || c->shape[2] < 0) return 1;
    if (!a->data || !c->data) {
        if (a->shape[0] == 0 || a->shape[1] == 0 || a->shape[2] == 0 || c->shape[0] == 0 || c->shape[1] == 0 || c->shape[2] == 0) {
        } else {
            return 1;
        }
    }

    int64_t batch = a->shape[0];
    int64_t m = a->shape[1];
    int64_t k = a->shape[2];
    int64_t n = c->shape[2];

    if (batch != c->shape[0] || k != c->shape[1]) return 1;
    if (check_size_overflow_3(batch, m, n)) return 1;

    *out = (struct futhark_f32_3d*)malloc(sizeof(struct futhark_f32_3d));
    if (!*out) return 1;

    (*out)->shape[0] = batch;
    (*out)->shape[1] = m;
    (*out)->shape[2] = n;

    int64_t total = batch * m * n;
    if (total == 0) {
        (*out)->data = NULL;
        return 0;
    }

    size_t bytes;
    if (size_for_elems_i64(total, sizeof(float), &bytes)) {
        free(*out);
        *out = NULL;
        return 1;
    }

    (*out)->data = (float*)calloc((size_t)total, sizeof(float));
    if (!(*out)->data) {
        free(*out);
        *out = NULL;
        return 1;
    }

    for (int64_t bb = 0; bb < batch; bb++) {
        for (int64_t i = 0; i < m; i++) {
            for (int64_t kk = 0; kk < k; kk++) {
                float a_val = a->data[bb * m * k + i * k + kk];
                for (int64_t j = 0; j < n; j++) {
                    (*out)->data[bb * m * n + i * n + j] += a_val * c->data[bb * k * n + kk * n + j];
                }
            }
        }
    }

    (void)bytes;
    return 0;
}

int futhark_entry_dot(struct futhark_context *ctx, float *out, const struct futhark_f32_1d *a, const struct futhark_f32_1d *b) {
    (void)ctx;
    if (!a || !b || !out) return 1;
    if (a->shape[0] < 0 || b->shape[0] < 0) return 1;
    if (a->shape[0] != b->shape[0]) return 1;
    if (a->shape[0] == 0) {
        *out = 0.0f;
        return 0;
    }
    if (!a->data || !b->data) return 1;

    float sum = 0.0f;
    for (int64_t i = 0; i < a->shape[0]; i++) {
        sum += a->data[i] * b->data[i];
    }
    *out = sum;

    return 0;
}

int futhark_entry_apply_softmax(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_f32_1d *x) {
    (void)ctx;
    if (!x || !out) return 1;
    if (x->shape[0] < 0) return 1;

    int64_t n = x->shape[0];
    if (n == 0) {
        *out = (struct futhark_f32_1d*)malloc(sizeof(struct futhark_f32_1d));
        if (!*out) return 1;
        (*out)->shape[0] = 0;
        (*out)->data = NULL;
        return 0;
    }
    if (!x->data) return 1;

    *out = (struct futhark_f32_1d*)malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;

    (*out)->shape[0] = n;
    size_t bytes;
    if (size_for_elems_i64(n, sizeof(float), &bytes)) {
        free(*out);
        *out = NULL;
        return 1;
    }
    (*out)->data = (float*)malloc(bytes);
    if (!(*out)->data) {
        free(*out);
        *out = NULL;
        return 1;
    }

    float max_val = -INFINITY;
    int valid = 0;
    for (int64_t i = 0; i < n; i++) {
        float xi = x->data[i];
        if (isfinite(xi)) {
            if (!valid || xi > max_val) max_val = xi;
            valid = 1;
        }
    }

    if (!valid) {
        float uniform = 1.0f / (float)n;
        for (int64_t i = 0; i < n; i++) (*out)->data[i] = uniform;
        return 0;
    }

    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float xi = x->data[i];
        float ev;
        if (!isfinite(xi)) {
            ev = 0.0f;
        } else {
            float t = xi - max_val;
            ev = expf(t);
            if (!isfinite(ev)) ev = 0.0f;
        }
        (*out)->data[i] = ev;
        sum += ev;
    }

    if (!(sum > 0.0f) || !isfinite(sum)) {
        float uniform = 1.0f / (float)n;
        for (int64_t i = 0; i < n; i++) (*out)->data[i] = uniform;
        return 0;
    }

    float inv = 1.0f / sum;
    for (int64_t i = 0; i < n; i++) {
        (*out)->data[i] *= inv;
    }

    return 0;
}

int futhark_entry_apply_layer_norm(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_f32_1d *x, const struct futhark_f32_1d *gamma, const struct futhark_f32_1d *beta, float eps) {
    (void)ctx;
    if (!x || !gamma || !beta || !out) return 1;
    if (x->shape[0] < 0 || gamma->shape[0] < 0 || beta->shape[0] < 0) return 1;
    if (!(eps > 0.0f)) eps = 1e-5f;

    int64_t n = x->shape[0];
    if (gamma->shape[0] != n || beta->shape[0] != n) return 1;

    if (n == 0) {
        *out = (struct futhark_f32_1d*)malloc(sizeof(struct futhark_f32_1d));
        if (!*out) return 1;
        (*out)->shape[0] = 0;
        (*out)->data = NULL;
        return 0;
    }
    if (!x->data || !gamma->data || !beta->data) return 1;

    *out = (struct futhark_f32_1d*)malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;

    (*out)->shape[0] = n;
    size_t bytes;
    if (size_for_elems_i64(n, sizeof(float), &bytes)) {
        free(*out);
        *out = NULL;
        return 1;
    }
    (*out)->data = (float*)malloc(bytes);
    if (!(*out)->data) {
        free(*out);
        *out = NULL;
        return 1;
    }

    float mean = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        mean += x->data[i];
    }
    mean /= (float)n;

    float variance = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float diff = x->data[i] - mean;
        variance += diff * diff;
    }
    variance /= (float)n;

    float denom = sqrtf(variance + eps);
    float inv = (denom > 0.0f && isfinite(denom)) ? (1.0f / denom) : 0.0f;

    for (int64_t i = 0; i < n; i++) {
        (*out)->data[i] = gamma->data[i] * ((x->data[i] - mean) * inv) + beta->data[i];
    }

    return 0;
}

int futhark_entry_apply_relu(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_f32_1d *x) {
    (void)ctx;
    if (!x || !out) return 1;
    if (x->shape[0] < 0) return 1;

    int64_t n = x->shape[0];
    if (n == 0) {
        *out = (struct futhark_f32_1d*)malloc(sizeof(struct futhark_f32_1d));
        if (!*out) return 1;
        (*out)->shape[0] = 0;
        (*out)->data = NULL;
        return 0;
    }
    if (!x->data) return 1;

    *out = (struct futhark_f32_1d*)malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;

    (*out)->shape[0] = n;
    size_t bytes;
    if (size_for_elems_i64(n, sizeof(float), &bytes)) {
        free(*out);
        *out = NULL;
        return 1;
    }
    (*out)->data = (float*)malloc(bytes);
    if (!(*out)->data) {
        free(*out);
        *out = NULL;
        return 1;
    }

    for (int64_t i = 0; i < n; i++) {
        float xi = x->data[i];
        (*out)->data[i] = xi > 0.0f ? xi : 0.0f;
    }

    return 0;
}

int futhark_entry_apply_gelu(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_f32_1d *x) {
    (void)ctx;
    if (!x || !out) return 1;
    if (x->shape[0] < 0) return 1;

    int64_t n = x->shape[0];
    const float sqrt_2_over_pi = 0.7978845608f;

    if (n == 0) {
        *out = (struct futhark_f32_1d*)malloc(sizeof(struct futhark_f32_1d));
        if (!*out) return 1;
        (*out)->shape[0] = 0;
        (*out)->data = NULL;
        return 0;
    }
    if (!x->data) return 1;

    *out = (struct futhark_f32_1d*)malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;

    (*out)->shape[0] = n;
    size_t bytes;
    if (size_for_elems_i64(n, sizeof(float), &bytes)) {
        free(*out);
        *out = NULL;
        return 1;
    }
    (*out)->data = (float*)malloc(bytes);
    if (!(*out)->data) {
        free(*out);
        *out = NULL;
        return 1;
    }

    for (int64_t i = 0; i < n; i++) {
        float xi = x->data[i];
        float cdf = 0.5f * (1.0f + tanhf(sqrt_2_over_pi * (xi + 0.044715f * xi * xi * xi)));
        (*out)->data[i] = xi * cdf;
    }

    return 0;
}

int futhark_entry_clip_fisher(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_f32_1d *fisher, float clip_val) {
    (void)ctx;
    if (!fisher || !out) return 1;
    if (fisher->shape[0] < 0) return 1;

    int64_t n = fisher->shape[0];

    if (n == 0) {
        *out = (struct futhark_f32_1d*)malloc(sizeof(struct futhark_f32_1d));
        if (!*out) return 1;
        (*out)->shape[0] = 0;
        (*out)->data = NULL;
        return 0;
    }
    if (!fisher->data) return 1;

    *out = (struct futhark_f32_1d*)malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;

    (*out)->shape[0] = n;
    size_t bytes;
    if (size_for_elems_i64(n, sizeof(float), &bytes)) {
        free(*out);
        *out = NULL;
        return 1;
    }
    (*out)->data = (float*)malloc(bytes);
    if (!(*out)->data) {
        free(*out);
        *out = NULL;
        return 1;
    }

    for (int64_t i = 0; i < n; i++) {
        float v = fisher->data[i];
        (*out)->data[i] = v < clip_val ? v : clip_val;
    }

    return 0;
}

int futhark_entry_reduce_gradients(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_f32_2d *gradients) {
    (void)ctx;
    if (!gradients || !out) return 1;
    if (gradients->shape[0] < 0 || gradients->shape[1] < 0) return 1;
    if (check_size_overflow_2(gradients->shape[0], gradients->shape[1])) return 1;

    int64_t batch = gradients->shape[0];
    int64_t n = gradients->shape[1];

    if (n == 0) {
        *out = (struct futhark_f32_1d*)malloc(sizeof(struct futhark_f32_1d));
        if (!*out) return 1;
        (*out)->shape[0] = 0;
        (*out)->data = NULL;
        return 0;
    }
    if (batch > 0 && !gradients->data) return 1;

    *out = (struct futhark_f32_1d*)malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;

    (*out)->shape[0] = n;
    size_t bytes;
    if (size_for_elems_i64(n, sizeof(float), &bytes)) {
        free(*out);
        *out = NULL;
        return 1;
    }
    (*out)->data = (float*)calloc((size_t)n, sizeof(float));
    if (!(*out)->data) {
        free(*out);
        *out = NULL;
        return 1;
    }

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < n; i++) {
            (*out)->data[i] += gradients->data[b * n + i];
        }
    }

    (void)bytes;
    return 0;
}

int futhark_entry_rank_segments(struct futhark_context *ctx, struct futhark_f32_1d **out, uint64_t query_hash, const struct futhark_u64_1d *segment_hashes, const struct futhark_f32_1d *base_scores) {
    (void)ctx;
    if (!segment_hashes || !base_scores || !out) return 1;
    if (segment_hashes->shape[0] < 0 || base_scores->shape[0] < 0) return 1;

    int64_t n = segment_hashes->shape[0];
    if (base_scores->shape[0] != n) return 1;

    if (n == 0) {
        *out = (struct futhark_f32_1d*)malloc(sizeof(struct futhark_f32_1d));
        if (!*out) return 1;
        (*out)->shape[0] = 0;
        (*out)->data = NULL;
        return 0;
    }
    if (!segment_hashes->data || !base_scores->data) return 1;

    *out = (struct futhark_f32_1d*)malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;

    (*out)->shape[0] = n;
    size_t bytes;
    if (size_for_elems_i64(n, sizeof(float), &bytes)) {
        free(*out);
        *out = NULL;
        return 1;
    }
    (*out)->data = (float*)malloc(bytes);
    if (!(*out)->data) {
        free(*out);
        *out = NULL;
        return 1;
    }

    for (int64_t i = 0; i < n; i++) {
        float match_bonus = (segment_hashes->data[i] == query_hash) ? 1.0f : 0.0f;
        (*out)->data[i] = base_scores->data[i] + match_bonus;
    }

    return 0;
}

static int matmul_f32_nn(const float *a, const float *b, float *c, int64_t m, int64_t k, int64_t n) {
    if (m < 0 || k < 0 || n < 0) return 1;
    if (m == 0 || k == 0 || n == 0) return 0;
    if (!a || !b || !c) return 1;
    if (check_size_overflow_2(m, k)) return 1;
    if (check_size_overflow_2(k, n)) return 1;
    if (check_size_overflow_2(m, n)) return 1;

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            c[i * n + j] = 0.0f;
        }
    }

    for (int64_t i = 0; i < m; i++) {
        for (int64_t kk = 0; kk < k; kk++) {
            float av = a[i * k + kk];
            for (int64_t j = 0; j < n; j++) {
                c[i * n + j] += av * b[kk * n + j];
            }
        }
    }
    return 0;
}

static int matmul_f32_nt(const float *a, const float *b, float *c, int64_t m, int64_t k, int64_t n) {
    if (m < 0 || k < 0 || n < 0) return 1;
    if (m == 0 || k == 0 || n == 0) return 0;
    if (!a || !b || !c) return 1;
    if (check_size_overflow_2(m, k)) return 1;
    if (check_size_overflow_2(n, k)) return 1;
    if (check_size_overflow_2(m, n)) return 1;

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            c[i * n + j] = 0.0f;
        }
    }

    for (int64_t i = 0; i < m; i++) {
        for (int64_t kk = 0; kk < k; kk++) {
            float av = a[i * k + kk];
            for (int64_t j = 0; j < n; j++) {
                c[i * n + j] += av * b[j * k + kk];
            }
        }
    }
    return 0;
}

static int matmul_f32_tn(const float *a, const float *b, float *c, int64_t m, int64_t k, int64_t n) {
    if (m < 0 || k < 0 || n < 0) return 1;
    if (m == 0 || k == 0 || n == 0) return 0;
    if (!a || !b || !c) return 1;
    if (check_size_overflow_2(m, k)) return 1;
    if (check_size_overflow_2(m, n)) return 1;
    if (check_size_overflow_2(k, n)) return 1;

    for (int64_t i = 0; i < k; i++) {
        for (int64_t j = 0; j < n; j++) {
            c[i * n + j] = 0.0f;
        }
    }

    for (int64_t i = 0; i < m; i++) {
        for (int64_t kk = 0; kk < k; kk++) {
            float av = a[i * k + kk];
            for (int64_t j = 0; j < n; j++) {
                c[kk * n + j] += av * b[i * n + j];
            }
        }
    }
    return 0;
}

static int layer_norm_forward_row(float *y, const float *x, int64_t d, float eps, float *out_inv_std) {
    if (d < 0) return 1;
    if (d == 0) {
        if (out_inv_std) *out_inv_std = 0.0f;
        return 0;
    }
    float mean = 0.0f;
    for (int64_t j = 0; j < d; j++) mean += x[j];
    mean /= (float)d;

    float var = 0.0f;
    for (int64_t j = 0; j < d; j++) {
        float t = x[j] - mean;
        var += t * t;
    }
    var /= (float)d;

    float denom = sqrtf(var + eps);
    float inv = (denom > 0.0f && isfinite(denom)) ? (1.0f / denom) : 0.0f;
    if (out_inv_std) *out_inv_std = inv;

    for (int64_t j = 0; j < d; j++) {
        y[j] = (x[j] - mean) * inv;
    }
    return 0;
}

static int layer_norm_backward_row(float *dx, const float *dy, const float *y, int64_t d, float inv_std) {
    if (d < 0) return 1;
    if (d == 0) return 0;
    if (inv_std == 0.0f || !isfinite(inv_std)) {
        for (int64_t j = 0; j < d; j++) dx[j] = 0.0f;
        return 0;
    }

    float mean_dy = 0.0f;
    float mean_dy_y = 0.0f;
    for (int64_t j = 0; j < d; j++) {
        mean_dy += dy[j];
        mean_dy_y += dy[j] * y[j];
    }
    mean_dy /= (float)d;
    mean_dy_y /= (float)d;

    for (int64_t j = 0; j < d; j++) {
        dx[j] = inv_std * (dy[j] - mean_dy - y[j] * mean_dy_y);
    }
    return 0;
}

static int f16_matmul(const f16 *a, const f16 *b, f16 *c, int64_t m, int64_t k, int64_t n) {
    if (m < 0 || k < 0 || n < 0) return 1;
    if (m == 0 || k == 0 || n == 0) return 0;
    if (!a || !b || !c) return 1;
    if (check_size_overflow_2(m, n)) return 1;
    if (check_size_overflow_2(m, k)) return 1;
    if (check_size_overflow_2(k, n)) return 1;

    int64_t mn = m * n;
    size_t bytes;
    if (size_for_elems_i64(mn, sizeof(float), &bytes)) return 1;

    float *c_f32 = (float*)calloc((size_t)mn, sizeof(float));
    if (!c_f32) return 1;

    for (int64_t i = 0; i < m; i++) {
        for (int64_t kk = 0; kk < k; kk++) {
            float a_val = f16_to_f32(a[i * k + kk]);
            for (int64_t j = 0; j < n; j++) {
                c_f32[i * n + j] += a_val * f16_to_f32(b[kk * n + j]);
            }
        }
    }

    for (int64_t i = 0; i < mn; i++) {
        c[i] = f32_to_f16(c_f32[i]);
    }

    free(c_f32);
    (void)bytes;
    return 0;
}

static int f16_layer_norm_2d(f16 *data, int64_t rows, int64_t cols, float eps) {
    if (rows < 0 || cols < 0) return 1;
    if (rows == 0 || cols == 0) return 0;
    if (!data) return 1;
    if (!(eps > 0.0f)) eps = 1e-5f;

    for (int64_t i = 0; i < rows; i++) {
        float mean = 0.0f;
        for (int64_t j = 0; j < cols; j++) {
            mean += f16_to_f32(data[i * cols + j]);
        }
        mean /= (float)cols;

        float variance = 0.0f;
        for (int64_t j = 0; j < cols; j++) {
            float diff = f16_to_f32(data[i * cols + j]) - mean;
            variance += diff * diff;
        }
        variance /= (float)cols;

        float denom = sqrtf(variance + eps);
        float inv = (denom > 0.0f && isfinite(denom)) ? (1.0f / denom) : 0.0f;

        for (int64_t j = 0; j < cols; j++) {
            float v = f16_to_f32(data[i * cols + j]);
            data[i * cols + j] = f32_to_f16((v - mean) * inv);
        }
    }
    return 0;
}

int futhark_entry_rsf_forward(struct futhark_context *ctx,
                               struct futhark_f16_2d **out,
                               struct futhark_f16_2d *input,
                               struct futhark_f16_2d *weights_s,
                               struct futhark_f16_2d *weights_t) {
    (void)ctx;
    if (!input || !weights_s || !weights_t || !out) return 1;
    if (input->shape[0] < 0 || input->shape[1] < 0) return 1;
    if (weights_s->shape[0] < 0 || weights_s->shape[1] < 0) return 1;
    if (weights_t->shape[0] < 0 || weights_t->shape[1] < 0) return 1;

    int64_t n = input->shape[0];
    int64_t d = input->shape[1];

    if (weights_s->shape[0] != d || weights_s->shape[1] != d) return 1;
    if (weights_t->shape[0] != d || weights_t->shape[1] != d) return 1;
    if (check_size_overflow_2(n, d)) return 1;

    *out = (struct futhark_f16_2d*)malloc(sizeof(struct futhark_f16_2d));
    if (!*out) return 1;

    (*out)->shape[0] = n;
    (*out)->shape[1] = d;

    int64_t total = n * d;
    if (total == 0) {
        (*out)->data = NULL;
        return 0;
    }

    size_t bytes;
    if (size_for_elems_i64(total, sizeof(f16), &bytes)) {
        free(*out);
        *out = NULL;
        return 1;
    }

    if (!input->data || !weights_s->data || !weights_t->data) {
        free(*out);
        *out = NULL;
        return 1;
    }

    (*out)->data = (f16*)malloc(bytes);
    if (!(*out)->data) {
        free(*out);
        *out = NULL;
        return 1;
    }

    f16 *temp1 = (f16*)malloc(bytes);
    f16 *temp2 = (f16*)malloc(bytes);

    if (!temp1 || !temp2) {
        free(temp1);
        free(temp2);
        free((*out)->data);
        free(*out);
        *out = NULL;
        return 1;
    }

    if (f16_matmul(input->data, weights_s->data, temp1, n, d, d) != 0) {
        free(temp1);
        free(temp2);
        free((*out)->data);
        free(*out);
        *out = NULL;
        return 1;
    }
    if (f16_layer_norm_2d(temp1, n, d, 1e-5f) != 0) {
        free(temp1);
        free(temp2);
        free((*out)->data);
        free(*out);
        *out = NULL;
        return 1;
    }
    if (f16_matmul(temp1, weights_t->data, temp2, n, d, d) != 0) {
        free(temp1);
        free(temp2);
        free((*out)->data);
        free(*out);
        *out = NULL;
        return 1;
    }
    if (f16_layer_norm_2d(temp2, n, d, 1e-5f) != 0) {
        free(temp1);
        free(temp2);
        free((*out)->data);
        free(*out);
        *out = NULL;
        return 1;
    }

    for (int64_t i = 0; i < total; i++) {
        float v = f16_to_f32(temp2[i]) + f16_to_f32(input->data[i]);
        (*out)->data[i] = f32_to_f16(v);
    }

    free(temp1);
    free(temp2);

    return 0;
}

int futhark_entry_rsf_backward(struct futhark_context *ctx,
                                struct futhark_f16_2d **out,
                                struct futhark_f16_2d *grad_output,
                                struct futhark_f16_2d *weights) {
    (void)ctx;
    if (!grad_output || !weights || !out) return 1;
    if (grad_output->shape[0] < 0 || grad_output->shape[1] < 0) return 1;
    if (weights->shape[0] < 0 || weights->shape[1] < 0) return 1;

    int64_t n = grad_output->shape[0];
    int64_t d = grad_output->shape[1];

    if (weights->shape[0] != d || weights->shape[1] != d) return 1;
    if (check_size_overflow_2(n, d)) return 1;

    *out = (struct futhark_f16_2d*)malloc(sizeof(struct futhark_f16_2d));
    if (!*out) return 1;

    (*out)->shape[0] = n;
    (*out)->shape[1] = d;

    int64_t total = n * d;
    if (total == 0) {
        (*out)->data = NULL;
        return 0;
    }

    size_t bytes;
    if (size_for_elems_i64(total, sizeof(f16), &bytes)) {
        free(*out);
        *out = NULL;
        return 1;
    }

    if (!grad_output->data || !weights->data) {
        free(*out);
        *out = NULL;
        return 1;
    }

    (*out)->data = (f16*)malloc(bytes);
    if (!(*out)->data) {
        free(*out);
        *out = NULL;
        return 1;
    }

    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < d; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < d; k++) {
                float gv = f16_to_f32(grad_output->data[i * d + k]);
                float wv = f16_to_f32(weights->data[k * d + j]);
                sum += gv * wv;
            }
            (*out)->data[i * d + j] = f32_to_f16(sum);
        }
    }

    return 0;
}

int futhark_entry_scale_weights_inplace(struct futhark_context *ctx,
                                         struct futhark_f16_2d *weights,
                                         float scale) {
    (void)ctx;
    if (!weights) return 1;
    if (weights->shape[0] < 0 || weights->shape[1] < 0) return 1;
    if (!weights->data) {
        if (weights->shape[0] == 0 || weights->shape[1] == 0) return 0;
        return 1;
    }

    int64_t total = weights->shape[0] * weights->shape[1];
    if (total == 0) return 0;

    for (int64_t i = 0; i < total; i++) {
        float v = f16_to_f32(weights->data[i]);
        weights->data[i] = f32_to_f16(v * scale);
    }

    return 0;
}

static int alloc_f16_2d(struct futhark_f16_2d **arr, int64_t dim0, int64_t dim1) {
    if (!arr) return 1;
    if (dim0 < 0 || dim1 < 0) return 1;
    if (check_size_overflow_2(dim0, dim1)) return 1;
    *arr = (struct futhark_f16_2d*)malloc(sizeof(struct futhark_f16_2d));
    if (!*arr) return 1;
    (*arr)->shape[0] = dim0;
    (*arr)->shape[1] = dim1;
    int64_t total = dim0 * dim1;
    if (total == 0) {
        (*arr)->data = NULL;
        return 0;
    }
    size_t bytes;
    if (size_for_elems_i64(total, sizeof(f16), &bytes)) {
        free(*arr);
        *arr = NULL;
        return 1;
    }
    (*arr)->data = (f16*)malloc(bytes);
    if (!(*arr)->data) {
        free(*arr);
        *arr = NULL;
        return 1;
    }
    return 0;
}

static int copy_f16_2d(struct futhark_f16_2d *dst, const struct futhark_f16_2d *src) {
    if (!dst || !src) return 1;
    if (src->shape[0] < 0 || src->shape[1] < 0) return 1;
    if (check_size_overflow_2(src->shape[0], src->shape[1])) return 1;
    int64_t total = src->shape[0] * src->shape[1];
    if (total == 0) return 0;
    if (!dst->data || !src->data) return 1;
    size_t bytes;
    if (size_for_elems_i64(total, sizeof(f16), &bytes)) return 1;
    memcpy(dst->data, src->data, bytes);
    return 0;
}

int futhark_entry_training_step(
    struct futhark_context *ctx,
    struct futhark_f16_2d **new_weights_s,
    struct futhark_f16_2d **new_weights_t,
    struct futhark_f16_2d **new_velocity_s,
    struct futhark_f16_2d **new_velocity_t,
    f16 *loss,
    struct futhark_f16_2d *inputs,
    struct futhark_f16_2d *targets,
    struct futhark_f16_2d *weights_s,
    struct futhark_f16_2d *weights_t,
    struct futhark_f16_2d *velocity_s,
    struct futhark_f16_2d *velocity_t,
    f16 learning_rate,
    f16 momentum
) {
    (void)ctx;
    if (!inputs || !targets || !weights_s || !weights_t || !velocity_s || !velocity_t) return 1;
    if (!new_weights_s || !new_weights_t || !new_velocity_s || !new_velocity_t || !loss) return 1;

    if (inputs->shape[0] < 0 || inputs->shape[1] < 0) return 1;
    if (targets->shape[0] < 0 || targets->shape[1] < 0) return 1;
    if (weights_s->shape[0] < 0 || weights_s->shape[1] < 0) return 1;
    if (weights_t->shape[0] < 0 || weights_t->shape[1] < 0) return 1;
    if (velocity_s->shape[0] < 0 || velocity_s->shape[1] < 0) return 1;
    if (velocity_t->shape[0] < 0 || velocity_t->shape[1] < 0) return 1;

    int64_t n = inputs->shape[0];
    int64_t d = inputs->shape[1];

    if (targets->shape[0] != n || targets->shape[1] != d) return 1;
    if (weights_s->shape[0] != d || weights_s->shape[1] != d) return 1;
    if (weights_t->shape[0] != d || weights_t->shape[1] != d) return 1;
    if (velocity_s->shape[0] != d || velocity_s->shape[1] != d) return 1;
    if (velocity_t->shape[0] != d || velocity_t->shape[1] != d) return 1;

    if (check_size_overflow_2(n, d)) return 1;
    if (check_size_overflow_2(d, d)) return 1;

    int64_t total_elements = n * d;
    int64_t weight_size = d * d;

    if ((n == 0 || d == 0) || total_elements == 0 || weight_size == 0) {
        *loss = f32_to_f16(0.0f);

        if (alloc_f16_2d(new_weights_s, d, d) != 0) return 1;
        if (alloc_f16_2d(new_weights_t, d, d) != 0) return 1;
        if (alloc_f16_2d(new_velocity_s, d, d) != 0) return 1;
        if (alloc_f16_2d(new_velocity_t, d, d) != 0) return 1;

        if (copy_f16_2d(*new_weights_s, weights_s) != 0) return 1;
        if (copy_f16_2d(*new_weights_t, weights_t) != 0) return 1;
        if (copy_f16_2d(*new_velocity_s, velocity_s) != 0) return 1;
        if (copy_f16_2d(*new_velocity_t, velocity_t) != 0) return 1;

        return 0;
    }

    if (!inputs->data || !targets->data || !weights_s->data || !weights_t->data || !velocity_s->data || !velocity_t->data) return 1;

    float lr = f16_to_f32(learning_rate);
    float mom = f16_to_f32(momentum);

    size_t bytes_x, bytes_w, bytes_y;
    if (size_for_elems_i64(total_elements, sizeof(float), &bytes_x)) return 1;
    if (size_for_elems_i64(weight_size, sizeof(float), &bytes_w)) return 1;
    if (size_for_elems_i64(total_elements, sizeof(float), &bytes_y)) return 1;

    float *X = (float*)malloc(bytes_x);
    float *T = (float*)malloc(bytes_x);
    float *Ws = (float*)malloc(bytes_w);
    float *Wt = (float*)malloc(bytes_w);
    float *V_s = (float*)malloc(bytes_w);
    float *V_t = (float*)malloc(bytes_w);

    float *Z1 = (float*)malloc(bytes_y);
    float *Y1 = (float*)malloc(bytes_y);
    float *Z2 = (float*)malloc(bytes_y);
    float *Y2 = (float*)malloc(bytes_y);
    float *O = (float*)malloc(bytes_y);

    float *invstd1 = (float*)malloc((size_t)n * sizeof(float));
    float *invstd2 = (float*)malloc((size_t)n * sizeof(float));

    float *dO = (float*)malloc(bytes_y);
    float *dY2 = (float*)malloc(bytes_y);
    float *dZ2 = (float*)malloc(bytes_y);
    float *dY1 = (float*)malloc(bytes_y);
    float *dZ1 = (float*)malloc(bytes_y);

    float *dWs = (float*)malloc(bytes_w);
    float *dWt = (float*)malloc(bytes_w);

    float *tmp_dY1 = (float*)malloc(bytes_y);

    if (!X || !T || !Ws || !Wt || !V_s || !V_t || !Z1 || !Y1 || !Z2 || !Y2 || !O || !invstd1 || !invstd2 ||
        !dO || !dY2 || !dZ2 || !dY1 || !dZ1 || !dWs || !dWt || !tmp_dY1) {
        free(X); free(T); free(Ws); free(Wt); free(V_s); free(V_t);
        free(Z1); free(Y1); free(Z2); free(Y2); free(O);
        free(invstd1); free(invstd2);
        free(dO); free(dY2); free(dZ2); free(dY1); free(dZ1);
        free(dWs); free(dWt); free(tmp_dY1);
        return 1;
    }

    for (int64_t i = 0; i < total_elements; i++) {
        X[i] = f16_to_f32(inputs->data[i]);
        T[i] = f16_to_f32(targets->data[i]);
    }
    for (int64_t i = 0; i < weight_size; i++) {
        Ws[i] = f16_to_f32(weights_s->data[i]);
        Wt[i] = f16_to_f32(weights_t->data[i]);
        V_s[i] = f16_to_f32(velocity_s->data[i]);
        V_t[i] = f16_to_f32(velocity_t->data[i]);
    }

    if (matmul_f32_nn(X, Ws, Z1, n, d, d) != 0) {
        free(X); free(T); free(Ws); free(Wt); free(V_s); free(V_t);
        free(Z1); free(Y1); free(Z2); free(Y2); free(O);
        free(invstd1); free(invstd2);
        free(dO); free(dY2); free(dZ2); free(dY1); free(dZ1);
        free(dWs); free(dWt); free(tmp_dY1);
        return 1;
    }

    for (int64_t i = 0; i < n; i++) {
        if (layer_norm_forward_row(&Y1[i * d], &Z1[i * d], d, 1e-5f, &invstd1[i]) != 0) {
            free(X); free(T); free(Ws); free(Wt); free(V_s); free(V_t);
            free(Z1); free(Y1); free(Z2); free(Y2); free(O);
            free(invstd1); free(invstd2);
            free(dO); free(dY2); free(dZ2); free(dY1); free(dZ1);
            free(dWs); free(dWt); free(tmp_dY1);
            return 1;
        }
    }

    if (matmul_f32_nn(Y1, Wt, Z2, n, d, d) != 0) {
        free(X); free(T); free(Ws); free(Wt); free(V_s); free(V_t);
        free(Z1); free(Y1); free(Z2); free(Y2); free(O);
        free(invstd1); free(invstd2);
        free(dO); free(dY2); free(dZ2); free(dY1); free(dZ1);
        free(dWs); free(dWt); free(tmp_dY1);
        return 1;
    }

    for (int64_t i = 0; i < n; i++) {
        if (layer_norm_forward_row(&Y2[i * d], &Z2[i * d], d, 1e-5f, &invstd2[i]) != 0) {
            free(X); free(T); free(Ws); free(Wt); free(V_s); free(V_t);
            free(Z1); free(Y1); free(Z2); free(Y2); free(O);
            free(invstd1); free(invstd2);
            free(dO); free(dY2); free(dZ2); free(dY1); free(dZ1);
            free(dWs); free(dWt); free(tmp_dY1);
            return 1;
        }
    }

    for (int64_t i = 0; i < total_elements; i++) {
        O[i] = Y2[i] + X[i];
    }

    float total_loss = 0.0f;
    for (int64_t i = 0; i < total_elements; i++) {
        float diff = O[i] - T[i];
        total_loss += diff * diff;
    }
    float mean_loss = total_loss / (float)total_elements;
    *loss = f32_to_f16(mean_loss);

    float scale = 2.0f / (float)total_elements;
    for (int64_t i = 0; i < total_elements; i++) {
        dO[i] = scale * (O[i] - T[i]);
        dY2[i] = dO[i];
    }

    for (int64_t i = 0; i < n; i++) {
        if (layer_norm_backward_row(&dZ2[i * d], &dY2[i * d], &Y2[i * d], d, invstd2[i]) != 0) {
            free(X); free(T); free(Ws); free(Wt); free(V_s); free(V_t);
            free(Z1); free(Y1); free(Z2); free(Y2); free(O);
            free(invstd1); free(invstd2);
            free(dO); free(dY2); free(dZ2); free(dY1); free(dZ1);
            free(dWs); free(dWt); free(tmp_dY1);
            return 1;
        }
    }

    if (matmul_f32_tn(Y1, dZ2, dWt, n, d, d) != 0) {
        free(X); free(T); free(Ws); free(Wt); free(V_s); free(V_t);
        free(Z1); free(Y1); free(Z2); free(Y2); free(O);
        free(invstd1); free(invstd2);
        free(dO); free(dY2); free(dZ2); free(dY1); free(dZ1);
        free(dWs); free(dWt); free(tmp_dY1);
        return 1;
    }

    if (matmul_f32_nt(dZ2, Wt, tmp_dY1, n, d, d) != 0) {
        free(X); free(T); free(Ws); free(Wt); free(V_s); free(V_t);
        free(Z1); free(Y1); free(Z2); free(Y2); free(O);
        free(invstd1); free(invstd2);
        free(dO); free(dY2); free(dZ2); free(dY1); free(dZ1);
        free(dWs); free(dWt); free(tmp_dY1);
        return 1;
    }

    memcpy(dY1, tmp_dY1, bytes_y);

    for (int64_t i = 0; i < n; i++) {
        if (layer_norm_backward_row(&dZ1[i * d], &dY1[i * d], &Y1[i * d], d, invstd1[i]) != 0) {
            free(X); free(T); free(Ws); free(Wt); free(V_s); free(V_t);
            free(Z1); free(Y1); free(Z2); free(Y2); free(O);
            free(invstd1); free(invstd2);
            free(dO); free(dY2); free(dZ2); free(dY1); free(dZ1);
            free(dWs); free(dWt); free(tmp_dY1);
            return 1;
        }
    }

    if (matmul_f32_tn(X, dZ1, dWs, n, d, d) != 0) {
        free(X); free(T); free(Ws); free(Wt); free(V_s); free(V_t);
        free(Z1); free(Y1); free(Z2); free(Y2); free(O);
        free(invstd1); free(invstd2);
        free(dO); free(dY2); free(dZ2); free(dY1); free(dZ1);
        free(dWs); free(dWt); free(tmp_dY1);
        return 1;
    }

    if (alloc_f16_2d(new_weights_s, d, d) != 0) {
        free(X); free(T); free(Ws); free(Wt); free(V_s); free(V_t);
        free(Z1); free(Y1); free(Z2); free(Y2); free(O);
        free(invstd1); free(invstd2);
        free(dO); free(dY2); free(dZ2); free(dY1); free(dZ1);
        free(dWs); free(dWt); free(tmp_dY1);
        return 1;
    }
    if (alloc_f16_2d(new_weights_t, d, d) != 0) {
        futhark_free_f16_2d(ctx, *new_weights_s);
        free(X); free(T); free(Ws); free(Wt); free(V_s); free(V_t);
        free(Z1); free(Y1); free(Z2); free(Y2); free(O);
        free(invstd1); free(invstd2);
        free(dO); free(dY2); free(dZ2); free(dY1); free(dZ1);
        free(dWs); free(dWt); free(tmp_dY1);
        return 1;
    }
    if (alloc_f16_2d(new_velocity_s, d, d) != 0) {
        futhark_free_f16_2d(ctx, *new_weights_s);
        futhark_free_f16_2d(ctx, *new_weights_t);
        free(X); free(T); free(Ws); free(Wt); free(V_s); free(V_t);
        free(Z1); free(Y1); free(Z2); free(Y2); free(O);
        free(invstd1); free(invstd2);
        free(dO); free(dY2); free(dZ2); free(dY1); free(dZ1);
        free(dWs); free(dWt); free(tmp_dY1);
        return 1;
    }
    if (alloc_f16_2d(new_velocity_t, d, d) != 0) {
        futhark_free_f16_2d(ctx, *new_weights_s);
        futhark_free_f16_2d(ctx, *new_weights_t);
        futhark_free_f16_2d(ctx, *new_velocity_s);
        free(X); free(T); free(Ws); free(Wt); free(V_s); free(V_t);
        free(Z1); free(Y1); free(Z2); free(Y2); free(O);
        free(invstd1); free(invstd2);
        free(dO); free(dY2); free(dZ2); free(dY1); free(dZ1);
        free(dWs); free(dWt); free(tmp_dY1);
        return 1;
    }

    for (int64_t i = 0; i < weight_size; i++) {
        float new_vs = mom * V_s[i] + lr * dWs[i];
        float new_vt = mom * V_t[i] + lr * dWt[i];
        float new_ws = Ws[i] - new_vs;
        float new_wt = Wt[i] - new_vt;
        (*new_velocity_s)->data[i] = f32_to_f16(new_vs);
        (*new_velocity_t)->data[i] = f32_to_f16(new_vt);
        (*new_weights_s)->data[i] = f32_to_f16(new_ws);
        (*new_weights_t)->data[i] = f32_to_f16(new_wt);
    }

    free(X); free(T); free(Ws); free(Wt); free(V_s); free(V_t);
    free(Z1); free(Y1); free(Z2); free(Y2); free(O);
    free(invstd1); free(invstd2);
    free(dO); free(dY2); free(dZ2); free(dY1); free(dZ1);
    free(dWs); free(dWt); free(tmp_dY1);

    return 0;
}