#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

/* CPU reconstruction of the first IQ3_S lane using the archived GGML grid and
 * Strata's documented byte offsets. This is a synthetic block, not model data. */
int main(void) {
    block_iq3_s b;
    memset(&b, 0, sizeof b);
    b.d = 0x2c00; /* binary16 +1/16 */
    b.qs[0] = 2; /* 9-bit grid index 2, remaining high bit zero */
    assert(sizeof b == 110 && offsetof(block_iq3_s, qs) == 2);
    assert(offsetof(block_iq3_s, qh) == 66 && offsetof(block_iq3_s, signs) == 74 && offsetof(block_iq3_s, scales) == 106);
    assert(iq3s_grid[2] == UINT32_C(0x01010105));
    const unsigned index = b.qs[0] | ((b.qh[0] & 1u) << 8);
    const unsigned code = iq3s_grid[index] & 255u;
    const unsigned scale = 1u + 2u * (b.scales[0] & 15u);
    const int sign = (b.signs[0] & 1u) ? -1 : 1;
    assert(index == 2 && code == 5 && scale == 1 && sign == 1);
    printf("{\"block_bytes\":%zu,\"grid_index\":%u,\"grid_word\":%u,\"d_binary16_bits\":%u,\"scale_multiplier\":%u,\"first_weight_numerator\":%d,\"denominator\":16,\"control_grid_index\":0,\"control_weight_numerator\":1}\n", sizeof b,index,iq3s_grid[index],b.d,scale,sign*(int)code);
    return 0;
}
