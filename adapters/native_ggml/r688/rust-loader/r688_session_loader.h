#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {
void *vf_r688_open(const char *source_gguf, const char *rank_session_binary,
                   const char *ledger_root);
int32_t vf_r688_run_layer(void *context, double *output, std::size_t output_len);
void vf_r688_close(void *context);
}
