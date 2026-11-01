cmake_minimum_required(VERSION 3.16)
project(vf_native_ggml LANGUAGES C CXX ASM)
if(NOT GGML_SOURCE_DIR OR NOT EXISTS "${GGML_SOURCE_DIR}/.git")
  message(FATAL_ERROR "GGML_SOURCE_DIR must be an isolated pinned llama.cpp checkout")
endif()
execute_process(COMMAND git -C "${GGML_SOURCE_DIR}" rev-parse HEAD OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE git_status)
if(NOT git_status EQUAL 0 OR NOT revision STREQUAL "3cf03257f219afbe7334045ff7c6a06ac68c627d")
  message(FATAL_ERROR "unexpected GGML source revision")
endif()
execute_process(COMMAND git -C "${GGML_SOURCE_DIR}" status --porcelain -- ggml OUTPUT_VARIABLE dirty RESULT_VARIABLE status_result)
if(NOT status_result EQUAL 0 OR NOT dirty STREQUAL "")
  message(FATAL_ERROR "GGML source modifications are not admitted")
endif()
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${GGML_SOURCE_DIR}/ggml" ggml EXCLUDE_FROM_ALL)
find_package(Threads REQUIRED)
add_library(vf_native_ggml STATIC provider.cpp bridge.cpp route_graph_transaction.cpp qwen2moe_route_invocation.cpp
  session_vfr3_collector.cpp ../strata_native_bridge/pipe_client.cpp)
target_compile_features(vf_native_ggml PRIVATE cxx_std_17)
target_compile_options(vf_native_ggml PRIVATE -Wall -Wextra -Werror)
target_include_directories(vf_native_ggml PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../strata_native_bridge")
target_link_libraries(vf_native_ggml PUBLIC ggml-cpu ggml-base Threads::Threads)
add_executable(vf_native_ggml_test provider_test.cpp)
target_link_libraries(vf_native_ggml_test PRIVATE vf_native_ggml)
enable_testing()
add_test(NAME native_provider COMMAND vf_native_ggml_test)

# CPU-only model-graph transaction fixture: downstream GGML nodes are gated on collector success.
add_executable(vf_route_graph_transaction_test route_graph_transaction_test.cpp)
target_link_libraries(vf_route_graph_transaction_test PRIVATE vf_native_ggml)
target_compile_options(vf_route_graph_transaction_test PRIVATE -Wall -Wextra -Werror)
add_test(NAME route_graph_transaction COMMAND vf_route_graph_transaction_test)

# Capture donor-produced expert IDs and final weights at graph execution time.
add_executable(vf_route_graph_dynamic_capture_test route_graph_dynamic_capture_test.cpp)
target_link_libraries(vf_route_graph_dynamic_capture_test PRIVATE vf_native_ggml)
target_compile_options(vf_route_graph_dynamic_capture_test PRIVATE -Wall -Wextra -Werror)
add_test(NAME route_graph_dynamic_capture COMMAND vf_route_graph_dynamic_capture_test)

# One-shot per-layer wrapper matching the pinned Qwen2MoE raw routed-expert seam.
add_executable(vf_qwen2moe_route_invocation_test qwen2moe_route_invocation_test.cpp)
target_link_libraries(vf_qwen2moe_route_invocation_test PRIVATE vf_native_ggml)
target_compile_options(vf_qwen2moe_route_invocation_test PRIVATE -Wall -Wextra -Werror)
add_test(NAME qwen2moe_route_invocation COMMAND vf_qwen2moe_route_invocation_test)

# Optional callback bridge into the pinned llama.cpp internal Qwen2MoE seam.
# This compiles only against a separately patched donor worktree; VoltaFabric
# does not vendor or install llama.cpp as its engine.
option(VF_BUILD_LLAMA_QWEN2MOE_BRIDGE "Build CPU fixture for pinned llama.cpp route hooks" OFF)
option(VF_BUILD_LLAMA_QWEN2MOE_PROCESS_UBATCH_TEST
  "Register live process_ubatch hook integration test" OFF)
if(VF_BUILD_LLAMA_QWEN2MOE_PROCESS_UBATCH_TEST AND NOT VF_BUILD_LLAMA_QWEN2MOE_BRIDGE)
  message(FATAL_ERROR "process_ubatch test requires VF_BUILD_LLAMA_QWEN2MOE_BRIDGE=ON")
endif()
option(VF_BUILD_LLAMA_QWEN2MOE_BUILD_MOE_FFN_TEST
  "Register donor build_moe_ffn Qwen2MoE CPU integration test" OFF)
if(VF_BUILD_LLAMA_QWEN2MOE_BUILD_MOE_FFN_TEST AND NOT VF_BUILD_LLAMA_QWEN2MOE_BRIDGE)
  message(FATAL_ERROR "build_moe_ffn test requires VF_BUILD_LLAMA_QWEN2MOE_BRIDGE=ON")
endif()

option(VF_BUILD_LLAMA_QWEN2MOE_GGUF_BUILD_MOE_FFN_TEST
  "Register tiny GGUF-backed donor build_moe_ffn Qwen2MoE CPU integration test" OFF)
if(VF_BUILD_LLAMA_QWEN2MOE_GGUF_BUILD_MOE_FFN_TEST AND NOT VF_BUILD_LLAMA_QWEN2MOE_BRIDGE)
  message(FATAL_ERROR "GGUF build_moe_ffn test requires VF_BUILD_LLAMA_QWEN2MOE_BRIDGE=ON")
endif()

option(VF_BUILD_LLAMA_QWEN2MOE_R659_Q8_GGUF_TEST
  "Register R659 Q8_0 GGUF donor build_moe_ffn CPU integration test" OFF)
if(VF_BUILD_LLAMA_QWEN2MOE_R659_Q8_GGUF_TEST AND NOT VF_BUILD_LLAMA_QWEN2MOE_BRIDGE)
  message(FATAL_ERROR "R659 Q8_0 GGUF test requires VF_BUILD_LLAMA_QWEN2MOE_BRIDGE=ON")
endif()

if(VF_BUILD_LLAMA_QWEN2MOE_BRIDGE)
  if(NOT VF_LLAMA_SOURCE_DIR OR NOT EXISTS "${VF_LLAMA_SOURCE_DIR}/.git")
    message(FATAL_ERROR "VF_LLAMA_SOURCE_DIR must name the isolated pinned llama.cpp worktree")
  endif()
  if(NOT VF_LLAMA_LIBRARY OR NOT EXISTS "${VF_LLAMA_LIBRARY}")
    message(FATAL_ERROR "VF_LLAMA_LIBRARY must name the built pinned llama.cpp shared library")
  endif()
  execute_process(COMMAND git -C "${VF_LLAMA_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE vf_llama_revision OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE vf_llama_status)
  if(NOT vf_llama_status EQUAL 0 OR
     NOT vf_llama_revision STREQUAL "3cf03257f219afbe7334045ff7c6a06ac68c627d")
    message(FATAL_ERROR "unexpected llama.cpp source revision")
  endif()
  execute_process(COMMAND git -C "${VF_LLAMA_SOURCE_DIR}" status --porcelain -- ggml
    OUTPUT_VARIABLE vf_llama_ggml_dirty RESULT_VARIABLE vf_llama_status)
  if(NOT vf_llama_status EQUAL 0 OR NOT vf_llama_ggml_dirty STREQUAL "")
    message(FATAL_ERROR "llama.cpp GGML dependency modifications are not admitted")
  endif()
  file(READ "${VF_LLAMA_SOURCE_DIR}/src/llama-context.h" vf_llama_context_header)
  file(READ "${VF_LLAMA_SOURCE_DIR}/src/llama-graph.h" vf_llama_graph_header)
  string(FIND "${vf_llama_context_header}" "get_vf_route_hooks() const" vf_llama_getter_pos)
  string(FIND "${vf_llama_graph_header}" "struct llm_vf_route_hooks" vf_llama_hooks_pos)
  if(vf_llama_getter_pos EQUAL -1 OR vf_llama_hooks_pos EQUAL -1)
    message(FATAL_ERROR "llama.cpp worktree lacks the R679 route-hook adaptation")
  endif()

  add_library(vf_qwen2moe_llama_route_bridge STATIC qwen2moe_llama_route_bridge.cpp)
  target_compile_features(vf_qwen2moe_llama_route_bridge PUBLIC cxx_std_17)
  target_compile_options(vf_qwen2moe_llama_route_bridge PRIVATE -Wall -Wextra -Werror)
  target_include_directories(vf_qwen2moe_llama_route_bridge PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${VF_LLAMA_SOURCE_DIR}/src"
    "${VF_LLAMA_SOURCE_DIR}/include")
  target_link_libraries(vf_qwen2moe_llama_route_bridge PUBLIC vf_native_ggml "${VF_LLAMA_LIBRARY}")
  get_filename_component(vf_llama_library_dir "${VF_LLAMA_LIBRARY}" DIRECTORY)

  add_executable(vf_qwen2moe_llama_route_bridge_test qwen2moe_llama_route_bridge_test.cpp)
  target_compile_options(vf_qwen2moe_llama_route_bridge_test PRIVATE -Wall -Wextra -Werror)
  target_link_libraries(vf_qwen2moe_llama_route_bridge_test PRIVATE vf_qwen2moe_llama_route_bridge)
  set_target_properties(vf_qwen2moe_llama_route_bridge_test PROPERTIES
    BUILD_RPATH "${vf_llama_library_dir}")
  add_test(NAME qwen2moe_llama_route_bridge COMMAND vf_qwen2moe_llama_route_bridge_test)

  add_executable(vf_qwen2moe_llama_context_attach_test qwen2moe_llama_context_attach_test.cpp)
  target_compile_options(vf_qwen2moe_llama_context_attach_test PRIVATE -Wall -Wextra -Werror)
  target_link_libraries(vf_qwen2moe_llama_context_attach_test PRIVATE vf_qwen2moe_llama_route_bridge)
  set_target_properties(vf_qwen2moe_llama_context_attach_test PROPERTIES
    BUILD_RPATH "${vf_llama_library_dir}")
  add_test(NAME qwen2moe_llama_context_attach COMMAND vf_qwen2moe_llama_context_attach_test)

  if(VF_BUILD_LLAMA_QWEN2MOE_PROCESS_UBATCH_TEST)
    add_test(NAME qwen2moe_llama_process_ubatch
      COMMAND "${CMAKE_COMMAND}" -E env
        "VF_LLAMA_SOURCE_DIR=${VF_LLAMA_SOURCE_DIR}"
        "VF_LLAMA_LIBRARY=${VF_LLAMA_LIBRARY}"
        "VF_TEST_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/qwen2moe_process_ubatch"
        "CXX=${CMAKE_CXX_COMPILER}"
        "${CMAKE_CURRENT_SOURCE_DIR}/run_qwen2moe_process_ubatch_test.sh")
    set_tests_properties(qwen2moe_llama_process_ubatch PROPERTIES TIMEOUT 120)
  endif()
  if(VF_BUILD_LLAMA_QWEN2MOE_BUILD_MOE_FFN_TEST)
    add_test(NAME qwen2moe_llama_build_moe_ffn
      COMMAND "${CMAKE_COMMAND}" -E env
        "VF_LLAMA_SOURCE_DIR=${VF_LLAMA_SOURCE_DIR}"
        "VF_LLAMA_LIBRARY=${VF_LLAMA_LIBRARY}"
        "VF_TEST_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/qwen2moe_build_moe_ffn"
        "CXX=${CMAKE_CXX_COMPILER}"
        "${CMAKE_CURRENT_SOURCE_DIR}/run_qwen2moe_build_moe_ffn_test.sh")
    set_tests_properties(qwen2moe_llama_build_moe_ffn PROPERTIES TIMEOUT 120)
  endif()
  if(VF_BUILD_LLAMA_QWEN2MOE_GGUF_BUILD_MOE_FFN_TEST)
    add_test(NAME qwen2moe_llama_gguf_build_moe_ffn
      COMMAND "${CMAKE_COMMAND}" -E env
        "VF_LLAMA_SOURCE_DIR=${VF_LLAMA_SOURCE_DIR}"
        "VF_LLAMA_LIBRARY=${VF_LLAMA_LIBRARY}"
        "VF_TEST_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/qwen2moe_gguf_build_moe_ffn"
        "CXX=${CMAKE_CXX_COMPILER}"
        "${CMAKE_CURRENT_SOURCE_DIR}/run_qwen2moe_gguf_build_moe_ffn_test.sh")
    set_tests_properties(qwen2moe_llama_gguf_build_moe_ffn PROPERTIES TIMEOUT 120)
  endif()
  if(VF_BUILD_LLAMA_QWEN2MOE_R659_Q8_GGUF_TEST)
    add_test(NAME qwen2moe_llama_r659_q8_gguf_build_moe_ffn
      COMMAND "${CMAKE_COMMAND}" -E env
        "VF_LLAMA_SOURCE_DIR=${VF_LLAMA_SOURCE_DIR}"
        "VF_LLAMA_LIBRARY=${VF_LLAMA_LIBRARY}"
        "VF_TEST_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/qwen2moe_r659_q8_gguf"
        "CXX=${CMAKE_CXX_COMPILER}"
        "${CMAKE_CURRENT_SOURCE_DIR}/run_qwen2moe_r659_q8_gguf_build_moe_ffn_test.sh")
    set_tests_properties(qwen2moe_llama_r659_q8_gguf_build_moe_ffn PROPERTIES TIMEOUT 120)
  endif()
endif()

# VFB3 session result enters the GGML graph atomically and is weighted once.
add_executable(vf_route_graph_vfr3_test route_graph_vfr3_test.cpp)
target_link_libraries(vf_route_graph_vfr3_test PRIVATE vf_native_ggml Threads::Threads)
target_compile_options(vf_route_graph_vfr3_test PRIVATE -Wall -Wextra -Werror)
add_test(NAME route_graph_vfr3 COMMAND vf_route_graph_vfr3_test)

# Process-level CPU integration against the real coordinator and its rank worker.
# The executables and fixture paths are supplied explicitly by the isolated run.
add_executable(vf_route_graph_vfr3_bridge_process_test route_graph_vfr3_bridge_process_test.cpp)
target_link_libraries(vf_route_graph_vfr3_bridge_process_test PRIVATE vf_native_ggml)
target_compile_options(vf_route_graph_vfr3_bridge_process_test PRIVATE -Wall -Wextra -Werror)
add_test(NAME route_graph_vfr3_bridge_process COMMAND vf_route_graph_vfr3_bridge_process_test)
set_tests_properties(route_graph_vfr3_bridge_process PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
if(VF_BUILD_LLAMA_QWEN2MOE_BRIDGE)
  target_link_libraries(vf_route_graph_vfr3_bridge_process_test PRIVATE vf_qwen2moe_llama_route_bridge)
  target_compile_definitions(vf_route_graph_vfr3_bridge_process_test PRIVATE
    VF_R681_ENABLE_QWEN2MOE_BRIDGE=1)
  set_target_properties(vf_route_graph_vfr3_bridge_process_test PROPERTIES
    BUILD_RPATH "${vf_llama_library_dir}")
  add_test(NAME qwen2moe_llama_route_bridge_process
    COMMAND vf_route_graph_vfr3_bridge_process_test --qwen2moe-bridge)
  set_tests_properties(qwen2moe_llama_route_bridge_process PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 120)
endif()

# Diagnostic reference only; it is not part of the live worker/provider path.
option(VF_BUILD_SCALAR_REFERENCE "Build bounded SC117 scalar diagnostic" OFF)
if(VF_BUILD_SCALAR_REFERENCE)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "Scalar reference requires a compiler supporting strict float flags")
  endif()
  add_library(vf_native_scalar_reference STATIC scalar_reference.cpp scalar_reference_c.cpp)
  target_compile_features(vf_native_scalar_reference PRIVATE cxx_std_17)
  target_compile_options(vf_native_scalar_reference PRIVATE -Wall -Wextra -Werror -ffp-contract=off -fno-fast-math)
  target_link_libraries(vf_native_scalar_reference PUBLIC ggml-cpu ggml-base)
  add_executable(vf_native_scalar_reference_test scalar_reference_test.cpp)
  target_compile_options(vf_native_scalar_reference_test PRIVATE -Wall -Wextra -Werror -ffp-contract=off -fno-fast-math)
  target_link_libraries(vf_native_scalar_reference_test PRIVATE vf_native_scalar_reference)
  add_test(NAME native_scalar_reference COMMAND vf_native_scalar_reference_test)
endif()

# Optional reference test only: actual pinned Strata generic native CPU functions.
# x86-specific paths are aborting stubs on AArch64 and are not qualified here.
if(STRATA_GENERIC_SOURCE_DIR)
  if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    message(FATAL_ERROR "Strata generic parity stubs are only for AArch64")
  endif()
  execute_process(COMMAND git -C "${STRATA_GENERIC_SOURCE_DIR}" rev-parse HEAD OUTPUT_VARIABLE strata_revision OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE strata_status)
  if(NOT strata_status EQUAL 0 OR NOT strata_revision STREQUAL "fb58e0dbc8399662c0e47c76578c6e878b14f6cf")
    message(FATAL_ERROR "unexpected Strata source revision")
  endif()
  execute_process(COMMAND git -C "${STRATA_GENERIC_SOURCE_DIR}" status --porcelain -- src/kernels/cpu/native_expert.cpp include/strata/kernels/cpu OUTPUT_VARIABLE strata_dirty RESULT_VARIABLE strata_clean_status)
  if(NOT strata_clean_status EQUAL 0 OR NOT strata_dirty STREQUAL "")
    message(FATAL_ERROR "Strata parity source modifications are not admitted")
  endif()
  add_executable(vf_strata_generic_parity strata_generic_parity.cpp strata_generic_stubs.cpp "${STRATA_GENERIC_SOURCE_DIR}/src/kernels/cpu/native_expert.cpp")
  target_compile_features(vf_strata_generic_parity PRIVATE cxx_std_17)
  target_include_directories(vf_strata_generic_parity PRIVATE "${STRATA_GENERIC_SOURCE_DIR}/include")
  target_link_libraries(vf_strata_generic_parity PRIVATE vf_native_ggml)
  add_test(NAME strata_generic_parity COMMAND vf_strata_generic_parity)
endif()
# Explicitly invoked local CPU diagnostic; never run automatically by CTest.
add_executable(vf_real_expert_oracle real_expert_oracle.cpp)
target_link_libraries(vf_real_expert_oracle PRIVATE vf_native_ggml)
target_compile_options(vf_real_expert_oracle PRIVATE -Wall -Wextra -Werror)
# Explicit local model-input diagnostic; does not perform inference.
add_executable(vf_sc117_embedding sc117_embedding.cpp)
target_link_libraries(vf_sc117_embedding PRIVATE ggml-base)
target_compile_options(vf_sc117_embedding PRIVATE -Wall -Wextra -Werror)
# Explicit sampled CPU scalar check for a previously produced SC117 QKV vector.
add_executable(vf_check_sc117_qkv check_sc117_qkv.cpp)
target_link_libraries(vf_check_sc117_qkv PRIVATE ggml-cpu ggml-base)
target_compile_options(vf_check_sc117_qkv PRIVATE -Wall -Wextra -Werror)
# Explicit SC117 GDN matrix diagnostic, not a CTest.
add_executable(vf_check_sc117_gdn_output check_sc117_gdn_output.cpp)
target_link_libraries(vf_check_sc117_gdn_output PRIVATE ggml-cpu ggml-base)
target_compile_options(vf_check_sc117_gdn_output PRIVATE -Wall -Wextra -Werror)
# Explicit SC117 shared-expert sampled matrix diagnostic, not a CTest.
add_executable(vf_check_sc117_shared check_sc117_shared.cpp)
target_link_libraries(vf_check_sc117_shared PRIVATE ggml-cpu ggml-base)
target_compile_options(vf_check_sc117_shared PRIVATE -Wall -Wextra -Werror)
# Explicit layer-1 Q5_K QKV sampled diagnostic; not a CTest.
add_executable(vf_check_sc117_layer1_qkv check_sc117_layer1_qkv.cpp)
target_link_libraries(vf_check_sc117_layer1_qkv PRIVATE ggml-cpu ggml-base)
target_compile_options(vf_check_sc117_layer1_qkv PRIVATE -Wall -Wextra -Werror)
# Explicit layer-1 attention projections diagnostic; not a CTest.
add_executable(vf_check_sc117_layer1_route check_sc117_layer1_route.cpp)
target_link_libraries(vf_check_sc117_layer1_route PRIVATE ggml-cpu ggml-base)
target_compile_options(vf_check_sc117_layer1_route PRIVATE -Wall -Wextra -Werror)
# Explicit layer-1 shared-expert sampled diagnostic; not a CTest.
add_executable(vf_check_sc117_layer1_shared check_sc117_layer1_shared.cpp)
target_link_libraries(vf_check_sc117_layer1_shared PRIVATE ggml-cpu ggml-base)
target_compile_options(vf_check_sc117_layer1_shared PRIVATE -Wall -Wextra -Werror)
# Explicit layer-2 Q6_K QKV sampled diagnostic; not a CTest.
add_executable(vf_check_sc117_layer2_qkv check_sc117_layer2_qkv.cpp)
target_link_libraries(vf_check_sc117_layer2_qkv PRIVATE ggml-cpu ggml-base)
target_compile_options(vf_check_sc117_layer2_qkv PRIVATE -Wall -Wextra -Werror)

