# R671 — VFR3 collector at the pinned GGML graph boundary

## Result

R671 connects the reusable `RouteGraphTransaction` CPU graph node to VoltaFabric's existing VFB3/VFR3 AF_UNIX client. Remote-owner rows now enter the GGML graph through the same all-or-nothing collector boundary; the test proves that a missing response row aborts before a downstream graph consumer runs.

## Reuse and adaptation

- Donor: `ggml-org/llama.cpp`, pinned commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`, MIT. The build verifies this exact revision and an unmodified `ggml` subtree.
- Reuses the donor's `ggml_custom_4d` graph operation and CPU graph-plan abort callback from R670. No llama.cpp runtime or model loader is copied or installed.
- Reuses VoltaFabric's existing `vf::strata_bridge::Client` (`adapters/strata_native_bridge/pipe_client.cpp`) for the VFB3 request and VFR3 response over a local AF_UNIX stream. The transport implementation itself is not forked.
- Adds the original activation tensor and per-route owner kind to the graph transaction. The collector sends the route batch only when a route is remote (`owner_kind == -1`), merges returned raw FP32 rows into those route slots, and keeps local-owner slot outputs unchanged.
- It validates run/model/plan/layer/epoch against the connected session and delegates batch/attempt and VFR3 framing checks to the existing client. It validates route extents, expert IDs, owner kinds, coefficients, and finite intermediates before committing output.
- It applies the original route coefficient once, after FP32-to-FP16-to-FP32 decoding, and accumulates in f64 in token/slot order. The result is copied to the graph only after the entire batch succeeds.

## Verification

Built locally on the AArch64 CPU against the pinned GGML checkout, using the repository CMake target's `-Wall -Wextra -Werror` settings:

```text
cmake -S adapters/native_ggml -B build-full \
  -DGGML_SOURCE_DIR=/home/pc6/NAOS/60_travaux/voltafabric-gitlab-r365/.runs/llama-ggml-3cf03257
cmake --build build-full --target vf_native_ggml_test vf_route_graph_transaction_test vf_route_graph_vfr3_test -j2
ctest --test-dir build-full --output-on-failure
```

Result: **3/3 CTest tests passed** (`native_provider`, `route_graph_transaction`, `route_graph_vfr3`). The VFR3 graph fixture checks the request identity and route body, exact FP16-decoded/f64 scalar result, missing-row rejection, no downstream call after rejection, graceful client shutdown, and single-use replay refusal.

## Limits

The VFR3 peer in this new graph test is a small fake AF_UNIX peer implementing the wire protocol. This test does **not** launch `vf-strata-native-bridge` or `vf-rank-session`; R648/R649's separate local worker evidence does not turn this graph fixture into an actual graph-to-worker integration. No checkpoint, model graph, complete token generation, GPU, inter-PC/RDMA, production R53 reservation, non-Q8_0 quantization path, or performance comparison is qualified. This code is a tested graph/collector boundary, not a complete llama.cpp VoltaFabric runtime.

## Boundary clarification

The C++ `vf::strata_bridge::Client` speaks VFB3/VFR3 to a VoltaFabric native-bridge coordinator over an AF_UNIX stream. `vf-rank-session` is a separate child-side worker protocol. The graph fixture currently replaces the coordinator endpoint with its fake peer; attaching the raw session binary directly to this C++ client would test the wrong protocol boundary.

## Next step

Replace the fake peer with the actual local `vf-strata-native-bridge` coordinator and its `vf-rank-session` child, using a fresh isolated test identity and ledger; do not reuse R648/R649 identities or replay `UNKNOWN` receipts. Then identify a pinned donor graph capture point that feeds real selected IDs and post-scale weights into this transaction before attempting a checkpoint-backed generation path.
