# R696 donor audit

- `ggml-org/llama.cpp` `3cf03257f219afbe7334045ff7c6a06ac68c627d` is the pinned MIT donor used through the public model/decode APIs in R693. No donor source file is included or changed by R696.
- InplaceKVCache arXiv `2609.14507v1` is a research reference only. The paper describes KV-cache placement; this pass did not find or validate a reusable code component/license and copied none.
- R696 extends only VoltaFabric's internal CPU planner. The capacity inputs are synthetic in tests and are not measurements from V100, DGX Spark, RTX 5080, PC1 or PC6.
- Still unbuilt: live checkpoint compatibility, runtime memory telemetry, per-layer KV/GDN inventories, a safe weight replacement executor, GPU kernels/graph qualification, inter-PC worker/transport, and complete generated text with actual VoltaFabric experts.
