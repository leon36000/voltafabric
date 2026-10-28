# R116 workload output bound

This private adapter carries `workload.max_new_tokens` into the byte-admission
step and checks every integer in the already-hashed output-count artifact against
that ceiling. The comparison is per request; the sum across requests may exceed
the ceiling. The exact output artifact bytes are opened and hashed once by the
admission module, then parsed and checked. The R97 comparator remains pinned and
continues to require both run records to declare the same workload.

This contract assumes the producer defines `max_new_tokens` as the maximum
number of generated output tokens for each request, in the same unit as each
entry in `r100-token-counts-v1`. The check validates consistency of declarations
and evidence bytes. It does not authenticate the producer, establish tokenizer
correctness, prove model execution, or measure performance. The module and its
synthetic fixtures remain private research material; no service or canonical
engine behavior is changed.
