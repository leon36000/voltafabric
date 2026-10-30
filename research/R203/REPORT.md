# R203: checkpoint-to-session CPU pipeline

The pipeline packs a bounded checkpoint selection, starts two Rust-session owners and checks two batches against the independent R201 oracle. Test with eight clones of the same expert passed; maximum absolute error 1.239776611328125e-05. This is a CPU fixture, not eight distinct checkpoint experts, GPU execution, generation or production network.

Run: `r202-7ea4b6859c974f3183aa4303b60621a7`. Manifest SHA256: `7efcb3b50346c4f0308bc3186f9df5caedb32c95fc53963f3c721e1dcc007228`.
