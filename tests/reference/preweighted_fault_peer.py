"""Protocol fault peer; numeric correctness uses the actual Rust worker."""

import json
import os
import socket
import struct
import sys

mode = sys.argv[1]
control = None
if len(sys.argv) > 2:
    control = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    address = sys.argv[2]
    control.connect("\0" + address[1:] if address.startswith("@") else address)
stream = sys.stdin.buffer
output = sys.stdout.buffer
execs = 0


def emit(value):
    output.write(json.dumps(value, separators=(",", ":")).encode() + b"\n")
    output.flush()


def exact(length):
    assert len(stream.read(length)) == length


while line := stream.readline():
    request = json.loads(line)
    op = request["op"]
    if op == "hello":
        capabilities = ["cpu_f64_dense_moe_v1"]
        if mode != "old_peer":
            capabilities.append("cpu_mxfp4_r9_preweighted_fp32_v1")
        if mode == "bf16_cut":
            capabilities.append("mxfp4_exec_bf16_exact_raw_v1")
        emit(
            dict(
                reply="hello",
                hello=dict(
                    protocol=1,
                    rank=request["rank"],
                    arch="protocol-test-peer",
                    os="linux",
                    endian="little",
                    build="fault-only",
                    capabilities=capabilities,
                    max_message_bytes=16 * 1024 * 1024,
                    pid=os.getpid(),
                ),
            )
        )
        if mode == "fork_hold_outputs":
            child = os.fork()
            if child == 0:
                control.sendall(f"{os.getpid()}\n".encode())
                assert control.recv(1) == b"R"
                control.sendall(b"D")
                os._exit(0)
            control.close()
        if mode == "control_flood":
            control.sendall(b"S")
            stamp = dict(
                run_id="flood",
                model_revision="model",
                plan_id="plan",
                layer=0,
                batch=0,
                epoch=1,
            )
            for _ in range(16384):
                emit(
                    dict(
                        reply="event",
                        event=dict(
                            stamp=stamp,
                            rank=request["rank"],
                            attempt=1,
                            event=dict(kind="started"),
                        ),
                    )
                )
            control.sendall(b"D")
            break
    elif op == "load_mxfp4_raw":
        if mode == "fork_no_load_read":
            child = os.fork()
            if child == 0:
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, 1)
                os.dup2(devnull, 2)
                os.close(devnull)
                control.sendall(f"{os.getpid()}\n".encode())
                control.recv(1)
                control.sendall(b"D")
                os._exit(0)
            control.recv(1)
            break
        if mode == "no_load_read":
            control.sendall(b"L")
            control.recv(1)
            break
        binding = request["binding"]
        exact(request["artifact_bytes"])
        emit(
            dict(
                reply="loaded",
                artifact_sha256=binding["artifact_sha256"],
                epoch=binding["epoch"],
                experts=len(binding["experts"]),
                transferred=True,
            )
        )
    elif op in ("exec_preweighted_fp32_raw", "exec_preweighted_bf16_raw"):
        if mode == "bf16_cut":
            assert op == "exec_preweighted_bf16_raw"
            exact(1)  # no terminal receipt: the coordinator cannot infer non-execution.
            break
        exact(request["activation_bytes"])
        execs += 1
        dispatch = request["dispatch"]
        stamp = dispatch["stamp"]
        event = dict(stamp=stamp, rank=dispatch["rank"], attempt=request["attempt"])
        emit(dict(reply="event", event=dict(**event, event=dict(kind="started"))))
        if mode == "late":
            control.sendall(b"S")
            assert control.recv(1) == b"R"
        header = dict(
            producer="cpu_r9_fp32_warp_reference_v1",
            artifact_sha256=request["artifact_sha256"],
            contract=request["contract"],
            stamp=stamp,
            completion_stamp=dict(stamp),
            rank=dispatch["rank"],
            attempt=request["attempt"],
            hidden=dispatch["hidden"],
            rows=[],
        )
        for routed in dispatch["slots"]:
            slot = routed["slot"]
            header["rows"].append(
                dict(
                    token=slot["token"],
                    slot=slot["slot"],
                    expert=slot["expert"],
                    local=slot["local"],
                    weight_bits=struct.unpack("<I", struct.pack("<f", slot["weight"]))[
                        0
                    ],
                )
            )
        values = len(header["rows"]) * header["hidden"]
        body = bytes(values * 4)
        if mode == "wrong_artifact":
            header["artifact_sha256"] = "f" * 64
        elif mode == "wrong_contract":
            header["contract"] = dict(header["contract"], sha256="0" * 64)
        elif mode == "old_epoch":
            header["completion_stamp"]["epoch"] -= 1
        elif mode == "old_attempt":
            header["attempt"] -= 1
        elif mode == "foreign_rank":
            header["rank"] += 1
        elif mode == "wrong_weight":
            header["rows"][0]["weight_bits"] = 0x3F800000
        elif mode == "nan":
            body = struct.pack("<I", 0x7FC00000) + body[4:]
        elif mode == "oversized_values":
            values = (1 << 64) - 1
        elif mode == "truncated_json":
            output.write(b'{"reply":')
            output.flush()
            break
        reply = dict(reply="preweighted_fp32_raw", header=header, values=values)
        if mode == "padded_header":
            encoded = json.dumps(reply, separators=(",", ":")).encode()
            output.write(encoded + b" " * (16 * 1024 * 1024 - 1 - len(encoded)) + b"\n")
            output.flush()
        else:
            emit(reply)
        if mode == "truncate_body":
            output.write(body[:4])
            output.flush()
            break
        try:
            output.write(body)
            output.flush()
        except BrokenPipeError:
            if mode == "late":
                control.sendall(b"D")
            break
        if mode == "missing_finished":
            break
        emit(
            dict(
                reply="event",
                event=dict(
                    **event, event=dict(kind="finished", slots=len(dispatch["slots"]))
                ),
            )
        )
        if mode == "late":
            control.sendall(b"D")
        elif mode == "after_finished":
            control.sendall(b"F")
            assert control.recv(1) == b"R"
            try:
                emit(reply)
                output.write(body)
                output.flush()
            except BrokenPipeError:
                pass
            control.sendall(b"D")
    elif op == "shutdown":
        emit(dict(reply="bye", execs=execs, peak_rss_kib=None))
        break
    else:
        raise AssertionError(op)
