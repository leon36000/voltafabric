"""Private CPU owner multiplexer with explicit local control, not a network lease.

The control FD is an inherited trusted pipe. `S` stops admission; existing
connections finish independently. A timeout reports uncertainty and never
closes or kills an active owner connection.
"""
import os
import selectors
import threading
import time

class MultiplexError(RuntimeError):
    pass

def serve_fleet(listener, control_fd, serve, max_sessions, join_seconds):
    if type(max_sessions) is not int or not 1 <= max_sessions <= 40 or not 0 < join_seconds <= 180:
        raise ValueError('invalid fleet bounds')
    threads, errors = [], []
    def worker(conn):
        try:
            with conn:
                serve(conn)
        except BaseException as exc:
            errors.append((type(exc).__name__, str(exc)))
    control_error = None
    with selectors.DefaultSelector() as sel:
        sel.register(listener, selectors.EVENT_READ, 'listener')
        sel.register(control_fd, selectors.EVENT_READ, 'control')
        while len(threads) < max_sessions:
            events = sel.select()
            # A pending stop wins over accepting another new connection.
            if any(k.data == 'control' for k, _ in events):
                if os.read(control_fd, 1) != b'S':
                    control_error = 'invalid or lost control pipe'
                break
            conn, _ = listener.accept()
            thread = threading.Thread(target=worker, args=(conn,))
            thread.start()
            threads.append(thread)
    deadline = time.monotonic() + join_seconds
    for thread in threads:
        thread.join(max(0, deadline-time.monotonic()))
    if any(thread.is_alive() for thread in threads):
        raise MultiplexError('active connection outcome unknown; reconciliation required')
    if control_error or errors:
        raise MultiplexError(control_error or f'owner connection failed: {errors!r}')
    return len(threads)
