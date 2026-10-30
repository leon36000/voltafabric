"""Sequential CPU session admission with conservative, observable cleanup.

Resource admission is a snapshot, not a reservation. An uncertain session is
never closed, killed, reopened or replayed by this helper.
"""
from adapters.resource_admission import AdmissionError
from adapters.session_bridge import DurableSession

class GuardedFleet:
    def __init__(self, admission, *, correlated=False):
        if not callable(admission):
            raise ValueError('admission must be callable')
        if type(correlated) is not bool:
            raise ValueError('correlated must be boolean')
        self.admission = admission
        self.correlated = correlated
        self.sessions = {}
        self.cleanup_errors = []
        self._closed_layers = set()
        self.closed = False

    def open(self, binary, init, ledger, deadline_seconds=10):
        if self.closed:
            raise RuntimeError('fleet is terminal')
        layer = init['routing']['stamp']['layer']
        if layer in self.sessions:
            raise ValueError('layer already admitted')
        try:
            session = DurableSession(binary, init, ledger, deadline_seconds,
                                     admission=self.admission, correlated=self.correlated)
        except AdmissionError:
            # No process or identity was created for the refused session.
            self.close_known()
            raise
        except BaseException:
            # The attempted session may have been sent work; do not infer its
            # state or kill it. The caller must reconcile its durable ledger.
            self.closed = True
            raise
        self.sessions[layer] = session
        return session

    def close_known(self):
        self.closed = True
        for layer, session in self.sessions.items():
            if layer in self._closed_layers:
                continue
            if session.terminal:
                self.cleanup_errors.append((layer, 'UNKNOWN or terminal; reconciliation required'))
                continue
            try:
                session.close()
                self._closed_layers.add(layer)
            except Exception as exc:
                self.cleanup_errors.append((layer, f'{type(exc).__name__}: {exc}'))
        return not self.cleanup_errors
