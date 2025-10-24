import time
from collections import deque
from threading import Lock

class RateLimiter:
    """
    Simple per-host, per-minute limiter.
    Also exposes current_window_count() so we can report usage in GUI.
    """
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.calls = deque()  # timestamps (epoch seconds)
        self.lock = Lock()

    def _purge_old(self):
        now = time.time()
        window_start = now - 60
        while self.calls and self.calls[0] < window_start:
            self.calls.popleft()

    def acquire(self):
        with self.lock:
            # purge old calls outside 60s window
            self._purge_old()

            if len(self.calls) >= self.per_minute:
                # sleep until oldest falls out of the 60s window
                sleep_for = 60 - (time.time() - self.calls[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)

            # record this call timestamp
            self.calls.append(time.time())

    def current_window_count(self) -> int:
        """
        How many calls in the current rolling 60s window.
        Used for GUI status like "45/240 this_min".
        """
        with self.lock:
            self._purge_old()
            return len(self.calls)
