import os, errno, json, time
from typing import Optional, Deque
from collections import deque

class PipeWriter:
    """
    Non-blocking JSON-lines writer to a Unix FIFO.
    - Retries open() until a reader attaches, but only up to max_wait.
    - If no reader, either buffer (in-memory queue) or drop.
    - Handles BrokenPipe (reader disappeared) gracefully.
    """
    def __init__(self,
                 pipe_path: str = "/tmp/orderbook_pipe",
                 max_wait: float = 3.0,           # seconds to wait for a reader
                 buffer_on_no_reader: bool = True,
                 buffer_max_lines: int = 5000):
        self.pipe_path = pipe_path
        self.max_wait = max_wait
        self.buffer_on_no_reader = buffer_on_no_reader
        self.fd: Optional[int] = None
        self._buf: Deque[bytes] = deque(maxlen=buffer_max_lines)
        self._ensure_fifo()

    def _ensure_fifo(self):
        try:
            if not os.path.exists(self.pipe_path):
                os.mkfifo(self.pipe_path, 0o666)
        except FileExistsError:
            pass

    def _try_open(self) -> bool:
        """Attempt a non-blocking open; return True if opened, else False."""
        try:
            self.fd = os.open(self.pipe_path, os.O_WRONLY | os.O_NONBLOCK)
            return True
        except OSError as e:
            if e.errno == errno.ENXIO:
                # No reader attached yet
                return False
            raise

    def _ensure_open(self) -> bool:
        """Ensure pipe is open, respecting max_wait. Returns True if open, False otherwise."""
        if self.fd is not None:
            return True
        deadline = time.time() + self.max_wait
        while time.time() < deadline:
            if self._try_open():
                return True
            time.sleep(0.1)
        return False  # no reader within timeout

    def _encode(self, obj) -> bytes:
        return (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")

    def write_json(self, obj):
        payload = self._encode(obj)

        # If not open, try to open within max_wait
        if not self._ensure_open():
            # No reader yet — buffer or drop
            if self.buffer_on_no_reader:
                self._buf.append(payload)
            return  # don't block

        # If we got here, fd is open; try to flush any buffered lines first
        while self._buf:
            try:
                os.write(self.fd, self._buf[0])
                self._buf.popleft()
            except BrokenPipeError:
                os.close(self.fd)
                self.fd = None
                # reader disappeared; stop trying this write right now
                return
            except BlockingIOError:
                # pipe is full — give up this round
                return

        # Now write the current payload
        try:
            os.write(self.fd, payload)
        except BrokenPipeError:
            # Reader disappeared; drop this payload and reset fd
            try:
                os.close(self.fd)
            finally:
                self.fd = None
        except BlockingIOError:
            # Pipe buffer full; optionally buffer
            if self.buffer_on_no_reader:
                self._buf.append(payload)

    def close(self):
        if self.fd is not None:
            try:
                os.close(self.fd)
            finally:
                self.fd = None
