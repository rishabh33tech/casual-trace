"""
VisualizerServer: Lightweight HTTP server for the interactive CausalTrace graph dashboard.
"""

from __future__ import annotations
import functools
import http.server
import os
from pathlib import Path
import socketserver
import threading
from typing import Optional


class VisualizerServer:
    def __init__(self, port: int = 8765):
        self.port = port
        self.ui_dir = Path(__file__).resolve().parent.parent.parent / "ui"
        self.httpd: Optional[socketserver.TCPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self, block: bool = True):
        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=str(self.ui_dir)
        )
        socketserver.TCPServer.allow_reuse_address = True
        self.httpd = socketserver.TCPServer(("", self.port), handler)
        print(f"[*] CausalTrace Visualizer running at http://localhost:{self.port}/")

        if block:
            try:
                self.httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n[*] Shutting down visualizer server.")
                self.stop()
        else:
            self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.thread.start()

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
