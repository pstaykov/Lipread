"""Local demo server with caching disabled.

A plain `python -m http.server` lets browsers cache index.html/app.js/style.css/
predictions.json indefinitely (no Cache-Control header is sent), so after editing
any of those files a normal reload can keep showing an old cached copy -- looking
like the page is broken instead of just stale. This server sends
Cache-Control: no-store on every response so every reload is guaranteed fresh.

Usage: python serve.py [port]   (default port 8000)
"""
import http.server
import sys


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        super().end_headers()


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    # bind explicitly to IPv4 loopback -- the default bind=None picks a dual-stack
    # "::" socket that can end up unreachable at 127.0.0.1 on some Windows setups.
    http.server.test(HandlerClass=NoCacheHandler, port=port, bind='127.0.0.1')
