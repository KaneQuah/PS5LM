#!/usr/bin/env python3
"""Forward to a PS5LM llama-server, rewriting JSON floats as exponent integers (0.8 -> 8e-1).

With --system-file, a chat request that has no system or developer message gets that file as its leading
system message ({date} becomes today's date, kept last so the prefix cache stays valid all day).

Works around the console's localeconv() bug (docs/TUNING.md section 6): nlohmann's lexer aborts on any
number with a '.', but an exponent form without a '.' goes through strtod untouched.
"""
import argparse, datetime, http.server, json, re, socketserver, sys, urllib.request, urllib.error

ap = argparse.ArgumentParser()
ap.add_argument("--upstream", required=True)            # http://10.168.1.160:8081
ap.add_argument("--port", type=int, default=6781)
ap.add_argument("--system-file")                        # default system prompt for requests that carry none
a = ap.parse_args()
UP = a.upstream.rstrip("/")
MARK = re.compile(r'"@@F(-?\d+)e(-?\d+)@@"')


def walk(o):
    if isinstance(o, float):
        m = re.fullmatch(r"(-?)(\d+)\.(\d+)", repr(o))
        if m:
            sign, whole, frac = m.groups()
            mant = int(whole + frac)
            return f"@@F{'-' if sign else ''}{mant}e{-len(frac)}@@"
        return o
    if isinstance(o, dict):
        return {k: walk(v) for k, v in o.items()}
    if isinstance(o, list):
        return [walk(v) for v in o]
    return o


def with_system(o):
    msgs = o.get("messages") if isinstance(o, dict) else None
    if a.system_file and isinstance(msgs, list) and not any(m.get("role") in ("system", "developer") for m in msgs):
        text = open(a.system_file).read().strip().replace("{date}", datetime.date.today().isoformat())
        o = {**o, "messages": [{"role": "system", "content": text}] + msgs}
    return o


def safe_dumps(o):
    return MARK.sub(lambda m: f"{m.group(1)}e{m.group(2)}", json.dumps(walk(o), ensure_ascii=False))


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"                       # close-delimited, so streaming needs no chunking

    def log_message(self, *x):
        pass

    def _go(self, method):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        if body and "json" in (self.headers.get("Content-Type") or ""):
            o = json.loads(body)
            if self.path.rstrip("/").endswith("/chat/completions"):
                o = with_system(o)
            body = safe_dumps(o).encode()
        hdr = {k: v for k, v in self.headers.items() if k.lower() in ("content-type", "authorization", "accept")}
        try:
            r = urllib.request.urlopen(urllib.request.Request(UP + self.path, body, hdr, method=method), timeout=600)
        except urllib.error.HTTPError as e:
            r = e
        except Exception as e:
            self.send_response(502); self.end_headers(); self.wfile.write(str(e).encode()); return
        self.send_response(r.status if hasattr(r, "status") else r.code)
        for k, v in r.headers.items():
            if k.lower() in ("content-type", "cache-control"):
                self.send_header(k, v)
        self.end_headers()
        try:
            while True:
                c = r.read1(4096) if hasattr(r, "read1") else r.read(4096)
                if not c:
                    break
                self.wfile.write(c); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self): self._go("GET")
    def do_POST(self): self._go("POST")


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    print(f"floatsafe proxy 127.0.0.1:{a.port} -> {UP}", flush=True)
    S(("127.0.0.1", a.port), H).serve_forever()
