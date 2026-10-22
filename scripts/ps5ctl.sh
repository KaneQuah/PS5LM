#!/usr/bin/env bash
# Controls the PS5LM Plus app on the console from the PC, over FTP and its HTTP API. Everything but
# `launch` needs only curl; the app takes requests by polling files in /data/PS5LM (see ps5lm-app.sh).
#
#   PS5_HOST=10.168.1.160 scripts/ps5ctl.sh status           app, API, loaded model, last log lines
#   scripts/ps5ctl.sh models                                  .gguf files on the console
#   scripts/ps5ctl.sh load <file in models/ | absolute path>  switch model, wait until the API answers
#   scripts/ps5ctl.sh unload
#   scripts/ps5ctl.sh args <file|->                           write app-args.txt (whole llama-server arg list)
#   scripts/ps5ctl.sh args-clear                              delete app-args.txt (back to the planner)
#   scripts/ps5ctl.sh logs [app|llama] [lines]
#   scripts/ps5ctl.sh wait [seconds]                          wait for /health
#   scripts/ps5ctl.sh launch                                  start the app (needs the SDK, ps5lm-app.sh)
#   scripts/ps5ctl.sh proxy start|stop|status                 float-safe proxy on 127.0.0.1:${PROXY_PORT:-6781}; adds
#                                                             the assistant system prompt (PS5_SYSTEM_FILE) when a request has none
#
# Builds before the localeconv fix (5823ae3) abort on any JSON number with a '.', and the whole app
# goes down. Point clients at the proxy until the console runs a fixed build (docs/TUNING.md section 6).
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
host=${PS5_HOST:?set PS5_HOST to the console IP}
ftp="ftp://$host:${PS5_FTP_PORT:-2121}/data/PS5LM"
api="http://$host:${PS5_API_PORT:-8081}"
pport=${PROXY_PORT:-6781}
pidf=${TMPDIR:-/tmp}/ps5lm-proxy-$pport.pid

f() { curl -sS -m "${T:-15}" --disable-epsv "$@"; }
alive() { curl -sf -m 4 "$api/health" >/dev/null 2>&1; }
put() { printf '%s\n' "${2:--}" | f -T - "$ftp/$1"; }

wait_up() {
    for _ in $(seq 1 "${1:-180}"); do alive && { echo "ps5ctl: API is up"; return 0; }; sleep 1; done
    echo "ps5ctl: API did not answer in ${1:-180}s (see: ps5ctl.sh logs)" >&2; return 1
}

case ${1:-} in
    status)
        f -o /dev/null "$ftp/settings.json" && echo "ftp:   up" || echo "ftp:   down"
        if alive; then
            echo "api:   up"
            curl -s -m 5 "$api/v1/models" | python3 -c 'import json,sys; [print("model:", m["id"], "ctx", m.get("meta",{}).get("n_ctx")) for m in json.load(sys.stdin)["data"]]'
        else
            echo "api:   down"
        fi
        echo "--- app.log (last 4)"; f "$ftp/app.log" | tail -4
        echo "--- llama.log (last 4)"; f "$ftp/llama.log" | tail -4 ;;
    models) f "$ftp/models/" | awk '{printf "%7.2f GiB  %s\n", $5/2^30, $9}' ;;
    load)
        m=${2:?file name in /data/PS5LM/models, or an absolute path on the console}
        case $m in /*) ;; *) m=/data/PS5LM/models/$m ;; esac
        put load "$m"; echo "ps5ctl: load requested: $m"; sleep 5; wait_up "${3:-240}" ;;
    unload) put unload ;;
    args) if [ "${2:?file or -}" = - ]; then f -T - "$ftp/app-args.txt"; else f -T "$2" "$ftp/app-args.txt"; fi ;;
    args-clear) f -Q "DELE /data/PS5LM/app-args.txt" "ftp://$host:${PS5_FTP_PORT:-2121}/" -o /dev/null ;;
    logs) f "$ftp/${2:-app}.log" | tail -n "${3:-40}" ;;
    wait) wait_up "${2:-180}" ;;
    launch) exec "$here/ps5lm-app.sh" launch ;;
    proxy)
        case ${2:-status} in
            start)
                [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null && { echo "ps5ctl: proxy already running"; exit 0; }
                nohup python3 "$here/ps5lm-floatsafe-proxy.py" --upstream "$api" --port "$pport" --system-file "${PS5_SYSTEM_FILE:-$here/ps5lm-assistant-prompt.md}" >/dev/null 2>&1 &
                echo $! >"$pidf"; echo "ps5ctl: proxy on http://127.0.0.1:$pport/v1 -> $api (pid $!)" ;;
            stop) [ -f "$pidf" ] && kill "$(cat "$pidf")" && rm -f "$pidf" && echo "ps5ctl: proxy stopped" ;;
            status) [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null && echo "proxy: running pid $(cat "$pidf")" || echo "proxy: not running" ;;
        esac ;;
    *) sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
