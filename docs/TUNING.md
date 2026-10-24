# Inference tuning notes (research, not yet measured on the console)

Source read: this repo at v0.8.3, llama.cpp `552f18f` (`third_party/llama.cpp`), and
PS5_Vulkan `252bf91` (read-only). Nothing here was run on a console.
**V** = verified in source or docs, **I** = inferred, **T** = needs a console test.

## 1. What app-args.txt does

- If `/data/PS5LM/app-args.txt` has any lines, those lines are the whole argument list.
  The planner and its fit check are skipped (`ps5/app/main.cpp:607-618`, V). The file must
  contain `-m <path>`.
- `start()` then adds `--log-file`, appends `--metrics` if absent, and appends `--tools ...`
  unless the flag is present or `settings.json` has `"tools": false` (`main.cpp:484-496`, V).
- Choosing a model in the library deletes app-args.txt (`main.cpp:870`, V). Re-create it after.
- Everything goes to llama-server's own parser, so `-ngl`, `-ot`, `--cpu-moe`, `-ncmoe`, `-lm`,
  `-fa`, `-b/-ub`, `-ctk/-ctv` are all honoured (V). Planner defaults it would otherwise write:
  `model_plan.cpp:371-377`.
- This build removed `--mmap/--no-mmap` (no such option in `common/arg.cpp`, V). Use
  `-lm mmap|none` (`arg.cpp:2691`). An unknown flag makes the server exit at startup
  (`arg.cpp:826-827,1309`, `tools/server/server.cpp:105`). The same goes for `-sysf`, and for the
  removed `--draft*` flags.

## 2. Models larger than the heap

- `-lm none` (the planner's setting) puts "CPU" tensors in `Vulkan_Host` buffers, which are
  allocated from the same 11.44 GiB heap (`ggml-vulkan-buffers.cpp:244-262`, V). `--cpu-moe` with
  `-lm none` therefore saves nothing.
- `-lm mmap` must be given explicitly. `auto` turns mmap off because the Vulkan iGPU reports
  `mmap_support=false` (`llama-model.cpp:1511-1520`, `ggml-vulkan.cpp:15290`, V). An explicit `-lm mmap`
  is not blocked by that flag (V).
- With `-lm mmap`, CPU-side tensors (experts, and the token embedding, which is always CPU,
  `llama-model.cpp:1603`) become views of the GGUF mapping: zero copy, paged from the SSD
  (`llama-model.cpp:1854-1873`, `ggml-cpu.cpp:391-402`, V). GPU tensors are copied into Vulkan buffers.
- The CPU device's 2.13 GiB comes from `sysconf(_SC_PHYS_PAGES)` (`ggml-cpu.cpp:365-381`). It is only
  logged and used for a tensor split when a GPU reports 0/0. It is not a ceiling (V).
- `posix_madvise` is a no-op in the title (`ps5/app/compat_app.c`), so there is no readahead hint (V).
- **Not known: whether file-backed mmap works inside a title.** PS5SX2 reports it failing
  (`docs/RESEARCH.md`). Anonymous mmap works (PS5_Vulkan `docs/HARDWARE_FINDINGS.md:3709`). File mmap of a
  GGUF worked in a payload (`docs/CONSOLE.md`). If it fails, the server exits with "mmap failed" (T).
- Decode with experts on the CPU runs MUL_MAT_ID on the 3 allowed CPU threads (`patches/0001`). Only the
  routed experts are read (`ggml-cpu.c:1674-1683`, V). Prompt processing at `-ub 64` re-reads nearly the
  whole expert set per ubatch (I).

### Estimates (I, SSD behaviour unverified)

Expert share of the file is about 88-91% for Gemma 4 26B-A4B and about 95% for Qwen3-30B-A3B. Active
experts per token are about 0.9 GB and about 1.1 GB at Q4_K_M.

| Setup | Decode | Prompt |
|---|---|---|
| `--cpu-moe`, all experts on SSD, cold cache | about 1-3 tok/s (use 1 as the planning figure) | about 4-20 tok/s at best |
| `-ncmoe N`, experts of N layers on SSD, the rest on the GPU | about 2-6 tok/s | better, scales with the share on GPU |
| a quant that fits the GPU (UD-IQ3_S / IQ3_XXS) | full GPU speed | full GPU speed |

Why not 5.5 GB/s: faults are synchronous, about 3 in flight from 3 threads, no readahead hint. Expect
1-3 GB/s. Page-cache hits are unknown, because the GPU heap takes 11.44 of the 12 GiB.

Sizing `-ncmoe` for Gemma 4 26B-A4B Q4_K_M (15.78 GiB, about 14 GiB experts over 30 layers): the GPU needs
file size minus the experts moved, plus KV and about 1 GiB reserve, under 11 GiB. That gives N of about 14-16
(I). Q6_K (21.58 GiB) needs N of about 27 or more, which is close to `--cpu-moe`.

## 3. Dense Qwen3.5-9B Q8_0

Decode at 37.5 tok/s on about 9.5 GB is about 350 GB/s, near the 448 GB/s peak (I). Fewer bytes per token
should help almost linearly.

| Change | Expected effect |
|---|---|
| Q6_K file (about 6.6 bpw vs 8.5) | decode about +25-30%, about 2 GiB freed (I) |
| `-lm mmap` | the token embedding (about 1 GiB at Q8_0 for a 248k vocabulary, I) leaves the GPU heap, so context headroom grows (T) |
| `-ctk f16 -ctv f16` at 32k | no dequant in the flash-attention shader; costs about 0.5 GiB more than q8_0 (I) |
| `-ub 128 -b 512` | better matmul tile use for prompts; the flash-attention scratch below does not grow with `-ub` (I) |
| n-gram speculation `--spec-type ngram-simple --spec-ngram-simple-size-m 7` | only helps repetitive output; drafts of 8 tokens or fewer stay on the mat-vec path (`mul_mat_vec_max_cols = 8`) (I) |
| `--spec-type draft-mtp` | only if the GGUF has `nextn` tensors (`general` metadata `nextn_predict_layers` above 0) |

`-ctxcp 1` and `-cram 0` do not slow decode (V). They matter for prefix reuse (section 5).

Flash attention is needed for a quantized V cache (`llama-context.cpp:3840-3848`, V) and works on the
scalar path with q8_0 (`ggml-vulkan.cpp:15451-15498`, V).

## 4. The "other" memory

- The dashboard's Other is `pool - free - model file - planner's KV estimate`
  (`dashboard.cpp:984-987`, `stats.cpp:296-304`, V). It is a remainder, not a measurement.
- Already used before any model: 12.00 - 11.03 = about 0.97 GiB. CONSOLE.md records 11.38 GiB free before the
  screen opens and 11.03 after (about 0.35 GiB screen, about 0.62 GiB driver and system, V/I). That alone
  accounts for most of 1.1 GB.
- `scratch_limit_gib` is a disk warning for the scratch folder, not memory (`main.cpp:752-763`, V).
- Hidden scratch that the logged compute-buffer size does not include: with a quantized KV cache and
  `-ub 64` or more query tokens, flash attention dequantizes a layer's whole K and V to f16, `n_kv x
  (k_gqa + v_gqa) x 2` bytes, about 128-256 MiB at 32k, so 256-512 MiB at 64k
  (`ggml-vulkan.cpp:8089-8100,8366-8372`, `12124-12158`, V for the mechanism, I for the sizes). It grows with
  context, so it is a plausible part of why 64k (10.8 of 11.0 GiB planned) hung with a blank screen while
  32k (10.3 of 11.0) did not.
- Prompt checkpoints live in the CPU heap, which is in the same pool (`common.h:1241`, V).
- The 0.35 GiB screen cannot be reduced without code changes.

## 5. System prompt, tools, prefix cache

- Stock llama.cpp has no server-side system prompt flag. `-sys/-sysf` are not registered for the server and nothing in
  `tools/server` reads `params.system_prompt` (V); patch 0006 below adds it. No `settings.json` key or app field exists either
  (`settings.cpp:20-27`, V).
- **Patch 0006 (this branch) adds the flag.** `-sys "text"` or `-sysf /data/PS5LM/system.txt` now work in
  llama-server: the text is added as the first message of a Chat Completions, Responses, Anthropic or
  `/apply-template` request that has no system or developer message. A client that sends its own system message
  keeps it untouched, so agents that bring their own prompt are not affected and the default only reaches plain
  chat clients and the web UI. Tested on a host build (V): patches 0001-0006 apply and `llama-server` compiles
  on both pins (552f18f and v0.7.0, whose JSON type differs, which is why the patch builds a new array instead of
  inserting); with a 115 KB random-weights GGUF and a chat template, `/apply-template` showed the default added
  when absent and left out when the client sent one, and `/v1/messages` and `/v1/responses` input tokens grew by
  the same amount; a float `temperature` body was accepted on the host. Not tested on the console. A ready chat
  persona is `docs/system-prompt-chatbot.txt` (about 150 tokens, 1.5 s once at 102 tok/s; the prefix cache keeps it).
  Use: copy that file to `/data/PS5LM/system.txt` and add the line `-sysf` and the line `/data/PS5LM/system.txt` to
  `app-args.txt` (one argument per line). The file is read once at startup, so restart the app after editing it.
  Without `app-args.txt` the planner does not pass it; an app hook that appends it when the file exists is the next step.
- Options that work without the patch:
  1. `--chat-template-file /data/PS5LM/<name>.jinja`: a copy of the model's template with a fixed system text
     baked in. It applies to every endpoint and to the web UI (I for the recipe, not run). Mode "only when the
     client sent none": a client system message replaces the default. Mode "always": merged into one system
     block. No duplication either way. Keep the `<tool_call>`, `<function=` and `<parameter=` strings in the
     template so the Qwen3-Coder handler is still detected (`chat.cpp:1218-1226`). Put the text before the
     `# Tools` block to keep it in a stable prefix.
  2. `--ui-config '{"systemMessage":"..."}'`: the web UI only, and only on a browser's first visit
     (`tools/ui/.../settings-keys.constants.ts:65`, V). API clients are not affected.
- Client system messages: Chat Completions uses messages as sent and maps `developer` to `system`. Anthropic
  `system` becomes one leading system message, and patch 0004 folds later ones into it. Responses `instructions`
  becomes a leading system message, and a later system item makes the Qwen template throw ("System message must
  be at the beginning", `Qwen3.5-4B.jinja:83-86`). Patch 0004 does not cover Responses (V).
- `--tools` adds nothing to API requests (V). The browser UI fetches `/tools` and sends the six definitions in
  its own requests: about 5.1k characters, about 1.7k tokens, about 17 s cold at 102 tok/s (I). To stop the
  app appending it, set `"tools": false` in `settings.json` (`main.cpp:490`). `/tools` is callable by anyone on
  the LAN while `--host 0.0.0.0` and no `--api-key` are in use; writes are confined to the scratch folder by
  patch 0003.
- Prefix cache: with `-np 1` the slot reuses the common prefix. For this hybrid model that needs a recurrent-state
  checkpoint. With `-ctxcp 1` each request replaces the one checkpoint, so the same agent appending turns reuses
  its prefix, but a different agent that shares only the system prompt reprocesses all of it (V, traced through
  `create_checkpoint`, `server-context.cpp:2309-2373`). With `-ctxcp 4`, a checkpoint at the end of the system
  prompt survives. Each checkpoint is about 50 MiB (I, from Qwen3.5-9B dimensions; the log prints the real size).
  Saved per request at 102 tok/s: 500 tokens 4.9 s, 2,000 tokens 19.6 s, 5,000 tokens 49 s.
- `-cram 0` is fine. A cache entry costs the full KV plus about 50 MiB of state.
- Per-request `chat_template_kwargs` works and wins over the CLI default per key
  (`server-common.cpp:1356-1360`). `{"chat_template_kwargs":{"enable_thinking":false}}` disables thinking on
  Qwen3.5 (V). Server-wide: `--reasoning off`. Gemma 4: `--reasoning-budget 0`.

## 6. The float crash

`json.hpp:8314` is the final `strtod` check in nlohmann's float path, aborted by `GGML_ASSERT`
(`common/json.cpp:5`). The lexer inserts the locale's decimal-point character in place of each `.`
(`json.hpp:8073,8110,7166`). PS5_Vulkan documents that the console's `localeconv()` returns an empty decimal
point while `strtod` reads `.` (PS5_Vulkan `README.md:127`, `docs/RADV_PHASE.md:135`, `tools/radv-link.sh:68`, V).
So `0.6` becomes `0\0` `6`, `strtod` stops after `0`, and the assert fires. Integers use `strtoull/strtoll`, a
different path (`json.hpp:8281,8297`), so a request with only integers works. `setlocale(LC_NUMERIC,"C")` at
`server.cpp:89` does not help (V).

Fix on this branch: `ps5/app/compat_app.c` defines a C-locale `localeconv`, and `scripts/build-app.sh` drops
PS5_Vulkan's own `--defsym=localeconv=...` so there is one definition. Built on the host for syntax only.

## 7. Console tests (all read-only except where noted)

`H=http://<console-ip>:8081`

1. Does strtof/strtod work at all: `curl -s $H/props | jq .default_generation_settings.params.temperature`.
   About 0.6 means `strtof` reads `.` (the `--temp` flag works), so only `localeconv` is wrong. 0 means `strtof`
   is broken too and the fix above is not enough.
2. Safe probes, in order, using `/tokenize` (parses the body, runs no inference):
   `{"content":"hi"}`, then `"p":16`, `"p":-16`, `"p":5e-1`, `"p":1E3`. Last: `"p":0.5`. Expect the last to
   crash the unfixed build (relaunch the app after). If `5e-1` crashes, `strtod` itself is broken.
3. Until a fixed build is on the console, send whole numbers or exponent forms (`"temperature":6e-1`) or omit
   the fields and rely on the `--temp` flags.

## 8. MoE on the SSD: the order to find out (needs `origin/main`'s probes and the console)

Nothing here is measured yet; each step decides whether the next is worth doing.

1. Read speed (`probes/ioprobe` on `main`, a payload through elfldr, read-only):
   `ioprobe.elf?args=/data/PS5LM/models/<a GGUF over 1 GiB>`. Look at the "buffered random 1 MiB, 4 threads" and
   "direct random 1 MiB, 4 threads" lines. Expert slabs are about 0.5-2 MiB at random places, from 3 threads. Rough
   gate (I): below 1 GB/s a 26B-A4B MoE is under 1 tok/s and not worth running; 2 GB/s or more makes `-ncmoe`
   worth trying.
2. Bytes per token: `python3 scripts/gguf-experts.py <gguf or its first 24 MiB>` prints the resident and routed-expert sizes
   and the bytes read per token. Divide the measured MB/s by that for the ceiling in tok/s.
3. Does a file mapping work in a title (Test A in section 7, with `-lm mmap` and a small MoE that fits the heap):
   if the model loads and generates, mapping works; if the load aborts or the log shows an mmap error, the SSD route
   is closed and only the heap-sized models remain.
4. Then Gemma 4 26B-A4B Q4_K_M (15.78 GiB) with `-lm mmap` and `-ncmoe 14`, raising N until it loads. Everything
   else on the GPU (`-ngl 999`), `--reasoning-budget 0`, q8_0 KV, `-ub 64`.
5. If steps 1-3 pass but step 4 is slow, the lever is the read pattern (section 2: a real `posix_madvise`, or a
   dedicated reader thread issuing large reads), not the flags.
