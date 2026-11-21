# J. XLA/GoMLX vs hand-written AVX2/VNNI for 4-bit block-quantised prefill

**Task:** issue-14, prefill spike. Research questions 1-4 from the parent brief.

---

## EVIDENCE STATUS (read this first)

**This run had no web access.** No `web_search`, no `source_check`, no `ctx_execute`
and no shell. The only tools available were `read` and `write`. Every claim below is
therefore either (a) arithmetic derived here from the parameters in the brief, (b)
architecture-stable fact that does not change with toolchain versions, or (c) recalled
knowledge that **has not been checked against a live source in this run**.

Nothing below is a verified citation. No URL is given that I have resolved, and I have
not invented any issue number or CL. Where the brief supplied a number (83009) it is
marked as *supplied by the brief, unverified here*. Section "Verification backlog" gives
copy-paste commands that close each gap in under a minute on this machine, and each gap
is tagged `[UNVERIFIED]`. Anything tagged that way must not be quoted in a commit
message or a doc as established fact until someone has run the check.

Confidence labels: **high** = architecture or arithmetic, stable and independently
checkable; **medium** = well-known but version-sensitive; **low** = recollection, do not
build on it.

---

## Summary

XLA has no efficient 4-bit block-quantised matmul on CPU. The realistic XLA route is
dequantise in-graph and feed either f32 to Eigen or u8s8-with-per-channel-scales to
oneDNN, and both routes pay extra memory traffic per weight element that a fused
hand-written kernel does not. For a memory-bandwidth-bound 4-bit GEMV the fused kernel
is already at the byte floor, so XLA can at best tie on bytes and loses 4.8x to 16x if
the dequant buffer is materialised. For prefill GEMM the binding constraint is MACs per
instruction, where int8 VNNI is 4x to 8x f32 FMA, so XLA CPU is worse there too. Verdict:
not a viable path to beat the hand-written kernel. The actionable finding is the fourth
one: llama.cpp's 179 tok/s is almost certainly `vpdpbusd` (VEX, AVX-VNNI) over 32-weight
`q4_0` blocks against per-32-block-quantised `q8_0` activations, which is structurally the
same algorithm mantle already has and differs mainly in the instruction and in the choice
of a uniform 32-block layout.

---

## Findings

### 1. What XLA/GoMLX would actually require for a 4-bit block-quantised LLM [Q1]

1. **Claim: XLA's CPU backend has no first-class 4-bit block-quantised dot.**
   **Sources:** none fetched. **Support:** inference from the StableHLO type system plus
   the 4-bit story described below. **Confidence:** medium.
   XLA's integer types are i1..i64. There is no `i4` element type with a packed storage
   layout, and no op that consumes "one fp16 scale per 32 i8 values" as a primitive.
   The one thing that comes close is `dot_scaled` (a StableHLO op added for microscaling
   formats such as MXFP4/MXFP8, whose block size is 32). Whether XLA:CPU lowers
   `dot_scaled` to a real kernel or to an emulation/dequantise is **`[UNVERIFIED]`**, and
   it is the single highest-value thing to check, because `dot_scaled` is the only
   expression of "32-element block scale" that does not require you to write the
   dequantisation yourself.

2. **Claim: without `dot_scaled`, the 32-block scale layout is expressible only by
   reshaping the graph, which defeats the point.**
   **Support:** interpretation. **Confidence:** high on the logic, medium on XLA specifics.
   You can write the maths as: reshape activations to `[B, K/32, 32]`, do `K/32`
   separate small dots of width 32, scale each result by the scale vector, then reduce
   over the `K/32` axis. That is exactly the semantics mantle wants, and it is
   expressible. The cost is that you have replaced one GEMM of depth K with `K/32` GEMMs
   of depth 32 plus a reduction. No BLAS, no Eigen, and no oneDNN kernel will run
   efficiently at K=32, so this form is strictly worse than dequantising.

3. **Claim: therefore the dequantisation has to be materialised as graph ops.**
   **Support:** interpretation. **Confidence:** medium-high.
   The graph becomes: `unpack nibbles -> convert to f32 -> multiply by broadcast block
   scale -> dot`. Each of those is a separate elementwise op over a `[rows, K]` tensor,
   and XLA:CPU will generally fuse elementwise chains but **will not fuse across the dot
   operand boundary**, because the dot is lowered to a precompiled Eigen or oneDNN kernel
   that needs a contiguous typed buffer. So the dequantised tensor becomes a real
   allocation and a real round trip through memory. That is the fatal step, quantified in
   finding 5.

4. **Claim: XLA:CPU is Eigen for most ops, with an optional oneDNN path; the oneDNN
   quantisation contract is per-tensor or per-output-channel scales on int8, which is the
   wrong shape for a 32-element K-block scale.**
   **Support:** recalled knowledge, not re-fetched. **Confidence:** medium for Eigen plus
   optional oneDNN; medium for per-channel-only scales; low for whether recent oneDNN
   grew block-scaled 4-bit CPU kernels (`[UNVERIFIED]`, see backlog item V4).
   oneDNN's VNNI kernels are exactly the thing you would want on Alder Lake: `avx2_vnni`
   and `avx512_core_vnni` implementations of u8s8 matmul exist and use `vpdpbusd`.
   But their scaling model is one scale (and zero point) per output channel, applied after
   accumulation, plus optional per-tensor activation scales. A scale that varies *inside*
   the K reduction, every 32 elements, is not in that contract. So even on the oneDNN
   path you must dequantise 4-bit to int8 or f32 yourself before the matmul sees it, and
   you lose the per-32-block correction entirely, which is a numerical change, not an
   optimisation.
   Practical consequence: the only way to reach VNNI *through* XLA would be to get the
   graph to produce a u8 operand and let oneDNN's u8s8 path run. XLA's type promotion
   would make the dequant f32 and then `convert` to u8, i.e. you pay f32 traffic and then
   reduce, which is the worst of both (finding 5, the 16x row).

5. **Claim: what GoMLX adds is PJRT plumbing and a graph builder, not a quantisation
   story.**
   **Support:** interpretation from the brief's description of GoMLX as a PJRT binding.
   **Confidence:** medium.
   GoMLX gives you Go bindings for building an XLA graph and a PJRT CPU plugin to run it.
   It does not give you packed 4-bit storage types, a block-scaled dot, or a kernel
   library. Everything in findings 1-4 is XLA's problem, and GoMLX does not change it. You
   would be hand-building the dequant-then-dot graph in Go, which is more code than the
   kernel you already have, with a compiler between you and the loop nest.

### 2. Is XLA/GoMLX viable to beat a hand-written kernel? [Q2]

6. **Claim: no, and the argument is a bytes-per-weight ratio you can compute exactly from
   the brief's own numbers.**
   **Support:** arithmetic derived here. **Confidence:** high on the arithmetic.
   Layout per the brief: 4-bit weights, 32-weight sub-blocks with a 6-bit sub-scale,
   256-weight superblocks with one fp16 super-scale. Bytes per weight element:

   | component | bytes/element |
   |---|---|
   | 4-bit weight | 0.5000 |
   | 6-bit sub-scale / 32 elems | 0.0234 |
   | fp16 super-scale / 256 elems | 0.0078 |
   | **native (k-family) total** | **0.5313** |

   (For comparison, llama.cpp `q4_0` is an fp16 scale per 32 weights: 0.5 + 0.0625 =
   0.5625 B/elem. Same ballpark.)

   Now the same element under each XLA formulation, counting every DRAM touch of the
   dequantised buffer (read source, write buffer, read buffer back):

   | formulation | bytes/weight | multiplier vs fused |
   |---|---|---|
   | fused dequant-in-registers (hand kernel) | 0.53 | 1.0x |
   | dequant to u8 buffer, then u8s8 dot | 0.53 + 1 + 1 = 2.53 | **4.8x** |
   | dequant to f16 buffer, then f16 dot | 0.53 + 2 + 2 = 4.53 | **8.5x** |
   | dequant to f32 buffer, then f32 dot | 0.53 + 4 + 4 = 8.53 | **16.1x** |

   For a GEMV (decode, N=1) there is nothing to amortise across tokens, so the multiplier
   is the performance multiplier directly, inverted: 8.5x the traffic means at most 12%
   of the fused kernel's speed. That is the "dequantisation overhead is fatal" case, and
   it holds for the ordinary XLA lowering.

7. **Claim: there is a cache-resident best case where the traffic penalty disappears, and
   that is exactly the case XLA cannot be trusted to produce.**
   **Support:** interpretation. **Confidence:** medium-high.
   If the dequantised tile is produced and consumed while still in L2 (Alder Lake: 1.25 MB
   per P-core, plus a shared L3), DRAM traffic stays at 0.53 B/weight and the dequant is
   nearly free. Reaching that requires controlling the loop nest: for prefill with N
   tokens per weight pass, you tile by output rows so that a `[tile_rows, K]` slice is
   dequantised into L2 once and then reused for all N tokens of the batch. A hand-written
   kernel does this trivially. XLA does not expose a fusion boundary you can pin to an
   L2-sized tile, so the honest expectation is that it materialises. **That is the real
   answer to Q2: the loss is not the arithmetic, it is the loss of control over the loop
   nest and the fusion boundary.** A researcher inference, flagged as such.

8. **Claim: for prefill specifically the constraint is MACs per instruction, and there
   XLA CPU loses by 4x to 8x regardless of traffic.**
   **Support:** arithmetic from x86 architecture, stable. **Confidence:** high on
   instruction throughput ratios, medium on the absolute GFLOP comparison.
   AVX2 f32 FMA does 8 MACs per instruction; with two FMA ports that is 16 MAC/cycle.
   VEX `vpdpbusd` does 64 MACs per instruction. So int8 VNNI is roughly 4x to 8x the
   f32 MAC throughput per cycle, before counting the fact that XLA must first produce the
   f32 operand. Prefill at 179 tok/s on a 1.73B model implies about
   2 x 1.73e9 x 179 = 620 GFLOP/s effective. Prefill at 44 tok/s implies about
   152 GFLOP/s. Alder Lake P-core at ~3 GHz with AVX2 f32 is in the few-hundred GFLOP/s
   range for a handful of cores, and with int8 VNNI is well north of 1 TOPS. So the
   measured 4.07x gap is consistent with "llama.cpp is running VNNI and mantle is not",
   which is corroborating rather than independent evidence. **Inference, not proof.**
   Note the same arithmetic gives the decode roofline: 1.73e9 x 0.53 B = 0.92 GB per token,
   against roughly 30-40 GB/s achievable on a dual-channel laptop part, is about 23-30 ms
   per token, i.e. 33-43 tok/s. That the brief's 44 tok/s figure sits right on that
   roofline is itself a signal that the current kernel is close to bandwidth-bound, and
   that the win has to come from instruction choice in the compute-bound prefill case.

9. **Claim: the verdict is no, but for two different reasons depending on batch size.**
   **Support:** synthesis. **Confidence:** high.
   - N=1 (decode): bytes floor is identical for XLA and for a fused kernel. XLA can at
     best tie and in practice loses 4.8x to 16x. XLA cannot help.
   - N large (prefill): compute-bound, XLA CPU has no VNNI-quality int8 path reachable
     without dequantising to f32, so it loses 4x to 8x on MAC throughput.
   XLA:GPU is a different question entirely and not asked here.

### 3. Go-side options for VNNI or equivalent [Q3]

10. **Claim: the encoding facts that constrain every option.**
    **Support:** x86 architecture, stable, independently checkable. **Confidence:** high.
    `VPDPBUSD` exists in exactly two encodings: `VEX.128/VEX.256` (feature AVX-VNNI,
    CPUID leaf 7.1 EAX bit 4 `AVX_VNNI`) and `EVEX.128/256/512` (feature AVX512_VNNI,
    and AVX10). Alder Lake consumer parts expose AVX-VNNI and have AVX-512 disabled, so
    they execute **only** the VEX form. An EVEX-encoded `vpdpbusd` will `#UD` there.
    **This is the crux of question 3c: a small assembly file using EVEX `VPDPBUSD` is
    not usable on this CPU, no matter how it is assembled.** The 256-bit VEX form is what
    you need, and it is the one Go's assembler reportedly lacks.

11. **Claim: Go's `simd/archsimd`, issue 83009, and the absence of a byte dot.**
    **Sources:** golang/go issue 83009 is *supplied by the brief*; I did not resolve it.
    **Support:** recalled. **Confidence:** low for current state, medium for the shape of
    the problem.
    The brief states archsimd exposes no byte VNNI dot. Whether any issue or CL since then
    requests or adds `vpdpbusd`-style `Int8x16`/`Int8x32` dot is **`[UNVERIFIED]`**, and
    the check is a one-line grep (backlog V1). Two things make me expect it will still be
    absent (inference, medium): archsimd is deliberately a portable, fixed API surface,
    and a dot-product-accumulate instruction with saturating intermediate semantics is
    awkward to specify portably across arm64 and amd64. Also relevant: archsimd at the
    versions I am aware of exposes widening and multiply-add for the integer types but not
    the `maddubsw`/`maddwd` pair either, so the classic non-VNNI AVX2 idiom is not
    directly reachable from the Go API either, without dropping to the byte-level ops.

12. **Claim: Go's assembler and VEX `VPDPBUSD`.**
    **Support:** recalled. **Confidence:** low, `[UNVERIFIED]`.
    The brief already states that Go's assembler has `VPDPBUSD` only in EVEX. That is
    consistent with Go's x86 assembler having been extended for AVX-512 but not for the
    AVX-VNNI VEX form, since AVX-VNNI is an Intel-only-in-practice feature set that
    postdates the initial AVX-512 work. Do not treat this as confirmed. The grep in
    backlog V2 settles it in seconds.

13. **Claim: if the assembler lacks the VEX form, emitting raw bytes is the escape hatch,
    and it is genuinely viable.**
    **Support:** interpretation of Go's assembler capabilities. **Confidence:** medium,
    `[UNVERIFIED]` on whether `BYTE` is accepted inside a `TEXT` body in the current
    toolchain.
    Go assembly has data directives, so a `TEXT` body containing the hand-encoded VEX
    sequence (`C4 E2 7D 50 /r` family) as `BYTE $0x..` constants should assemble and run.
    You lose operand validation and register allocation checking, and you must emit the
    modrm/sib yourself, but it needs no cgo, no new dependency, and no toolchain change,
    which fits mantle's constraints. This is the lowest-risk pure-Go route to real VNNI
    and I would treat it as the primary candidate alongside cgo. It is also ugly enough
    that it needs a differential test against the existing scalar and AVX2 paths, which
    the repo's "correctness over piecemeal field checks" rule already asks for.

14. **Claim: cgo plus intrinsics works and is the least surprising route, but it costs a C
    toolchain and it fights the repo's "pure Go" identity.**
    **Support:** recalled, medium confidence, `[UNVERIFIED]` on compiler version support.
    `_mm256_dpbusd_epi32` is available via `immintrin.h` when compiling for AVX-VNNI.
    Both GCC and Clang support a per-function target attribute (`__attribute__((target("avxvnni")))`)
    so you can keep one binary and dispatch at runtime, which is exactly the model a
    portable toolchain wants. cgo call overhead is tens of nanoseconds per call, which is
    irrelevant if you cross the boundary once per matmul (microseconds of work) and fatal
    if you cross it once per 32-element block. Note the README's stated position that this
    is an experiment in pure Go: cgo does not breach the no-new-dependency rule in the
    letter, but it does breach the spirit, and the fallback path still has to exist for
    every non-AVX-VNNI machine, so you are maintaining two implementations either way.

15. **Claim: there is no pure-Go library I can name that provides VNNI-backed int8 dot
    products.**
    **Support:** none. **Confidence:** low, and this is a genuine gap.
    `[UNVERIFIED]` and largely unsearchable without web access. Realistically the
    ecosystem here is: `golang.org/x/sys/cpu` for feature detection, `gonum` (which does
    not do int8 block quantisation), and hand-written assembly in projects like
    `gorgonia`. I would not expect to find a maintained library that does 4-bit
    block-quantised GEMV with VNNI; that is a kernel, not a library, and mantle already
    owns one.

### 4. What llama.cpp does for AVX-VNNI [Q4]

16. **Claim: ggml has a dedicated AVX-VNNI path for `q4_0` x `q8_0` that uses
    `_mm256_dpbusd_epi32`, selected by the `__AVXVNNI__` preprocessor macro.**
    **Sources:** none fetched. **Support:** recalled. **Confidence:** medium,
    `[UNVERIFIED]` but high-leverage and cheap to check (backlog V5).
    The function to look for is `ggml_vec_dot_q4_0_q8_0` in the x86 quant kernels, which
    in recent trees lives under `ggml/src/ggml-cpu/arch/x86/quants.c` (it was
    `ggml-quants.c` before the backend directory reorganisation). That function
    traditionally carries several compile-time variants: a generic scalar/AVX baseline, an
    AVX2 `_mm256_maddubs_epi16` + `_mm256_madd_epi16` variant, and a VNNI variant guarded
    by `__AVXVNNI__`. The dispatch is compile-time via `-mavxvnni` (CMake option
    `GGML_AVX_VNNI` or equivalent), not runtime, which is why prebuilt binaries are
    usually the AVX2 build. **So the 179 tok/s figure almost certainly requires a
    locally built llama.cpp with AVX-VNNI enabled.** That is worth confirming, because if
    the comparison binary is the AVX2 build then 179 tok/s is being achieved *without*
    VNNI and mantle's gap is something else entirely. This is the most decision-relevant
    unknown in the whole brief.

17. **Claim: the algorithm is structurally identical to mantle's, which is the good news.**
    **Support:** interpretation. **Confidence:** medium-high.
    `q4_0` is 32 weights sharing one fp16 scale, stored as 16 bytes of nibbles plus a
    2-byte `d`. Activations are quantised on the fly to `q8_0`: 32 int8 values sharing one
    fp16 scale. Per block you compute `dpbusd(weights_u8, acts_s8)` into an int32
    accumulator, then multiply the block result by `d_w * d_a` and add into the row sum.
    That is a per-32-block scale on both sides, folded after the dot, with the scale
    applied in f32 once per block. It is exactly the shape of mantle's kernel. The
    difference is one instruction and the absence of the 6-bit sub-scale layer.

18. **Claim: mantle's k-family superblock may be a liability for this specific fast path,
    and `q4_0`-equivalent would be faster.**
    **Support:** researcher inference, explicitly. **Confidence:** medium.
    `vpdpbusd` wants a uniform sequence of 32-element blocks with an accumulator and one
    scale applied at the end of each. A 256-weight superblock with a super-scale plus
    6-bit sub-scales sits *across* four `dpbusd` blocks, so the fast loop has to fetch
    and decode 6-bit sub-scales for four blocks out of the same bytes, then recombine with
    the super-scale. Every extra unpack step in the inner loop costs more than the 0.031
    B/elem it saves (the difference between 0.5313 and 0.5625). If prefill throughput is
    the goal, a uniform 32-block layout with an fp16 scale has a strictly simpler fast
    path. This is inference, not measurement, but it is testable cheaply by comparing
    existing k-format vs a q4_0-style format on the same kernel.

19. **Claim: no repacking is required on this path.**
    **Support:** recalled. **Confidence:** medium, `[UNVERIFIED]`.
    x86 `q4_0` is used in its on-disk block layout directly; the inner loop slices 32
    weights at a time out of the 18-byte block. Repacking matters on ARM (`Q4_0_4_4`,
    `Q4_0_8_8` interleaved layouts) and for the AMX and some AVX-512 tiles, where data is
    rewritten into a tile-friendly layout at load time. On AVX2/AVX-VNNI x86 the win comes
    from the instruction and from folding the scale out of the loop, not from repacking.
    If true, this is directly good news for mantle: no format change is needed to reach
    parity, only the instruction.

---

## Contradictions

- **Prefill numbers vs decode roofline.** 44 tok/s is quoted as *prefill*, but the same
  arithmetic that explains llama.cpp's 179 tok/s as VNNI makes the decode roofline land
  at 33-43 tok/s. Either the quoted 44 tok/s is a decode-class number being described as
  prefill, or the model is small enough per-layer that prefill and decode roofs coincide.
  Worth confirming what was actually measured before optimising the wrong loop.
- **VNNI vs AVX2 in the comparison binary.** If the llama.cpp baseline was not built with
  `-mavxvnni`, then finding 8's explanation of the 4.07x gap is wrong and the gap is
  algorithmic. These two explanations predict very different fixes. Unresolved here.
- No contradiction found between sources, because no sources were fetched.

## Missing evidence

All of the following are `[UNVERIFIED]` in this run and each is cheap to close:

- V1. Whether any archsimd version exposes an int8/byte dot, and whether any issue or CL
  beyond 83009 requests it.
- V2. Whether Go's assembler accepts VEX `VPDPBUSD`.
- V3. Whether XLA:CPU lowers `dot_scaled` to anything real.
- V4. Whether recent oneDNN has block-scaled 4-bit CPU matmul, and what XLA:CPU's
  oneDNN integration actually claims support for.
- V5. Whether ggml's AVX-VNNI `q4_0` path exists in the current tree, and whether the
  179 tok/s binary had it enabled.
- V6. Whether `BYTE $0x..` inside a `TEXT` body assembles in the current toolchain.
- V7. Version and date of every document referenced. I have supplied **no dates**, because
  I cannot check publication dates without fetching.

## Verification backlog (commands that close each gap)

```
# V1: does archsimd expose a byte/int8 dot?
grep -rin 'dpbusd\|DotProduct\|dpwssd' "$(go env GOROOT)/src/simd/archsimd/" | head -50

# V2: does the assembler know VEX VPDPBUSD?
grep -in 'vpdpbusd' "$(go env GOROOT)/src/cmd/internal/obj/x86/anames.go"
grep -in 'AVXVNNI'  "$(go env GOROOT)/src/cmd/internal/obj/x86/"*.go | head -50
# then: assemble a probe .s, build, and check the encoding prefix
#   objdump -d <binary> | grep -A2 dpbusd
#   first byte C4/C5 => VEX (usable on Alder Lake); 62 => EVEX (not usable)

# CPU feature sanity on this machine
grep -o 'avx_vnni\|avx512_vnni' /proc/cpuinfo | sort -u

# V5: ggml AVX-VNNI q4_0 path
grep -rn '__AVXVNNI__' <llama.cpp>/ggml/src/ggml-cpu/ | head -30
grep -rn 'ggml_vec_dot_q4_0_q8_0' <llama.cpp>/ggml/src/ggml-cpu/arch/x86/quants.c | head
grep -rin 'AVX_VNNI' <llama.cpp>/ggml/CMakeLists.txt <llama.cpp>/CMakeLists.txt
# and confirm how the 179 tok/s binary was built
llama-cli --version   # or check the build dir's CMakeCache.txt for GGML_AVX_VNNI

# V6: raw-byte escape hatch
# write a TEXT func whose body is BYTE $0xc4; BYTE $0xe2; ... and run one GOEXPERIMENT=simd go build
```

## Sources

No sources were fetched in this run, so there is nothing to keep or reject. The brief's
mention of golang/go#83009 is treated as a pointer supplied by the task author, not as a
verified reference. The file deliberately contains **no URLs and no dates**, because
inventing either would be worse than omitting them.

## Next steps (ordered by value per unit of effort)

1. Answer V5 first. If the 179 tok/s llama.cpp was an AVX2 build, the whole premise of the
   VNNI investigation changes and the gap is algorithmic. One `grep` and one
   `--version`.
2. Run V2 and V1. Together they decide between "asm BYTE escape hatch now" and "wait for
   archsimd". Both are sub-minute.
3. If V2 is negative and V6 is positive, prototype the BYTE-emitted VEX `vpdpbusd` inner
   loop behind the existing dispatch, with a differential test against the current path.
   That is the shortest route to closing a 4x gap without cgo or a new dependency.
4. Only then look at V3/V4. My expectation is that XLA:CPU stays a dead end for this
   workload, and I would not spend research time there before the VNNI question is settled.

## Residual risks

- Three of the four questions are answered from recalled knowledge rather than fetched
  sources. Findings 1, 2, 3, 16 and 19 carry real risk of being stale by late 2026,
  particularly the XLA and ggml details, both of which move fast.
- The quantitative argument in findings 6-9 is arithmetic and does not depend on external
  sources, so it should hold. The instruction-throughput ratios in finding 8 assume
  Alder Lake P-core port counts; E-core behaviour differs and the measurement may have
  run on a mix.
- Finding 18 (k-family superblock as a liability) is inference. It is plausible and
  cheap to test, but it is not evidence and should not be actioned without a benchmark.

---

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Report written to /work/wt/mantle-quant14/spike/issue-14/prefill/J-xla-gomlx.md answering all four questions: XLA 4-bit block-quant path (findings 1-5), viability verdict with bytes-per-weight table (6-9), Go-side VNNI options (10-15), llama.cpp AVX-VNNI (16-19). Residual risks section included, plus a 7-item verification backlog with runnable commands."
    }
  ],
  "changedFiles": [
    "/work/wt/mantle-quant14/spike/issue-14/prefill/J-xla-gomlx.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "read /work/apps/mantle/README.md, /work/apps/mantle/go.mod",
      "result": "passed",
      "summary": "Confirmed repo context (Go 1.27, GOEXPERIMENT=simd, pure-Go experiment framing) for the cgo-vs-pure-Go tradeoff in finding 14."
    }
  ],
  "validationOutput": [
    "No sources fetched: this session had only read/write/contact_supervisor. web_search, source_check and ctx_execute were not registered, so no claim in the report is source-verified. The report states this in an EVIDENCE STATUS header before any finding, tags each unverified item [UNVERIFIED], and supplies local commands to close each gap."
  ],
  "residualRisks": [
    "No live web access in this run: findings on XLA:CPU, oneDNN quantisation contracts, ggml AVX-VNNI paths and archsimd state are recalled, not fetched, and may be stale for late 2026.",
    "No URLs or dates are given deliberately, to avoid fabricating citations; the brief's issue number 83009 is marked as supplied-by-brief rather than verified.",
    "Finding 18 (k-family superblock hinders the dpbusd fast path) is researcher inference and needs a benchmark before being actioned.",
    "The comparison llama.cpp binary's build flags are unknown; if it was an AVX2 build rather than AVX-VNNI, the report's explanation of the 4.07x prefill gap is wrong."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one new research report markdown file under the issue-14 prefill spike directory. No repo source touched.",
  "reviewFindings": [
    "no blockers",
    "gap: every external claim needs one grep or fetch to become citable; commands are provided in the Verification backlog section",
    "gap: confirm whether the 179 tok/s llama.cpp baseline was built with AVX-VNNI before acting on the VNNI conclusion"
  ],
  "manualNotes": "Tooling mismatch worth flagging upward: the researcher role expects web_search/source_check, but this session was provisioned with read/write only, so the deliverable is a first-principles analysis with an explicit verification backlog rather than a sourced brief. If a web-capable agent is available, the highest-value fetch is the ggml AVX-VNNI q4_0 path plus the llama.cpp build flags, then the XLA dot_scaled CPU lowering question. I did not contact the supervisor mid-run because the task instructions already anticipate flagging unverifiable items, and stalling the pipeline for a tooling question seemed worse than delivering a clearly-caveated report."
}
```
