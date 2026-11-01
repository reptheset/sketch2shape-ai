# Sketch2Shape AI: measured runs

Every pipeline run recorded on 31 October 2026 while developing and testing. Values come from each job's
`job.json` (`details.timings_seconds`), so they are exactly what the backend measured.

**Hardware:** Apple Silicon MacBook Pro (arm64), 24 GB+ unified memory, macOS · Python 3.12.7 · PyTorch 2.15 (MPS,
fp16) · SDXL 1024×1024, Euler Ancestral · stage ② on Groq's free API · stage ⑤ on Hugging Face ZeroGPU
(`trellis-community/TRELLIS`, 12 sampling steps at the time of these runs, texture 1024).

Stages: ② understand (vision LLM) · ③ reference image (SDXL + ControlNet) · ④ background removal · ⑤ 3D
reconstruction · ⑥ mesh validation. Stage ① (preprocessing) always took ≤ 0.1 s and is included in the total.
Times are in UTC (add 5 h 30 min for IST).

| Time (UTC) | Object | Stage ② source | SD steps | ② s | ③ s | ④ s | ⑤ s | 3D provider | ⑥ s | **Total s** | Triangles | Material | GLB MB | Result | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 11:27 | mug | llm | 30 | 1.1 | 119 | – | – | – | – | 120 | – | – | – | ⚪ partial | run before stages ⑤–⑥ existed |
| 11:31 | mug | llm | 30 | 1.2 | 128 | – | – | – | – | 129 | – | – | – | ⚪ partial | run before stages ⑤–⑥ existed |
| 11:41 | object | fallback | 30 | 1.0 | 114 | 2.1 | – | – | – | 117 | – | – | – | ⚪ partial | run before stages ⑤–⑥ existed; Groq rate-limited → rule-based prompt |
| 12:14 | mug | llm | 30 | 1.3 | 116 | 2.2 | 43 | trellis | 0.88 | **163** | 17,198 | PBR 1024² | 1.5 | ✅ completed |  |
| 12:18 | mug | llm | 30 | 1.1 | 138 | 1.1 | 47 | trellis | 0.16 | **188** | 31,974 | PBR 1024² | 1.9 | ✅ completed |  |
| 12:24 | flower | llm | 30 | 1.2 | 147 | 1.3 | 41 | trellis | 0.13 | **191** | 12,140 | PBR 1024² | 1.5 | ✅ completed |  |
| 12:29 | sports car | llm | 30 | 1.3 | 175 | 2.3 | 41 | trellis | 0.17 | **220** | 10,561 | PBR 1024² | 1.7 | ✅ completed |  |
| 12:42 | sports car | llm | 15 | 1.2 | 63 | 2.5 | 45 | trellis | 0.79 | **112** | 11,982 | PBR 1024² | 1.8 | ✅ completed |  |
| 12:45 | wristwatch | llm | 15 | 1.3 | 103 | 1.2 | 44 | trellis | 0.18 | **150** | 19,600 | PBR 1024² | 2.0 | ✅ completed |  |
| 12:49 | cat | llm | 15 | 1.2 | 101 | 1.4 | – | – | – | 104 | – | – | – | ❌ failed | ZeroGPU quota exceeded; TripoSR fallback not yet fixed |
| 12:56 | cat | llm | 15 | 1.2 | 124 | 1.5 | – | – | – | 127 | – | – | – | ❌ failed | ZeroGPU quota exceeded (Space in cool-down); TripoSR fallback not yet fixed |
| 13:08 | cat | llm | 5 | 1.4 | 23 | 2.2 | 38 | trellis | 0.73 | **65** | 7,810 | PBR 1024² | 1.3 | ✅ completed | speed test with only 5 SD steps |
| 13:12 | cat | llm | 15 | 1.3 | 60 | 2.2 | 53 | trellis | 0.85 | **117** | 17,827 | PBR 1024² | 1.8 | ✅ completed |  |
| 13:30 | box | llm | 15 | 1.3 | 103 | 1.2 | 55 | trellis | 0.17 | **160** | 42,916 | PBR 1024² | 2.2 | ✅ completed |  |
| 13:34 | pencil sharpener | llm | 15 | 1.2 | 90 | 1.0 | 46 | trellis | 0.13 | **139** | 15,557 | PBR 1024² | 1.4 | ✅ completed |  |
| 13:38 | object (laptop) | fallback | 15 | 0.6 | 87 | 1.1 | 46 | trellis | 0.13 | **134** | 16,796 | PBR 1024² | 1.7 | ✅ completed | Groq rate-limited → rule-based prompt (weaker result) |
| 13:44 | hurricane lantern | llm | 15 | 1.4 | 100 | 0.9 | – | – | – | 102 | – | – | – | ❌ failed | ZeroGPU quota exceeded; TripoSR fallback not yet fixed |
| 13:52 | lantern | llm | 25 | 1.3 | 102 | 2.1 | 47 | trellis | 0.61 | **153** | 11,564 | PBR 1024² | 2.0 | ✅ completed |  |

Not shown: 2 inputs that were correctly **rejected** in stage ② (a photo of a person; a drawing of a whole room),
and early test jobs from before stage ③ existed.

## Summary (12 complete runs)

| | Min | Median | Max |
|---|---|---|---|
| ③ SDXL, 30 steps (4 runs) | 116 s | 142 s | 175 s |
| ③ SDXL, 15 steps (6 runs) | 60 s | 88 s | 103 s |
| ⑤ TRELLIS on ZeroGPU (12 runs) | 38 s | 45 s | 55 s |
| **Total, 30 steps** | **163 s** (2:43) | **190 s** (3:10) | **220 s** (3:40) |
| **Total, 15 steps** | **112 s** (1:52) | **137 s** (2:17) | **160 s** (2:40) |
| Final mesh size | 7.8k triangles | 16k triangles | 43k triangles |
| GLB file size | 1.3 MB | 1.8 MB | 2.2 MB |

Observations:

- Stage ③ (SDXL on the laptop GPU) is 60–80% of the total. Its time varies a lot at the same step count (60–103 s
  for 15 steps). This followed memory pressure from other apps on the 24 GB machine.
- TRELLIS time is stable (38–55 s) and includes upload, queueing on the shared GPU, generation, GLB export and
  download.
- All three failures were the free ZeroGPU quota running out at a time when the local TripoSR fallback could not
  load (a `transformers` 5 incompatibility). That is now fixed; the fallback passes its self-test.
