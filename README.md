# agentVigil

## VideoLLaMA3 Notes

- The VideoLLaMA3 adapter currently captions per chunk (not full-video in one pass).
- Chunk context is local to each chunk. Cross-chunk reasoning is done later by the adaptive agent layer.
- Optional overlap can be enabled with `CHUNK_OVERLAP_SEC` in `config.py` to reduce boundary misses.

## Runtime Requirements

- Install Python deps from `requirements.txt`.
- For Windows video decoding stability, install FFmpeg and ensure `ffmpeg` is available on `PATH`.
- Temporary chunk frames/videos are written under:
- `AGENTVIGIL_TEMP_DIR` (if set), else
- system temp directory (`tempfile.gettempdir()`).