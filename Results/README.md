# Published experiment results

Each numbered experiment publishes its user-facing outputs under `Results/NNN` after the durable source files have
been written and validated. Large model files and statistics are NTFS hard links to their canonical files, not
copies. On filesystems that cannot create a hard link, publication attempts a symbolic link and fails rather than
silently duplicating the artifact.

Every experiment directory contains `publication.json`, which identifies each source file, published name, kind,
link type, and byte count. Deleting a published hard link does not delete its canonical source; rerunning the
experiment recreates or updates managed links. Publications from separate stages of the same experiment are merged,
so a later benchmark does not hide an earlier GGUF or statistics file. Do not replace an unmanaged file in a
numbered Results directory.

Existing outputs can be backfilled with `tools/publish_results.py`, for example:

```powershell
.\.venv\Scripts\python.exe tools\publish_results.py 2 `
  --statistics evidence\m9\002-gemma-3-1b-it-quality-benchmark.json `
  --report evidence\m9\002-gemma-3-1b-it-quality-benchmark.md
```
