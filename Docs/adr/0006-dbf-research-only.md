# ADR 0006: DBF remains research-only

Status: accepted

DBF remains available only as an explicitly selected, versioned research component. It is not part of the supported
production quantization or deployment compatibility surface. The first release may omit its implementation while
returning the stable `CAL004` unsupported-mode diagnostic; DBF artifacts never silently identify as NanoQuant ADMM.

This preserves historical replay work without requiring DBF parity, scaling, packing, and runtime support to block the
product path. Promotion would require CPU fixture parity, bit-accounting coverage, a packed-format decision, and an
updated compatibility table.

