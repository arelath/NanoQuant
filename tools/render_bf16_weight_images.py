"""Render source BF16 weights only for projections present in the NanoQuant GGUF."""

from weight_image_common import run

if __name__ == "__main__":
    run("bf16")
