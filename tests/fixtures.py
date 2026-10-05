"""Small GGUF fixtures and an independent double-precision forward reference."""
import math
import struct
import sys
from pathlib import Path


def string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def value(data):
    if isinstance(data, str):
        return 8, string(data)
    if isinstance(data, bool):
        return 7, bytes([data])
    if isinstance(data, int):
        return 4, struct.pack("<I", data)
    if isinstance(data, float):
        return 6, struct.pack("<f", data)
    subtype = 8 if isinstance(data[0], str) else 4
    return 9, struct.pack("<IQ", subtype, len(data)) + b"".join(value(x)[1] for x in data)


def write_gguf(path, metadata, tensors):
    header = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(metadata))
    for key, data in metadata.items():
        kind, payload = value(data)
        header += string(key) + struct.pack("<I", kind) + payload
    offset = 0
    body = bytearray()
    for name, dims, kind, payload in tensors:
        header += string(name) + struct.pack("<I", len(dims))
        header += struct.pack("<" + "Q" * len(dims), *dims) + struct.pack("<IQ", kind, offset)
        body.extend(payload)
        padding = (-len(payload)) % 32
        body.extend(bytes(padding))
        offset += len(payload) + padding
    header += bytes((-len(header)) % 32)
    path.write_bytes(header + body)


def encoder():
    extra = 256
    result = []
    for i in range(256):
        if 33 <= i <= 126 or 161 <= i <= 172 or 174 <= i <= 255:
            result.append(chr(i))
        else:
            result.append(chr(extra))
            extra += 1
    return result


def fixture(path, quantized, fault=None):
    tokens = encoder() + ["ab", "abc", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    metadata = {
        "general.architecture": "qwen3", "general.alignment": 32,
        "qwen3.embedding_length": 32, "qwen3.feed_forward_length": 64,
        "qwen3.attention.head_count": 4, "qwen3.attention.head_count_kv": 2,
        "qwen3.attention.key_length": 8, "qwen3.attention.value_length": 8,
        "qwen3.block_count": 1, "qwen3.context_length": 32,
        "qwen3.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen3.rope.freq_base": 10000.0, "qwen3.rope.dimension_count": 8,
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": tokens, "tokenizer.ggml.token_type": [1] * 258 + [3] * 3,
        "tokenizer.ggml.merges": ["a b", "ab c"],
        "tokenizer.ggml.eos_token_id": 260, "tokenizer.ggml.bos_token_id": 258,
        "tokenizer.ggml.add_bos_token": False,
    }
    tensors, weights = [], {}
    descriptions = [
        ("token_embd", (32, len(tokens))), ("output_norm", (32,)),
        ("blk.0.attn_norm", (32,)), ("blk.0.attn_q", (32, 32)),
        ("blk.0.attn_k", (32, 16)), ("blk.0.attn_v", (32, 16)),
        ("blk.0.attn_output", (32, 32)), ("blk.0.attn_q_norm", (8,)),
        ("blk.0.attn_k_norm", (8,)), ("blk.0.ffn_norm", (32,)),
        ("blk.0.ffn_gate", (32, 64)), ("blk.0.ffn_up", (32, 64)),
        ("blk.0.ffn_down", (64, 32)),
    ]
    for seed, (name, dims) in enumerate(descriptions):
        norm = len(dims) == 1
        data = [(1.0 if norm else 0.0) + (0.1 if norm else 0.07) * math.sin(i * 1.7 + seed)
                for i in range(math.prod(dims))]
        kind = 8 if quantized and not norm else (1 if name.endswith("q_norm") else 30 if name.endswith("k_norm") else 0)
        if kind == 8:
            payload, restored = bytearray(), []
            scale = struct.unpack("<e", struct.pack("<e", 0.001))[0]
            for start in range(0, len(data), 32):
                q = [round(x / scale) for x in data[start:start + 32]]
                payload.extend(struct.pack("<e32b", scale, *q))
                restored.extend(x * scale for x in q)
        elif kind == 1:
            payload = struct.pack("<" + "e" * len(data), *data)
            restored = list(struct.unpack("<" + "e" * len(data), payload))
        elif kind == 30:
            bits = [struct.unpack("<I", struct.pack("<f", x))[0] >> 16 for x in data]
            payload = struct.pack("<" + "H" * len(data), *bits)
            restored = [struct.unpack("<f", struct.pack("<I", x << 16))[0] for x in bits]
        else:
            payload = struct.pack("<" + "f" * len(data), *data)
            restored = list(struct.unpack("<" + "f" * len(data), payload))
        tensors.append((name + ".weight", dims, kind, payload))
        weights[name] = restored
    if fault == "alignment":
        metadata["general.alignment"] = 3
    elif fault == "architecture":
        metadata["general.architecture"] = "llama"
    elif fault == "block-count":
        metadata["qwen3.block_count"] = 1000000
    elif fault == "head-size":
        metadata["qwen3.attention.key_length"] = 7
    elif fault == "tokenizer":
        metadata["tokenizer.ggml.pre"] = "unsupported"
    elif fault == "duplicate":
        tensors.append(tensors[-1])
    elif fault == "encoding":
        name, dims, kind, payload = tensors[-1]
        tensors[-1] = name, dims, 2, payload
    elif fault == "q8-width":
        name, dims, kind, payload = tensors[-1]
        tensors[-1] = name, (33, 32), 8, payload
    elif fault == "missing":
        tensors = tensors[:-1]
    write_gguf(path, metadata, tensors)
    return weights


def norm(x, w):
    scale = 1 / math.sqrt(sum(v * v for v in x) / len(x) + 1e-6)
    return [v * scale * a for v, a in zip(x, w)]


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def quantize(x):
    out = []
    for start in range(0, len(x), 32):
        block = [f32(v) for v in x[start:start + 32]]
        scale = f32(max(abs(v) for v in block) / 127)
        inverse = f32(1 / scale) if scale else 0
        stored = struct.unpack("<e", struct.pack("<e", scale))[0]
        out.extend(round(f32(v * inverse)) * stored for v in block)
    return out


QUANTIZED = False


def matvec(w, x):
    if QUANTIZED:
        x = quantize(x)
    return [sum(a * b for a, b in zip(w[start:start + len(x)], x))
            for start in range(0, len(w), len(x))]


def forward(weights, tokens):
    keys, values, result = [], [], []
    for position, token in enumerate(tokens):
        x = weights["token_embd"][token * 32:(token + 1) * 32]
        nx = norm(x, weights["blk.0.attn_norm"])
        q = matvec(weights["blk.0.attn_q"], nx)
        k = matvec(weights["blk.0.attn_k"], nx)
        v = matvec(weights["blk.0.attn_v"], nx)
        for vector, heads, name in [(q, 4, "q"), (k, 2, "k")]:
            for head in range(heads):
                start = head * 8
                vector[start:start + 8] = norm(vector[start:start + 8], weights[f"blk.0.attn_{name}_norm"])
                for i in range(4):
                    angle = position * 10000 ** (-2 * i / 8)
                    a, b = vector[start + i], vector[start + i + 4]
                    vector[start + i] = a * math.cos(angle) - b * math.sin(angle)
                    vector[start + i + 4] = a * math.sin(angle) + b * math.cos(angle)
        keys.append(k)
        values.append(v)
        attention = []
        for head in range(4):
            qs, ks = head * 8, (head // 2) * 8
            scores = [sum(a * b for a, b in zip(q[qs:qs + 8], key[ks:ks + 8])) / math.sqrt(8) for key in keys]
            scores = [math.exp(s - max(scores)) for s in scores]
            scores = [s / sum(scores) for s in scores]
            attention.extend(sum(s * val[ks + i] for s, val in zip(scores, values)) for i in range(8))
        x = [a + b for a, b in zip(x, matvec(weights["blk.0.attn_output"], attention))]
        nx = norm(x, weights["blk.0.ffn_norm"])
        gate = matvec(weights["blk.0.ffn_gate"], nx)
        up = matvec(weights["blk.0.ffn_up"], nx)
        activated = [g / (1 + math.exp(-g)) * u for g, u in zip(gate, up)]
        x = [a + b for a, b in zip(x, matvec(weights["blk.0.ffn_down"], activated))]
        result.append(matvec(weights["token_embd"], norm(x, weights["output_norm"])))
    return result


def main(directory):
    global QUANTIZED
    directory.mkdir(parents=True, exist_ok=True)
    for quantized in [False, True]:
        name = "q8" if quantized else "f32"
        QUANTIZED = quantized
        weights = fixture(directory / (name + ".gguf"), quantized)
        rows = forward(weights, [97, 98, 99])
        (directory / (name + ".logits")).write_bytes(b"".join(struct.pack("<" + "f" * len(row), *row) for row in rows))
    for fault in ["alignment", "architecture", "block-count", "head-size", "tokenizer", "duplicate", "encoding", "q8-width", "missing"]:
        fixture(directory / (fault + ".gguf"), False, fault)
    data = (directory / "f32.gguf").read_bytes()
    for name, changed in [("truncated", data[:-1]), ("version", data[:4] + struct.pack("<I", 2) + data[8:]),
                          ("magic", b"FAIL" + data[4:])]:
        (directory / (name + ".gguf")).write_bytes(changed)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
