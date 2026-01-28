# Model Container Format (MCF)

This document describes the purpose, structure, and long-term design principles of the Model Container File (MCF) format.

MCF is a single-file, random-access container for machine learning models, designed for predictable performance, long-term
stability, and explicit runtime control. Memory mapping is an optional acceleration; it must not be required.

## Core Principles

1. All references in the container use absolute file offsets. No in-memory pointers or relocations are implied.
2. All multi-byte numeric fields are encoded in little-endian byte order.
3. No section is implicitly required. Consumers must explicitly declare which sections are dependent and tolerate absence
   of others.
4. The container describes data and structure only; runtime behavior must remain explicit. Runtimes may consult explicit
   configuration fields (for example HF `config.json`) but should gate behavior on clearly defined keys.
5. Performance-critical data must be stored contiguously and separately from descriptive or infrequently accessed metadata.

## Non-Goals

- MCF does not define inference behavior.
- MCF does not mandate tokenization strategy.
- MCF does not guarantee model compatibility across runtimes.

## Compatibility

Readers must reject unsupported major versions and may ignore unknown minor versions or section types.

## File Layout

    [ Fixed Header ]
    [ Section Directory ]

    [ Section 1 ]
    [ Section 2 ]
    [ Section N ]

## Section Types (current)

- 0x0001: ModelInfo
- 0x0002: QuantInfo
- 0x0003: TensorIndex
- 0x0004: TensorData

- 0x0100: HF config.json
- 0x0101: HF generation_config.json
- 0x0102: tokenizer.json
- 0x0103: tokenizer_config.json
- 0x0104: vocab.json
- 0x0105: merges.txt

## Header Flags

- FlagTensorDataAligned64 (bit 0): indicates quantized tensor payloads follow the 64-byte internal sub-region alignment
  rules defined in this spec. Files that contain DTypeQ* or DTypeK* tensors MUST set this flag.

## Quantization (v1.0)

MCF v1.0 quantization is a performance-focused format that cleanly separates hot-path data (payload) from cold-path
metadata (QuantInfo). The payload stores precomputed values required for SIMD dequantization. The QuantInfo section stores
the clipping bounds used to derive those parameters for reproducibility and debugging.

### 1) Method Registry and Taxonomy

These unique identifiers bind `TensorDType` (for fast dispatch) to the QuantInfo method definition.

| ID (hex) | TensorDType | Family | Config | Role |
| --- | --- | --- | --- | --- |
| 0x10 | DTypeInt8 | Raw | 8-bit, tensor scale | Activations (asymmetric). |
| 0x11 | DTypeInt4 | Raw | 4-bit, tensor scale | Activations (low precision). |
| 0x20 | DTypeQ8 | Block | 8-bit, block 32 | High precision weights. |
| 0x21 | DTypeQ4 | Block | 4-bit, block 32 | Standard weights. |
| 0x30 | DTypeK6 | Super | 6-bit, block 32, super 256 | Efficient high-precision weights. |
| 0x31 | DTypeK4 | Super | 4-bit, block 32, super 256 | Workhorse weights. |
| 0x32 | DTypeK3 | Super | 3-bit, block 32, super 256 | Compressed weights. |
| 0x33 | DTypeK2 | Super | 2-bit, block 32, super 256 | Extreme compression. |

Excluded methods: q2, q3, q6, k8.

### 2) Domain Logic and Rules

QuantInfo records carry a domain flag that selects reconstruction logic:

- DomainWeights (0): symmetric. Z is 0. Range is [-max, max] with the most-negative code unused.
- DomainActivations (1): asymmetric. Z is derived from MinClip/MaxClip and stored in the payload where applicable.

Constraints:

- q* and k* are restricted to DomainWeights.
- int* supports DomainWeights and DomainActivations.

### 3) QuantInfo Section Payload (v1)

The QuantInfo section payload is little-endian and contains a small header followed by a fixed-size record array.
Records map 1:1 to TensorIndex entries by index.

```
QuantInfoHeader (8 bytes)
- u32 Version     (currently 1)
- u32 RecordCount

QuantRecord (24 bytes, fixed)
- u32 TensorIndex
- u8  Method
- u8  Domain
- u16 BlockSize
- u16 SuperSize
- u8[6] Reserved (must be zero)
- f32 MinClip
- f32 MaxClip
```

QuantRecord layout (byte offsets):

```
0  : u32 TensorIndex
4  : u8  Method
5  : u8  Domain
6  : u16 BlockSize
8  : u16 SuperSize
10 : u8[6] Reserved (padding/future use)
16 : f32 MinClip
20 : f32 MaxClip
```

### 4) Payload Layouts (Structure of Arrays)

Global alignment rule: every sub-region within a quantized tensor payload MUST start on a 64-byte boundary. Writers must
insert zero padding between regions when required. Offsets are derived at runtime using this rule:

```
align64(x) = (x + 63) & ^uint64(63)
```

Let rows = shape[0] and cols = shape[1]. Blocks are taken along the last dimension (cols). Define:

- blocksPerRow = ceil(cols / 32)
- superBlocksPerRow = ceil(blocksPerRow / 8)
- totalBlocks = rows * blocksPerRow
- totalSuper = rows * superBlocksPerRow

Padding applies per row: values beyond cols in the final block of each row are treated as zero.

#### Family int* (Raw)

Used for activations or generic integer storage.

```
[ Region 1: Scale ] (float32, 1 value)
-> align64
[ Region 2: Zero ] (float32, 1 value) present only if Domain=Activations
-> align64
[ Region 3: Data ] (int8/int4 array, packed)
```

#### Family q* (Block)

Standard block-wise quantization.

```
[ Region 1: BlockScales ] (float16 array, Count = totalBlocks)
-> align64
[ Region 2: QuantData ] (packed)
```

#### Family k* (Super-Block)

Hierarchical quantization: super-block 256 containing sub-blocks of 32.

```
[ Region 1: SuperScales ] (float16 array, Count = totalSuper)
-> align64
[ Region 2: SuperMins ] (float16 array, Count = totalSuper) present only if Domain=Activations
-> align64
[ Region 3: SubScales ] (uint8 array, Count = totalBlocks)
  - each byte stores one 6-bit value in bits 0-5, top 2 bits zero
-> align64
[ Region 4: QuantData ] (packed)
```

### 5) Bit Packing

General rules:

- Byte order: little-endian.
- Fill order: LSB first (bit 0 filled before bit 7).

Packing by bit width:

- k2 (2-bit): 4 items per byte.
  - [ item3 (bits 6-7) | item2 (4-5) | item1 (2-3) | item0 (0-1) ]
- k3 (3-bit): tight stream, 32 items in 12 bytes. Pack contiguous 3-bit chunks.
- k4 / q4 / int4 (4-bit): 2 items per byte.
  - item0 = low nibble (bits 0-3), item1 = high nibble (bits 4-7)
- k6 (6-bit): tight stream, 4 items in 3 bytes (24 bits total).

For symmetric weights (q* and k*), quantized values are signed integers within the bit width. The most-negative code is
unused to keep Z=0 (for example, 4-bit uses [-7, 7]).

### 6) Reconstruction Formulas

Symmetric weights (q*, k*):

```
w = S_block * q_val
```

For k*:

```
S_block = S_super * (u6_subscale / 32.0)
w = S_block * q_val
```

Asymmetric activations (int*):

```
x = S_global * (q_val - Z_global)
```

### 7) Invariants

1. For each row, if cols is not a multiple of 32, QuantData is padded with zeros to the next block boundary within that row.
2. Offsets are derived at runtime using 64-byte alignment:
   OffsetN = (OffsetN-1 + SizeN-1 + 63) & ^63
3. If DomainWeights and method is int*, the Zero field is omitted from the payload (Z is assumed 0).
4. QuantRecord reserved bytes must be zero.
