# Model Container Format (MCF)

This document describes the purpose, structure, and long-term design principles of the Model Container File (MCF) format.

MCF is a single-file, memory-mappable container for machine learning models, designed for predictable performance, long-term stability, and explicit runtime control.

## Core Principals

1. All references in the container use absolute file offsets. No in-memory pointers or relocations are implid.
2. All multi-byte numeric fields are encoded in little-endian byte order
3. No section is implicitely required, consumers must explicitely declare which sections are dependent and tolerate absence of others.
4. The container describes data and structure only, runtime behaviour is never inferred from container contents.
5. Performance critical data must be stored contiguously and separately from descriptive or infrequently accessed metadata.

## Non-Goals

- MCF doesn't define inference behaviour
- MCF doesn't mandate tokenisation strategy
- MCF doesn't guarantee model compatibility across runtimes

## Compatibility

Readers must reject unsupported major versions and may ignore unknown minor versions or section types.

## File Layout

    [ Fixed Header ]
    [ Section Discovery ]

    [ Section 1 ]
    [ Section 2 ]
    [ Section N ]

Optional future sections may include:
    <!-- [ Tokenizer ] -->
    <!-- [ Chat Template ] -->
    <!-- [ Vocab ] -->

