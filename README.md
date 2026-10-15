# Cycle-Accurate Python RISC-V Simulator (`python-riscv-simulator`)

Cycle-accurate **RV32I** Single-Cycle (`SS`) and 5-Stage Pipelined (`FS`: `IF -> ID -> EX -> MEM -> WB`) RISC-V microarchitecture simulator in Python.

## Features

1. **RV32I Instruction Decoder & Assembler Helpers (`isa_rv32i.py`)**:
   - Encodes and decodes 32-bit R, I, S, B, and J-format instructions (`ADD`, `SUB`, `XOR`, `OR`, `AND`, `SLL`, `SRL`, `SRA`, `SLT`, `SLTU`, `ADDI`, `XORI`, `ORI`, `ANDI`, `LW`, `SW`, `BEQ`, `BNE`, `BLT`, `BGE`, `JAL`, `HALT`) with exact sign-extension and two's-complement 32-bit arithmetic.
2. **Single-Cycle Core (`cores.py::SingleCycleCore`)**:
   - Reference architectural execution engine retiring one instruction per clock edge (`x0` hardwired to `0`).
3. **5-Stage Pipelined Core (`cores.py::FiveStagePipelinedCore`)**:
   - **Data Forwarding Unit**: Bypasses `EX/MEM` and `MEM/WB` ALU results directly into `EX` operand multiplexers.
   - **Load-Use Hazard Unit**: Detects `LW` in `EX` whose destination `rd` matches `rs1` or `rs2` in `ID`, stalls `PC` and `IF/ID` for one cycle, and injects a `NOP` bubble into `ID/EX`.
   - **Control Hazard Handling**: Flushes the speculative `IF/ID` slot on taken branches/jumps and records cycle-by-cycle `CPI`, `IPC`, forwarding counts, and stall statistics.
