from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .cores import FiveStagePipelinedCore, SingleCycleCore
from .isa_rv32i import (
    encode_b_type,
    encode_halt,
    encode_i_type,
    encode_r_type,
    encode_s_type,
    encode_u_type,
)


def run_benchmark(seed: int = 42) -> List[Dict[str, Any]]:
    """Run reproducible RV32I Single-Cycle vs 5-Stage Pipelined Core benchmarks."""
    rng = random.Random(seed)

    # 1. ALU data forwarding + x0 hardwire stress program (with seeded ALU chain length)
    extra_alu_ops = rng.randint(1, 6)
    prog_alu = [
        encode_i_type(15, rs1=0, funct3=0x0, rd=1),
        encode_i_type(27, rs1=0, funct3=0x0, rd=2),
        encode_r_type(0x00, rs2=2, rs1=1, funct3=0x0, rd=3),
        encode_r_type(0x20, rs2=1, rs1=3, funct3=0x0, rd=4),
        encode_i_type(99, rs1=3, funct3=0x0, rd=0),
        encode_s_type(8, rs2=3, rs1=0),
    ]
    for step_idx in range(extra_alu_ops):
        prog_alu.append(encode_i_type(step_idx + 1, rs1=4, funct3=0x0, rd=5))
    prog_alu.append(encode_halt())

    ss_alu = SingleCycleCore(prog_alu).run()
    fs_alu = FiveStagePipelinedCore(prog_alu).run()

    # 2. Load-use hazard + speculative branch flush recovery program
    prog_hazard = [
        encode_i_type(0, rs1=0, funct3=0x2, rd=1, opcode=0x03),
        encode_i_type(55, rs1=0, funct3=0x0, rd=2),
        encode_i_type(4, rs1=0, funct3=0x2, rd=3, opcode=0x03),
        encode_b_type(8, rs2=2, rs1=3, funct3=0x0),
        encode_halt(),
        encode_i_type(77, rs1=0, funct3=0x0, rd=4),
        encode_halt(),
    ]
    dmem = {0: 55, 4: 55}
    ss_haz = SingleCycleCore(prog_hazard, initial_dmem=dmem).run()
    fs_haz = FiveStagePipelinedCore(prog_hazard, initial_dmem=dmem).run()

    # 3. Upper-immediate (LUI/AUIPC), subword memory (SB/LB/LBU/SH/LHU), and JALR jump
    prog_isa = [
        encode_u_type(0x12345, rd=1, opcode=0x37),
        encode_u_type(0x00001, rd=2, opcode=0x17),
        encode_i_type(-120, rs1=0, funct3=0x0, rd=3),
        encode_s_type(1, rs2=3, rs1=0, funct3=0x0),
        encode_i_type(1, rs1=0, funct3=0x0, rd=4, opcode=0x03),
        encode_i_type(1, rs1=0, funct3=0x4, rd=5, opcode=0x03),
        encode_i_type(0x24, rs1=0, funct3=0x0, rd=6),
        encode_i_type(0, rs1=6, funct3=0x0, rd=7, opcode=0x67),
        encode_i_type(999, rs1=0, funct3=0x0, rd=8),
        encode_s_type(2, rs2=1, rs1=0, funct3=0x1),
        encode_i_type(2, rs1=0, funct3=0x5, rd=9, opcode=0x03),
        encode_halt(),
    ]
    ss_isa = SingleCycleCore(prog_isa).run()
    fs_isa = FiveStagePipelinedCore(prog_isa).run()

    return [
        {
            "benchmark": "alu_forwarding_pipeline",
            "instructions_retired": fs_alu.instructions_retired,
            "pipeline_cycles": fs_alu.cycles,
            "pipeline_cpi": round(fs_alu.cpi, 4),
            "forwarding_events": fs_alu.forwarding_events,
            "load_use_stalls": fs_alu.load_use_stalls,
            "branch_flushes": fs_alu.branch_flushes,
            "architectural_parity": ss_alu.registers == fs_alu.registers and ss_alu.data_memory == fs_alu.data_memory,
        },
        {
            "benchmark": "load_to_branch_hazard_recovery",
            "instructions_retired": fs_haz.instructions_retired,
            "pipeline_cycles": fs_haz.cycles,
            "pipeline_cpi": round(fs_haz.cpi, 4),
            "forwarding_events": fs_haz.forwarding_events,
            "load_use_stalls": fs_haz.load_use_stalls,
            "branch_flushes": fs_haz.branch_flushes,
            "architectural_parity": ss_haz.registers == fs_haz.registers and ss_haz.data_memory == fs_haz.data_memory,
        },
        {
            "benchmark": "rv32i_subword_jalr_upper_imm",
            "instructions_retired": fs_isa.instructions_retired,
            "pipeline_cycles": fs_isa.cycles,
            "pipeline_cpi": round(fs_isa.cpi, 4),
            "forwarding_events": fs_isa.forwarding_events,
            "load_use_stalls": fs_isa.load_use_stalls,
            "branch_flushes": fs_isa.branch_flushes,
            "architectural_parity": ss_isa.registers == fs_isa.registers and ss_isa.data_memory == fs_isa.data_memory,
        },
    ]


def format_markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers = [
        "benchmark",
        "instructions_retired",
        "pipeline_cycles",
        "pipeline_cpi",
        "forwarding_events",
        "load_use_stalls",
        "branch_flushes",
        "architectural_parity",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible python-riscv-simulator RV32I Single-Cycle and 5-Stage Pipeline benchmarks."
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format for benchmark results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible benchmark execution.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file path to write benchmark output.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = run_benchmark(seed=args.seed)
    rendered = (
        json.dumps({"seed": args.seed, "benchmarks": rows}, indent=2)
        if args.format == "json"
        else format_markdown_table(rows)
    )
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"Error writing output to {args.output}: {exc}", file=sys.stderr)
            return 1
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
