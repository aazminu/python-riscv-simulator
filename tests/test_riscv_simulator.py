from __future__ import annotations

from python_riscv_simulator import (
    FiveStagePipelinedCore,
    SingleCycleCore,
    encode_b_type,
    encode_halt,
    encode_i_type,
    encode_j_type,
    encode_r_type,
    encode_s_type,
    encode_u_type,
)


def test_alu_forwarding_and_x0_hardwire_equivalence() -> None:
    # Program:
    # ADDI x1, x0, 15
    # ADDI x2, x0, 27
    # ADD  x3, x1, x2   (MEM/WB x1 + EX/MEM x2 -> x3 = 42, 2 forwards)
    # SUB  x4, x3, x1   (EX/MEM x3 -> x4 = 27, 1 forward)
    # ADDI x0, x3, 99   (MEM/WB x3 -> attempt to overwrite x0 -> must remain 0, 1 forward)
    # SW   x3, 8(x0)    (no hazard: x3 already retired to regfile)
    # HALT
    prog = [
        encode_i_type(15, rs1=0, funct3=0x0, rd=1),
        encode_i_type(27, rs1=0, funct3=0x0, rd=2),
        encode_r_type(0x00, rs2=2, rs1=1, funct3=0x0, rd=3),
        encode_r_type(0x20, rs2=1, rs1=3, funct3=0x0, rd=4),
        encode_i_type(99, rs1=3, funct3=0x0, rd=0),
        encode_s_type(8, rs2=3, rs1=0),
        encode_halt(),
    ]
    ss = SingleCycleCore(prog).run()
    fs = FiveStagePipelinedCore(prog).run()

    assert ss.registers == fs.registers
    assert ss.data_memory == fs.data_memory
    assert ss.cpi == 1.0
    assert fs.registers[0] == 0
    assert fs.registers[3] == 42
    assert fs.registers[4] == 27
    assert fs.data_memory[8] == 42
    assert fs.forwarding_events == 4


def test_load_use_hazard_stall_and_mem_wb_forwarding() -> None:
    # Initial DMEM[16] = 100
    # LW   x5, 16(x0)
    # ADDI x6, x5, 23   (Load-use hazard: must stall 1 cycle and forward 100 -> x6 = 123)
    # HALT
    prog = [
        encode_i_type(16, rs1=0, funct3=0x2, rd=5, opcode=0x03),
        encode_i_type(23, rs1=5, funct3=0x0, rd=6),
        encode_halt(),
    ]
    ss = SingleCycleCore(prog, initial_dmem={16: 100}).run()
    fs = FiveStagePipelinedCore(prog, initial_dmem={16: 100}).run()

    assert ss.registers == fs.registers
    assert fs.registers[5] == 100
    assert fs.registers[6] == 123
    assert fs.load_use_stalls == 1
    assert fs.forwarding_events == 1


def test_loop_branch_execution_and_state_parity() -> None:
    # Accumulate sum 1 + 2 + 3 = 6 in x2 using a backward BNE loop:
    # 0x00: ADDI x1, x0, 3      (counter = 3)
    # 0x04: ADDI x2, x0, 0      (acc = 0)
    # 0x08: ADD  x2, x2, x1     (acc += counter)
    # 0x0C: ADDI x1, x1, -1     (counter -= 1)
    # 0x10: BNE  x1, x0, -8     (if counter != 0, branch to 0x08)
    # 0x14: HALT
    prog = [
        encode_i_type(3, rs1=0, funct3=0x0, rd=1),
        encode_i_type(0, rs1=0, funct3=0x0, rd=2),
        encode_r_type(0x00, rs2=1, rs1=2, funct3=0x0, rd=2),
        encode_i_type(-1, rs1=1, funct3=0x0, rd=1),
        encode_b_type(-8, rs2=0, rs1=1, funct3=0x1),
        encode_halt(),
    ]
    ss = SingleCycleCore(prog).run()
    fs = FiveStagePipelinedCore(prog).run()

    assert ss.registers == fs.registers
    assert fs.registers[1] == 0
    assert fs.registers[2] == 6
    assert fs.branch_flushes == 2


def test_lw_into_beq_load_to_branch_hazard_and_speculative_halt_recovery() -> None:
    # Back-to-back LW -> BEQ where the taken branch jumps over a HALT instruction:
    # 0x00: LW   x1, 0(x0)      (loads 55 from DMEM[0])
    # 0x04: ADDI x2, x0, 55     (sets x2 = 55)
    # 0x08: LW   x3, 4(x0)      (loads 55 from DMEM[4])
    # 0x0C: BEQ  x3, x2, +8     (depends immediately on LW x3 -> 2 stall cycles, jumps to 0x14)
    # 0x10: HALT                (skipped when branch is taken)
    # 0x14: ADDI x4, x0, 77     (executed on taken branch path)
    # 0x18: HALT
    prog = [
        encode_i_type(0, rs1=0, funct3=0x2, rd=1, opcode=0x03),
        encode_i_type(55, rs1=0, funct3=0x0, rd=2),
        encode_i_type(4, rs1=0, funct3=0x2, rd=3, opcode=0x03),
        encode_b_type(8, rs2=2, rs1=3, funct3=0x0),
        encode_halt(),
        encode_i_type(77, rs1=0, funct3=0x0, rd=4),
        encode_halt(),
    ]
    dmem = {0: 55, 4: 55}
    ss = SingleCycleCore(prog, initial_dmem=dmem).run()
    fs = FiveStagePipelinedCore(prog, initial_dmem=dmem).run()

    assert ss.registers == fs.registers
    assert fs.registers[3] == 55
    assert fs.registers[4] == 77
    # Exactly 2 stall cycles for LW -> BEQ (ID/EX load stall + EX/MEM load stall), not double-counted
    assert fs.load_use_stalls == 2
    assert fs.branch_flushes == 1


def test_lui_auipc_jalr_and_byte_halfword_memory_parity() -> None:
    # 0x00: LUI   x1, 0x12345         -> x1 = 0x12345000
    # 0x04: AUIPC x2, 0x00001         -> x2 = 0x00001004
    # 0x08: ADDI  x3, x0, -120        -> x3 = 0xFFFFFF88 (low byte = 0x88 = -120)
    # 0x0C: SB    x3, 1(x0)           -> stores 0x88 at byte offset 1 of DMEM[0]
    # 0x10: LB    x4, 1(x0)           -> sign-extends 0x88 to 0xFFFFFF88
    # 0x14: LBU   x5, 1(x0)           -> zero-extends 0x88 to 0x00000088
    # 0x18: ADDI  x6, x0, 0x24        -> x6 = 0x24 (target PC for JALR)
    # 0x1C: JALR  x7, 0(x6)           -> x7 = 0x20, jumps to 0x24
    # 0x20: ADDI  x8, x0, 999         -> skipped!
    # 0x24: SH    x1, 2(x0)           -> stores low halfword of x1 (0x5000) at offset 2
    # 0x28: LHU   x9, 2(x0)           -> loads 0x5000 into x9
    # 0x2C: HALT
    prog = [
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
    ss = SingleCycleCore(prog).run()
    fs = FiveStagePipelinedCore(prog).run()

    assert ss.registers == fs.registers
    assert ss.data_memory == fs.data_memory
    assert fs.registers[1] == 0x12345000
    assert fs.registers[2] == 0x00001004
    assert fs.registers[4] == 0xFFFFFF88
    assert fs.registers[5] == 0x00000088
    assert fs.registers[7] == 0x20
    assert fs.registers[8] == 0
    assert fs.registers[9] == 0x5000


def test_cli_benchmark_markdown_json_and_reproducibility(
    capsys: object, tmp_path: object
) -> None:
    import json
    import subprocess
    import sys
    from pathlib import Path
    import pytest
    from python_riscv_simulator.__main__ import format_markdown_table, main, run_benchmark

    rows_1 = run_benchmark(seed=42)
    rows_2 = run_benchmark(seed=42)
    assert rows_1 == rows_2
    assert len(rows_1) == 3
    assert all(r["architectural_parity"] is True for r in rows_1)

    md = format_markdown_table(rows_1)
    assert "| benchmark | instructions_retired | pipeline_cycles |" in md
    assert "load_to_branch_hazard_recovery" in md
    assert "rv32i_subword_jalr_upper_imm" in md

    out_path = Path(str(tmp_path)) / "riscv_bench.json"
    rc = main(["--format", "json", "--seed", "42", "--output", str(out_path)])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert data["seed"] == 42
    assert data["benchmarks"] == rows_1
    assert json.loads(out_path.read_text(encoding="utf-8")) == data

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "benchmark.py"), "--format", "markdown"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "alu_forwarding_pipeline" in proc.stdout

    with pytest.raises(SystemExit):
        main(["--format", "invalid_fmt"])

