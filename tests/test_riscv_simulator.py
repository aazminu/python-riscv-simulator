from __future__ import annotations

from python_riscv_simulator import (
    FiveStagePipelinedCore,
    SingleCycleCore,
    encode_b_type,
    encode_halt,
    encode_i_type,
    encode_r_type,
    encode_s_type,
)


def test_alu_forwarding_and_x0_hardwire_equivalence() -> None:
    # Program:
    # ADDI x1, x0, 15
    # ADDI x2, x0, 27
    # ADD  x3, x1, x2   (EX/MEM & MEM/WB forwarding -> x3 = 42)
    # SUB  x4, x3, x1   (EX/MEM forwarding -> x4 = 27)
    # ADDI x0, x3, 99   (Attempt to overwrite x0 -> must remain 0)
    # SW   x3, 8(x0)    (Store 42 at DMEM[8])
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
    assert fs.registers[0] == 0
    assert fs.registers[3] == 42
    assert fs.registers[4] == 27
    assert fs.data_memory[8] == 42
    assert fs.forwarding_events >= 2


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
