from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .isa_rv32i import (
    NOP_INSTRUCTION,
    DecodedInstruction,
    decode_rv32i,
    evaluate_branch,
    execute_alu,
    sign_extend,
    to_i32,
    to_u32,
)


def _read_mem(dmem: Dict[int, int], addr: int, funct3: int) -> int:
    word_addr = addr & ~0x3
    byte_offset = addr & 0x3
    word = dmem.get(word_addr, 0)
    if funct3 == 0x0:  # LB
        b = (word >> (8 * byte_offset)) & 0xFF
        return to_u32(sign_extend(b, 8))
    if funct3 == 0x4:  # LBU
        return (word >> (8 * byte_offset)) & 0xFF
    if funct3 == 0x1:  # LH
        shift = 8 * (byte_offset & ~0x1)
        h = (word >> shift) & 0xFFFF
        return to_u32(sign_extend(h, 16))
    if funct3 == 0x5:  # LHU
        shift = 8 * (byte_offset & ~0x1)
        return (word >> shift) & 0xFFFF
    return to_u32(word)


def _write_mem(dmem: Dict[int, int], addr: int, val: int, funct3: int) -> None:
    word_addr = addr & ~0x3
    byte_offset = addr & 0x3
    if funct3 == 0x0:  # SB
        shift = 8 * byte_offset
        cur = dmem.get(word_addr, 0)
        dmem[word_addr] = to_u32((cur & ~(0xFF << shift)) | ((val & 0xFF) << shift))
    elif funct3 == 0x1:  # SH
        shift = 8 * (byte_offset & ~0x1)
        cur = dmem.get(word_addr, 0)
        dmem[word_addr] = to_u32((cur & ~(0xFFFF << shift)) | ((val & 0xFFFF) << shift))
    else:  # SW
        dmem[word_addr] = to_u32(val)


@dataclass(frozen=True)
class CorePerformanceMetrics:
    core_type: str
    cycles: int
    instructions_retired: int
    cpi: float
    ipc: float
    forwarding_events: int
    load_use_stalls: int
    branch_flushes: int
    registers: List[int]
    data_memory: Dict[int, int]


class SingleCycleCore:
    """Cycle-accurate Single-Cycle RV32I processor model."""

    def __init__(self, program_words: Sequence[int], initial_dmem: Dict[int, int] | None = None) -> None:
        self.imem: List[int] = [to_u32(w) for w in program_words]
        self.dmem: Dict[int, int] = {addr: to_u32(val) for addr, val in (initial_dmem or {}).items()}
        self.regs: List[int] = [0] * 32
        self.pc: int = 0
        self.cycles: int = 0
        self.retired: int = 0
        self.halted: bool = False

    def _write_reg(self, rd: int, val: int) -> None:
        if rd != 0:
            self.regs[rd] = to_u32(val)
        self.regs[0] = 0

    def step(self) -> None:
        if self.halted:
            return
        self.cycles += 1
        word_idx = self.pc // 4
        raw = self.imem[word_idx] if 0 <= word_idx < len(self.imem) else 0xFFFFFFFF
        inst = decode_rv32i(raw)

        if inst.is_halt:
            self.retired += 1
            self.halted = True
            return

        v1 = self.regs[inst.rs1]
        v2 = self.regs[inst.rs2]
        next_pc = self.pc + 4

        if inst.opcode == 0x33:  # R-type
            res = execute_alu(inst, v1, v2)
            self._write_reg(inst.rd, res)
        elif inst.opcode == 0x13:  # I-type
            res = execute_alu(inst, v1, inst.imm)
            self._write_reg(inst.rd, res)
        elif inst.opcode == 0x37:  # LUI
            self._write_reg(inst.rd, inst.imm)
        elif inst.opcode == 0x17:  # AUIPC
            self._write_reg(inst.rd, execute_alu(inst, self.pc, inst.imm))
        elif inst.mem_read:  # LB, LH, LW, LBU, LHU
            addr = execute_alu(inst, v1, inst.imm)
            self._write_reg(inst.rd, _read_mem(self.dmem, addr, inst.funct3))
        elif inst.mem_write:  # SB, SH, SW
            addr = execute_alu(inst, v1, inst.imm)
            _write_mem(self.dmem, addr, v2, inst.funct3)
        elif inst.is_branch:
            if evaluate_branch(inst, v1, v2):
                next_pc = self.pc + inst.imm
        elif inst.is_jump:  # JAL or JALR
            self._write_reg(inst.rd, self.pc + 4)
            if inst.opcode == 0x67:  # JALR
                next_pc = (to_i32(v1) + inst.imm) & ~1
            else:
                next_pc = self.pc + inst.imm

        self.pc = next_pc
        self.retired += 1

    def run(self, max_cycles: int = 2000) -> CorePerformanceMetrics:
        while not self.halted and self.cycles < max_cycles:
            self.step()
        total_cycles = self.cycles
        cpi = total_cycles / max(self.retired, 1)
        ipc = self.retired / max(total_cycles, 1)
        return CorePerformanceMetrics(
            core_type="SingleCycle",
            cycles=total_cycles,
            instructions_retired=self.retired,
            cpi=round(cpi, 4),
            ipc=round(ipc, 4),
            forwarding_events=0,
            load_use_stalls=0,
            branch_flushes=0,
            registers=list(self.regs),
            data_memory=dict(self.dmem),
        )


@dataclass
class _PipeLatch:
    valid: bool = False
    pc: int = 0
    inst: DecodedInstruction = field(default_factory=lambda: NOP_INSTRUCTION)
    v1: int = 0
    v2: int = 0
    alu_out: int = 0
    mem_data: int = 0


class FiveStagePipelinedCore:
    """5-stage pipelined RV32I core (IF/ID/EX/MEM/WB) with EX/MEM & MEM/WB forwarding and load-use stalls."""

    def __init__(self, program_words: Sequence[int], initial_dmem: Dict[int, int] | None = None) -> None:
        self.imem: List[int] = [to_u32(w) for w in program_words]
        self.dmem: Dict[int, int] = {addr: to_u32(val) for addr, val in (initial_dmem or {}).items()}
        self.regs: List[int] = [0] * 32
        self.pc: int = 0
        self.fetch_halted: bool = False

        self.if_id = _PipeLatch()
        self.id_ex = _PipeLatch()
        self.ex_mem = _PipeLatch()
        self.mem_wb = _PipeLatch()

        self.cycles: int = 0
        self.retired: int = 0
        self.forwarding_events: int = 0
        self.load_use_stalls: int = 0
        self.branch_flushes: int = 0

    def _write_reg(self, rd: int, val: int) -> None:
        if rd != 0:
            self.regs[rd] = to_u32(val)
        self.regs[0] = 0

    def _resolve_operand(self, reg_idx: int, raw_val: int, from_id_stage: bool = False) -> int:
        if reg_idx == 0:
            return 0
        # Priority 1: EX/MEM forwarding (youngest in-flight ALU producer)
        if (
            self.ex_mem.valid
            and self.ex_mem.inst.reg_write
            and not self.ex_mem.inst.mem_read
            and self.ex_mem.inst.rd == reg_idx
        ):
            self.forwarding_events += 1
            return self.ex_mem.alu_out
        # Priority 2: MEM/WB forwarding (needed in EX since ID/EX latched before split-cycle WB)
        if (
            not from_id_stage
            and self.mem_wb.valid
            and self.mem_wb.inst.reg_write
            and self.mem_wb.inst.rd == reg_idx
        ):
            self.forwarding_events += 1
            return self.mem_wb.mem_data if self.mem_wb.inst.mem_read else self.mem_wb.alu_out
        return raw_val

    def _uses_rs1(self, inst: DecodedInstruction) -> bool:
        return inst.opcode in (0x33, 0x13, 0x03, 0x23, 0x63, 0x67)

    def _uses_rs2(self, inst: DecodedInstruction) -> bool:
        return inst.opcode in (0x33, 0x23, 0x63)

    def is_done(self) -> bool:
        return (
            self.fetch_halted
            and not self.if_id.valid
            and not self.id_ex.valid
            and not self.ex_mem.valid
            and not self.mem_wb.valid
        )

    def step(self) -> None:
        self.cycles += 1

        # 1. WB stage: commit register writeback first (split-cycle register file write-before-read)
        if self.mem_wb.valid:
            wb_inst = self.mem_wb.inst
            if not wb_inst.is_halt:
                if wb_inst.reg_write:
                    val = self.mem_wb.mem_data if wb_inst.mem_read else self.mem_wb.alu_out
                    self._write_reg(wb_inst.rd, val)
            self.retired += 1

        # 2. MEM stage -> produces next_mem_wb
        next_mem_wb = _PipeLatch()
        if self.ex_mem.valid:
            m_inst = self.ex_mem.inst
            mem_read_val = 0
            if not m_inst.is_halt:
                if m_inst.mem_read:
                    mem_read_val = _read_mem(self.dmem, self.ex_mem.alu_out, m_inst.funct3)
                elif m_inst.mem_write:
                    _write_mem(self.dmem, self.ex_mem.alu_out, self.ex_mem.v2, m_inst.funct3)
            next_mem_wb = _PipeLatch(
                valid=True,
                pc=self.ex_mem.pc,
                inst=m_inst,
                alu_out=self.ex_mem.alu_out,
                mem_data=mem_read_val,
            )

        # 3. EX stage -> produces next_ex_mem (using EX/MEM and MEM/WB forwarding)
        next_ex_mem = _PipeLatch()
        if self.id_ex.valid:
            x_inst = self.id_ex.inst
            if x_inst.is_halt:
                next_ex_mem = _PipeLatch(valid=True, pc=self.id_ex.pc, inst=x_inst)
            else:
                op1 = self._resolve_operand(x_inst.rs1, self.id_ex.v1) if self._uses_rs1(x_inst) else 0
                op2_reg = self._resolve_operand(x_inst.rs2, self.id_ex.v2) if self._uses_rs2(x_inst) else 0
                if x_inst.is_jump:
                    alu_res = self.id_ex.pc + 4
                elif x_inst.opcode == 0x37:  # LUI
                    alu_res = x_inst.imm
                elif x_inst.opcode == 0x17:  # AUIPC
                    alu_res = execute_alu(x_inst, self.id_ex.pc, x_inst.imm)
                elif x_inst.opcode == 0x33:
                    alu_res = execute_alu(x_inst, op1, op2_reg)
                else:
                    alu_res = execute_alu(x_inst, op1, x_inst.imm)
                next_ex_mem = _PipeLatch(
                    valid=True,
                    pc=self.id_ex.pc,
                    inst=x_inst,
                    v1=op1,
                    v2=op2_reg,
                    alu_out=alu_res,
                )

        # 4. Hazard Detection & ID stage
        stall_pipeline = False
        if self.if_id.valid and not self.if_id.inst.is_halt:
            d_inst = self.if_id.inst
            needs_id_resolve = d_inst.is_branch or d_inst.opcode == 0x67
            # Load-use hazard: load in ID/EX whose rd is needed by instruction in IF/ID
            if (
                self.id_ex.valid
                and self.id_ex.inst.mem_read
                and self.id_ex.inst.rd != 0
                and (
                    (self._uses_rs1(d_inst) and d_inst.rs1 == self.id_ex.inst.rd)
                    or (self._uses_rs2(d_inst) and d_inst.rs2 == self.id_ex.inst.rd)
                )
            ):
                stall_pipeline = True
                self.load_use_stalls += 1
            # Branch or JALR in ID waiting on ALU producer currently in ID/EX or load in EX/MEM
            elif needs_id_resolve:
                if (
                    self.id_ex.valid
                    and self.id_ex.inst.reg_write
                    and self.id_ex.inst.rd != 0
                    and (
                        (self._uses_rs1(d_inst) and d_inst.rs1 == self.id_ex.inst.rd)
                        or (self._uses_rs2(d_inst) and d_inst.rs2 == self.id_ex.inst.rd)
                    )
                ):
                    stall_pipeline = True
                    self.load_use_stalls += 1
                elif (
                    self.ex_mem.valid
                    and self.ex_mem.inst.mem_read
                    and self.ex_mem.inst.rd != 0
                    and (
                        (self._uses_rs1(d_inst) and d_inst.rs1 == self.ex_mem.inst.rd)
                        or (self._uses_rs2(d_inst) and d_inst.rs2 == self.ex_mem.inst.rd)
                    )
                ):
                    stall_pipeline = True
                    self.load_use_stalls += 1

        next_id_ex = _PipeLatch()
        next_if_id = self.if_id
        branch_redirect_pc: int | None = None

        if stall_pipeline:
            # Insert bubble into ID/EX and freeze IF/ID + PC
            next_id_ex = _PipeLatch(valid=False)
        else:
            if self.if_id.valid:
                d_inst = self.if_id.inst
                if d_inst.is_halt:
                    self.fetch_halted = True
                    next_id_ex = _PipeLatch(valid=True, pc=self.if_id.pc, inst=d_inst)
                elif d_inst.is_branch:
                    rv1 = self._resolve_operand(d_inst.rs1, self.regs[d_inst.rs1], from_id_stage=True)
                    rv2 = self._resolve_operand(d_inst.rs2, self.regs[d_inst.rs2], from_id_stage=True)
                    if evaluate_branch(d_inst, rv1, rv2):
                        branch_redirect_pc = self.if_id.pc + d_inst.imm
                        self.branch_flushes += 1
                    next_id_ex = _PipeLatch(valid=True, pc=self.if_id.pc, inst=d_inst, v1=rv1, v2=rv2)
                elif d_inst.is_jump:
                    rv1 = (
                        self._resolve_operand(d_inst.rs1, self.regs[d_inst.rs1], from_id_stage=True)
                        if d_inst.opcode == 0x67
                        else 0
                    )
                    if d_inst.opcode == 0x67:
                        branch_redirect_pc = (to_i32(rv1) + d_inst.imm) & ~1
                    else:
                        branch_redirect_pc = self.if_id.pc + d_inst.imm
                    self.branch_flushes += 1
                    next_id_ex = _PipeLatch(valid=True, pc=self.if_id.pc, inst=d_inst, v1=rv1, v2=0)
                else:
                    next_id_ex = _PipeLatch(
                        valid=True,
                        pc=self.if_id.pc,
                        inst=d_inst,
                        v1=self.regs[d_inst.rs1],
                        v2=self.regs[d_inst.rs2],
                    )

            # 5. IF stage
            if branch_redirect_pc is not None:
                self.pc = branch_redirect_pc
                self.fetch_halted = False
            if self.fetch_halted:
                next_if_id = _PipeLatch(valid=False)
            else:
                word_idx = self.pc // 4
                raw = self.imem[word_idx] if 0 <= word_idx < len(self.imem) else 0xFFFFFFFF
                f_inst = decode_rv32i(raw)
                next_if_id = _PipeLatch(valid=True, pc=self.pc, inst=f_inst)
                if f_inst.is_halt:
                    self.fetch_halted = True
                else:
                    self.pc += 4

        self.mem_wb = next_mem_wb
        self.ex_mem = next_ex_mem
        self.id_ex = next_id_ex
        self.if_id = next_if_id

    def run(self, max_cycles: int = 2000) -> CorePerformanceMetrics:
        while not self.is_done() and self.cycles < max_cycles:
            self.step()
        cpi = self.cycles / max(self.retired, 1)
        ipc = self.retired / max(self.cycles, 1)
        return CorePerformanceMetrics(
            core_type="FiveStagePipelined",
            cycles=self.cycles,
            instructions_retired=self.retired,
            cpi=round(cpi, 4),
            ipc=round(ipc, 4),
            forwarding_events=self.forwarding_events,
            load_use_stalls=self.load_use_stalls,
            branch_flushes=self.branch_flushes,
            registers=list(self.regs),
            data_memory=dict(self.dmem),
        )
