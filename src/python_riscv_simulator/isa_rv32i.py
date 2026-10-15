from __future__ import annotations

from dataclasses import dataclass

MASK32 = 0xFFFFFFFF


def to_u32(val: int) -> int:
    return val & MASK32


def to_i32(val: int) -> int:
    u = val & MASK32
    return u - 0x100000000 if (u & 0x80000000) else u


def sign_extend(val: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    return (val & (sign_bit - 1)) - (val & sign_bit)


@dataclass(frozen=True)
class DecodedInstruction:
    raw: int
    mnemonic: str
    opcode: int
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    funct3: int = 0
    funct7: int = 0
    imm: int = 0
    reg_write: bool = False
    mem_read: bool = False
    mem_write: bool = False
    is_branch: bool = False
    is_jump: bool = False
    is_halt: bool = False


NOP_INSTRUCTION = DecodedInstruction(raw=0x00000013, mnemonic="NOP", opcode=0x13)


def encode_r_type(funct7: int, rs2: int, rs1: int, funct3: int, rd: int, opcode: int = 0x33) -> int:
    return (
        ((funct7 & 0x7F) << 25)
        | ((rs2 & 0x1F) << 20)
        | ((rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | ((rd & 0x1F) << 7)
        | (opcode & 0x7F)
    )


def encode_i_type(imm: int, rs1: int, funct3: int, rd: int, opcode: int = 0x13) -> int:
    return (
        ((imm & 0xFFF) << 20)
        | ((rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | ((rd & 0x1F) << 7)
        | (opcode & 0x7F)
    )


def encode_s_type(imm: int, rs2: int, rs1: int, funct3: int = 0x2, opcode: int = 0x23) -> int:
    imm12 = imm & 0xFFF
    imm_hi = (imm12 >> 5) & 0x7F
    imm_lo = imm12 & 0x1F
    return (
        (imm_hi << 25)
        | ((rs2 & 0x1F) << 20)
        | ((rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | (imm_lo << 7)
        | (opcode & 0x7F)
    )


def encode_b_type(imm: int, rs2: int, rs1: int, funct3: int, opcode: int = 0x63) -> int:
    if imm % 2 != 0:
        raise ValueError("B-type immediate must be 2-byte aligned")
    imm13 = imm & 0x1FFF
    b12 = (imm13 >> 12) & 0x1
    b11 = (imm13 >> 11) & 0x1
    b10_5 = (imm13 >> 5) & 0x3F
    b4_1 = (imm13 >> 1) & 0xF
    return (
        (b12 << 31)
        | (b10_5 << 25)
        | ((rs2 & 0x1F) << 20)
        | ((rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | (b4_1 << 8)
        | (b11 << 7)
        | (opcode & 0x7F)
    )


def encode_u_type(imm20: int, rd: int, opcode: int = 0x37) -> int:
    return ((imm20 & 0xFFFFF) << 12) | ((rd & 0x1F) << 7) | (opcode & 0x7F)


def encode_j_type(imm: int, rd: int, opcode: int = 0x6F) -> int:
    if imm % 2 != 0:
        raise ValueError("J-type immediate must be 2-byte aligned")
    imm21 = imm & 0x1FFFFF
    b20 = (imm21 >> 20) & 0x1
    b19_12 = (imm21 >> 12) & 0xFF
    b11 = (imm21 >> 11) & 0x1
    b10_1 = (imm21 >> 1) & 0x3FF
    return (
        (b20 << 31)
        | (b10_1 << 21)
        | (b11 << 20)
        | (b19_12 << 12)
        | ((rd & 0x1F) << 7)
        | (opcode & 0x7F)
    )


def encode_halt() -> int:
    return 0xFFFFFFFF


def decode_rv32i(word: int) -> DecodedInstruction:
    u = to_u32(word)
    opcode = u & 0x7F
    if u == 0xFFFFFFFF or opcode == 0x7F:
        return DecodedInstruction(raw=u, mnemonic="HALT", opcode=0x7F, is_halt=True)

    rd = (u >> 7) & 0x1F
    funct3 = (u >> 12) & 0x7
    rs1 = (u >> 15) & 0x1F
    rs2 = (u >> 20) & 0x1F
    funct7 = (u >> 25) & 0x7F

    if opcode == 0x33:  # R-type
        r_map = {
            (0x0, 0x00): "ADD",
            (0x0, 0x20): "SUB",
            (0x4, 0x00): "XOR",
            (0x6, 0x00): "OR",
            (0x7, 0x00): "AND",
            (0x1, 0x00): "SLL",
            (0x5, 0x00): "SRL",
            (0x5, 0x20): "SRA",
            (0x2, 0x00): "SLT",
            (0x3, 0x00): "SLTU",
        }
        mnem = r_map.get((funct3, funct7), "UNKNOWN_R")
        return DecodedInstruction(
            raw=u,
            mnemonic=mnem,
            opcode=opcode,
            rd=rd,
            rs1=rs1,
            rs2=rs2,
            funct3=funct3,
            funct7=funct7,
            reg_write=True,
        )

    if opcode == 0x13:  # I-type ALU
        imm = sign_extend((u >> 20) & 0xFFF, 12)
        i_map = {
            0x0: "ADDI",
            0x4: "XORI",
            0x6: "ORI",
            0x7: "ANDI",
            0x1: "SLLI",
            0x5: "SRAI" if (funct7 == 0x20) else "SRLI",
            0x2: "SLTI",
            0x3: "SLTIU",
        }
        mnem = i_map.get(funct3, "ADDI")
        return DecodedInstruction(
            raw=u,
            mnemonic=mnem,
            opcode=opcode,
            rd=rd,
            rs1=rs1,
            funct3=funct3,
            funct7=funct7,
            imm=imm,
            reg_write=True,
        )

    if opcode == 0x03:  # Load (LB, LH, LW, LBU, LHU)
        imm = sign_extend((u >> 20) & 0xFFF, 12)
        load_map = {0x0: "LB", 0x1: "LH", 0x2: "LW", 0x4: "LBU", 0x5: "LHU"}
        return DecodedInstruction(
            raw=u,
            mnemonic=load_map.get(funct3, "LW"),
            opcode=opcode,
            rd=rd,
            rs1=rs1,
            funct3=funct3,
            imm=imm,
            reg_write=True,
            mem_read=True,
        )

    if opcode == 0x23:  # Store (SB, SH, SW)
        imm12 = (funct7 << 5) | rd
        imm = sign_extend(imm12, 12)
        store_map = {0x0: "SB", 0x1: "SH", 0x2: "SW"}
        return DecodedInstruction(
            raw=u,
            mnemonic=store_map.get(funct3, "SW"),
            opcode=opcode,
            rs1=rs1,
            rs2=rs2,
            funct3=funct3,
            imm=imm,
            mem_write=True,
        )

    if opcode == 0x63:  # B-type
        b12 = (u >> 31) & 0x1
        b11 = (u >> 7) & 0x1
        b10_5 = (u >> 25) & 0x3F
        b4_1 = (u >> 8) & 0xF
        imm13 = (b12 << 12) | (b11 << 11) | (b10_5 << 5) | (b4_1 << 1)
        imm = sign_extend(imm13, 13)
        b_map = {0x0: "BEQ", 0x1: "BNE", 0x4: "BLT", 0x5: "BGE", 0x6: "BLTU", 0x7: "BGEU"}
        return DecodedInstruction(
            raw=u,
            mnemonic=b_map.get(funct3, "BEQ"),
            opcode=opcode,
            rs1=rs1,
            rs2=rs2,
            funct3=funct3,
            imm=imm,
            is_branch=True,
        )

    if opcode in (0x37, 0x17):  # U-type (LUI, AUIPC)
        imm = to_u32(u & 0xFFFFF000)
        return DecodedInstruction(
            raw=u,
            mnemonic="LUI" if opcode == 0x37 else "AUIPC",
            opcode=opcode,
            rd=rd,
            imm=imm,
            reg_write=True,
        )

    if opcode == 0x6F:  # JAL
        b20 = (u >> 31) & 0x1
        b10_1 = (u >> 21) & 0x3FF
        b11 = (u >> 20) & 0x1
        b19_12 = (u >> 12) & 0xFF
        imm21 = (b20 << 20) | (b19_12 << 12) | (b11 << 11) | (b10_1 << 1)
        imm = sign_extend(imm21, 21)
        return DecodedInstruction(
            raw=u,
            mnemonic="JAL",
            opcode=opcode,
            rd=rd,
            imm=imm,
            reg_write=True,
            is_jump=True,
        )

    if opcode == 0x67:  # JALR
        imm = sign_extend((u >> 20) & 0xFFF, 12)
        return DecodedInstruction(
            raw=u,
            mnemonic="JALR",
            opcode=opcode,
            rd=rd,
            rs1=rs1,
            funct3=funct3,
            imm=imm,
            reg_write=True,
            is_jump=True,
        )

    return NOP_INSTRUCTION


def execute_alu(inst: DecodedInstruction, op_a: int, op_b: int) -> int:
    a_u = to_u32(op_a)
    b_u = to_u32(op_b)
    a_s = to_i32(a_u)
    b_s = to_i32(b_u)
    m = inst.mnemonic

    if m == "LUI":
        return b_u
    if m == "AUIPC":
        return to_u32(a_s + b_s)
    if m in ("ADD", "ADDI", "LB", "LH", "LW", "LBU", "LHU", "SB", "SH", "SW"):
        return to_u32(a_s + b_s)
    if m == "SUB":
        return to_u32(a_s - b_s)
    if m in ("XOR", "XORI"):
        return to_u32(a_u ^ b_u)
    if m in ("OR", "ORI"):
        return to_u32(a_u | b_u)
    if m in ("AND", "ANDI"):
        return to_u32(a_u & b_u)
    if m in ("SLL", "SLLI"):
        return to_u32(a_u << (b_u & 0x1F))
    if m in ("SRL", "SRLI"):
        return to_u32(a_u >> (b_u & 0x1F))
    if m in ("SRA", "SRAI"):
        return to_u32(a_s >> (b_u & 0x1F))
    if m in ("SLT", "SLTI"):
        return 1 if a_s < b_s else 0
    if m in ("SLTU", "SLTIU"):
        return 1 if a_u < b_u else 0
    return 0


def evaluate_branch(inst: DecodedInstruction, val1: int, val2: int) -> bool:
    u1, u2 = to_u32(val1), to_u32(val2)
    s1, s2 = to_i32(u1), to_i32(u2)
    m = inst.mnemonic
    if m == "BEQ":
        return u1 == u2
    if m == "BNE":
        return u1 != u2
    if m == "BLT":
        return s1 < s2
    if m == "BGE":
        return s1 >= s2
    if m == "BLTU":
        return u1 < u2
    if m == "BGEU":
        return u1 >= u2
    return False
