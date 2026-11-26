"""在 CPU 上拒绝最终 HSACO 中未被后端降解的 LLVM intrinsic。"""
import struct


def unresolved_llvm_intrinsics(image):
    if image[:6] != b'\x7fELF\x02\x01':
        raise RuntimeError('HSACO 必须是 ELF64 little endian')
    section_offset = struct.unpack_from('<Q', image, 40)[0]
    section_size, section_count = struct.unpack_from('<HH', image, 58)
    sections = [struct.unpack_from('<IIQQQQIIQQ', image, section_offset + i * section_size)
                for i in range(section_count)]
    unresolved = set()
    for section in sections:
        if section[1] not in (2, 11):  # SHT_SYMTAB / SHT_DYNSYM
            continue
        strings_section = sections[section[6]]
        strings = image[strings_section[4]:strings_section[4] + strings_section[5]]
        assert section[9] == 24, 'ELF64 符号条目长度不符'
        for offset in range(section[4], section[4] + section[5], section[9]):
            name, _, _, symbol_section, _, _ = struct.unpack_from('<IBBHQQ', image, offset)
            if symbol_section != 0 or not name:
                continue
            label = strings[name:strings.index(0, name)].decode()
            if label.startswith('llvm.'):
                unresolved.add(label)
    return sorted(unresolved)


def verify_lowered_hsaco(image):
    unresolved = unresolved_llvm_intrinsics(image)
    if unresolved:
        raise RuntimeError('HSACO 存在未降解的 LLVM intrinsic，禁止装载: ' + ', '.join(unresolved))
