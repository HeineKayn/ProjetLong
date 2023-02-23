import sys
from capstone import *
from capstone.x86 import *
import pefile
import pandas as pd
import os.path
from os import listdir
import pprint
import json
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def get_instruction_group(inst):
    inst_groups = {
        # Conditional Data Transfer
        'cdt': ['cmove', 'cmovz', 'cmovne', 'cmovnz', 'cmova', 'cmovnbe', 'cmovae', 'cmovnb', 'cmovb',
                'cmovnae', 'cmovbe', 'cmovna', 'cmovg',
                'cmovnle', 'cmovge', 'cmovnl', 'cmovl', 'cmovnge', 'cmovle', 'cmovng',
                'cmovc', 'cmovnc', 'cmovo', 'cmovno', 'cmovs', 'cmovns', 'cmovp', 'cmovpe',
                'cmovnp', 'cmovpo', ],
        # Unconditianl Data Transfer
        'udt': ['mov', 'xchg', 'bswap', 'movsx', 'movzx', 'movlps', 'movqda', 'lock xchg'],
        # Stack Data Transfer
        'sdt': ['push', 'pop', 'pusha', 'pushad', 'popa', 'popad', 'popal', 'pushal'],
        'adt': ['xadd'],

        # Compared Data Transfer
        'cmpdt': ['cmpxchg', 'cmpxchg8b', ],
        # Converting
        'cvt': ['cwd', 'cdq', 'cbw', 'cwde'],
        # Binary Arithmetic Instructions
        'bai': ['adcx', 'adox', 'add', 'adc', 'sub', 'sbb', 'imul', 'imulb', 'imulw', 'imull',
                'mul', 'mulb', 'mulw', 'mull', 'idiv', 'idivb', 'idivw', 'idivl',
                'div', 'inc', 'dec', 'neg', 'cmp', 'addb', 'addw', 'addl', 'adcb',
                'adcw', 'adcl', 'subb', 'subw', 'subl', 'sbbb', 'sbbw', 'sbbl',
                'cmpb', 'cmpw', 'cmpl', 'incb', 'incw', 'incl', 'decb', 'decw',
                'decl', 'negb', 'negw', 'negl', 'lock add', 'lock adc', 'lock sbb',
                'lock sub', 'lock neg', 'lock inc', 'lock dec'],
        # Integer Arithmetic Instructions
        'iai': ['fiadd', 'fiaddr', 'ficom', 'fidiv', 'fisub', 'fimul', 'ficomp', 'fisubr', 'fidivr', 'fimulr'],
        # Decimal Arithmetic Instructions
        'dai': ['daa', 'das', 'aaa', 'aas', 'aam', 'aad', ],
        # Flaot Arithmetic Instructions
        'fai': ['fabs', 'fadd', 'faddp', 'fchs', 'fdiv', 'fdivp', 'fdivr', 'fdivrp', 'fiadd',
                'fidiv', 'fidivr', 'fimul',
                'fisub', 'fisubr', 'fmul', 'fmulp', 'fprem', 'fprem1', 'frndint', 'fscale', 'fsqrt',
                'fsub', 'fsubp',
                'fsubr', 'fsubrp', 'fxtract'],
        # Float Comparison Instructions
        'fci': ['fcom', 'fcomi', 'fcomip', 'fcomp', 'fcompp', 'ftst', 'fucom',
                'fucomi', 'fucomip', 'fucomp', 'fucompp', 'fxam'],
        # Stack Arithmetic Instructions
        'sai': ['fsqrt', 'fscale', 'fprem', 'frndint', 'fxtract', 'fabs', 'fchs', ],
        # Logical Instructions
        'li': ['and', 'andb', 'andw', 'andl', 'or', 'orb', 'orw', 'orl', 'xor',
               'xorb', 'xorw', 'xorl', 'not', 'notb', 'notw', 'notl', 'lock or',
               'lock and', 'lock xor', 'lock not', ],
        # Shift Rotate Instructions
        'sri': ['sar', 'shr', 'sal', 'shl', 'shrd', 'shld', 'ror', 'rol', 'rcr', 'rcl',
                'sarb', 'sarw', 'sarl', 'salb', 'salw', 'sall', 'shrb', 'shrw', 'shrl',
                'shld', 'shldw', 'shldl', 'shrd', 'shrdw', 'shrdl', ],
        # Bit Instructions
        'bii': ['bt', 'bts', 'btr', 'btc', 'bsf', 'bsr', 'lock bt', 'lock bts',
                'lock btr', 'lockbtc'],
        # Byte Instructions
        'byi': ['sete', 'setz', 'setne', 'setnz', 'seta', 'setnbe', 'setae', 'setnb', 'setnc', 'setb', 'setnae',
                'setc', 'setbe', 'setna', 'setg', 'setnle', 'setge', 'setnl', 'setl', 'setnge', 'setle', 'setng',
                'sets', 'setns', 'seto', 'setno', 'setpe', 'setp', 'setpo', 'setnp', 'test', 'testb',
                'testw', 'testl', 'crc32', 'popcnt', ],
        # Conditional Jumping
        'cj': ['je', 'jz', 'jnz', 'jnz', 'ja', 'jnbe', 'jae', 'jnb', 'jb', 'jnae', 'jbe', 'jna', 'jg',
               'jnle', 'jge', 'jnl', 'jl', 'jnge', 'jle', 'jng', 'jc', 'jnc', 'jo', 'jno', 'js', 'jns',
               'jpo', 'jnp', 'jpe', 'jp', 'jcxz', 'jecxz', 'loopz', 'loope', 'loopnz', 'loopne', 'into',
               'jne'],
        # Unconditional Jumping/Looping
        'uj': ['jmp', 'loop', 'call', 'enter', 'leave', 'lcall', 'acall', 'ljmp', ],
        # Interruptions
        'int': ['ret', 'iret', 'retn', 'int', 'retf', 'hlt', 'iretd', ],
        # Strings Instructions
        'si': ['movs', 'movsb', 'movsw', 'movsd', 'cmps', 'cmpsb', 'cmpsw', 'cmpsd', 'scas',
               'scasb', 'scasw', 'scasd', 'lods', 'lodsb', 'lodsw', 'lodsd', 'rep', 'repe',
               'repz', 'repne', 'repnz', 'stos', 'stosd', 'stosb', 'stosw', 'stosl', ],
        # I/O Instructions
        'io': ['in', 'out', 'ins', 'insb', 'insw', 'insd', 'outs', 'outsb', 'outsw', 'outsd',
               'inb', 'inw', 'insl', 'outw', 'outsl', 'outl', ],
        # Flags
        'flg': ['stc', 'clc', 'cmc', 'cld', 'std', 'lahf', 'sahf', 'pushf', 'pushfd',
                'popf', 'popfd', 'sti', 'cli', 'popfw', 'popfl', 'pushfw', 'pushfl', 'salc'],
        # Segment Register Instructions
        'seg': ['lds', 'les', 'lfs', 'lgs', 'lss', ],
        #
        'misc': ['lea', 'nop', 'ud', 'xlat', 'xlatb', 'cpuid', 'prefetchw', 'prefetchwt',
                 'clflush', 'clflushopt', ],

        'sr': ['xsave', 'xsavec', 'xsaveopt', 'xrstor', 'xgetbv', ],

        'rng': ['rdrand', 'rdseed'],

        'arr': ['bound', 'boundb', 'boundw', 'boundl'],

        'pmi': ['sldt', 'str', 'lldt', 'ltr', 'verr', 'verw', 'sgdt', 'sidt',
                'smsw', 'lmsw', 'lar', 'lsl', 'clts', 'arpl', 'lgdt', 'lidt', ],

        'pci': ['frstor', 'finitfninit', 'finit', 'fnop', 'fsave', 'fnsave', 'fstcw',
                'fnstcw', 'fstenv', 'fnstenv', 'fstsw', 'fnstsw', 'fwait', 'wait',
                'fclex', 'fnclex', 'fdecstp', 'ffree', 'fincstp', 'pause', 'fclex',
                'fdecstp', 'ffree', 'fincstp', 'finit', 'fldcw', 'fldenv',
                'fnclex', 'fninit', 'fnop', 'fnsave', 'fnstcw', 'fnstenv',
                'fnstsw', 'frstor', 'fsave', 'fstcw', 'fstenv', 'fstsw', 'fwait',
                'rdtsc', 'fxrstor', 'fxsave', 'invd', 'winvd', ],
        # MMX Data Transfer
        'mmxt': ['movd', 'movq'],
        # MMX Conversion
        'mmxc': ['packssdw', 'packsswb', 'packuswb', 'punpckhbw', 'punpckhdq',
                 'punpckhwd', 'punpcklbw', 'punpckldq', 'punpcklwd'],
        # MMX Arithmetic Instuctions
        'mmxa': ['paddb', 'paddd', 'paddsb', 'paddsw', 'paddusb', 'paddusw', 'paddw', 'pmaddwd', 'pmulhw',
                 'pmullw', 'psubb', 'psubd', 'psubsb', 'psubsw', 'psubusb', 'psubusw', 'psubw'],
        # MMX Comparision
        'mmxcmp': ['pcmpeqd', 'pcmpeqb', 'pcmpeqw', 'pcmpgtb', 'pcmpgtd', 'pcmpgtw'],
        # MMX Logical
        'mmxl': ['pand', 'pandn', 'por', 'pxor'],
        # MMX Shift Rotate Instuctions
        'mmxsr': ['pslld', 'psllq', 'psllw', 'psrad', 'psraw', 'psrld', 'psrlq', 'psrlw'],
        # MMX State Management
        'mmxsm': ['emms'],
        # SSE Data Transfer
        'sset': ['movaps', 'movhlps', 'movhps', 'movlhps', 'movlps', 'movmskps', 'movss', 'movups'],
        # SSE Arithmetic Instructions
        'ssea': ['addps', 'addss', 'divps', 'divss', 'maxps', 'maxss', 'minps', 'minss', 'mulps',
                 'mulss', 'rcpps', 'rcpss', 'rsqrtps', 'rsqrtss', 'sqrtps', 'sqrtss', 'subps', 'subss'],
        # SSE Comparision
        'ssecmp': ['cmpps', 'cmpss', 'comiss', 'ucomiss', ],
        # SSE Logical
        'ssel': ['andnps', 'andps', 'orps', 'xorps'],
        # SSE Shuffle Unpack
        'ssesu': ['shufps', 'unpckhps', 'unpcklps'],
        # SSE Convertion
        'ssecvt': ['cvtpi2ps', 'cvtps2pi', 'cvtsi2ss', 'cvtss2si', 'cvttps2pi', 'cvttss2si'],
        # SSE

        # Floating Data Transfer
        'fdt': ['fbld', 'fbstp', 'fcmovb', 'fcmovbe', 'fcmove', 'fcmovnb', 'fcmovnbe', 'fcmovne',
                'fcmovnu', 'fcmovu', 'fild', 'fist', 'fistp', 'fld', 'fst', 'fstp', 'fxch', 'fisttp', ],
        # Flaot Transcedental
        'ftrdt': ['f2xm1', 'fcos', 'fpatan', 'fptan', 'fsin', 'fsincos', 'fyl2x', 'fyl2xp1'],
        # Float Load constant
        'flc': ['fld1', 'fldl2e', 'fldl2t', 'fldlg2', 'fldln2', 'fldpi', 'fldz'],

        'tse': ['xabort', 'xbegin', 'xbeginl', 'xbeginw', 'xend', 'xtest'],

        'ssebi': ['pavgb', 'pavgw', 'pextrw', 'pinsrw', 'pmaxsw', 'pmaxub', 'pminsw',
                  'pminub', 'pmovmskb',
                  'pmulhuw', 'psadbw', 'pshufw', ],
        'vmx': ['invept', 'invvpid', 'vmcall', 'vmclear', 'vmfunc', 'vmlaunch', 'vmresume', 'vmptrld',
                'vmptrst', 'vmread', 'vmwrite', 'vmxoff', 'vmxon', ]
    }
    inst = inst.split(' ')
    if len(inst) > 1:
        inst = inst[1]
    else:
        inst = inst[0]
    if 'int' in inst:
        return 'int'
    for gr in inst_groups.keys():
        if inst in inst_groups[gr]:
            return gr
    for gr in inst_groups.keys():
        for mmc in inst_groups[gr]:
            if inst in mmc or mmc in inst:
                return gr
    return 'other'

def get_main_code_section(sections, base_of_code):
    addresses = []
    for section in sections: 
        addresses.append(section.VirtualAddress)
    if base_of_code in addresses:
        #if sections[addresses.index(base_of_code)].Characteristics == int(0x60000020):
        if 1==1:    
            return sections[addresses.index(base_of_code)]
        else:
            return None
    else:
        addresses.append(base_of_code)
        addresses.sort()
        if addresses.index(base_of_code)!= 0:
            #if sections[addresses.index(base_of_code)-1].Characteristics == int(0x60000020):
            if 1==1: 
                return sections[addresses.index(base_of_code)-1]
            else:
                return None
        else:
            return None

def fine_disassemble(exe, depth=128000):
    main_code = get_main_code_section(exe.sections, exe.OPTIONAL_HEADER.BaseOfCode)
    md = Cs(CS_ARCH_X86, CS_MODE_32)
    md.detail = True
    last_address = 0
    last_size = 0
    begin = main_code.PointerToRawData
    end = begin + main_code.SizeOfRawData
    ins_count = 0
    size_count = 0
    sequence_of_groups = ['begin', ]
    while True:
        data = exe.get_memory_mapped_image()[begin:end]
        for i in md.disasm(data, begin):
            group = get_instruction_group(i.mnemonic)
            if sequence_of_groups[-1] == group:
                sequence_of_groups[-1] = (group, 2)
            elif sequence_of_groups[-1][0] == group:
                sequence_of_groups[-1] = (group, sequence_of_groups[-1][1] + 1)
            else:
                sequence_of_groups.append(group)
            last_address = int(i.address)
            last_size = i.size
            ins_count += 1
            size_count += last_size
        begin = max(int(last_address), begin) + last_size + 1
        if begin >= end:
            break
        if ins_count > depth:
            break
    return sequence_of_groups


def quick_disassemble(path, depth=128000):
    exe = pefile.PE(path)
    gr = fine_disassemble(exe, depth)
    return gr

def encode_sequence(sequence):
    labels = ["cdt", "udt", "sdt", "adt", "cmpdt", "cvt", "bai", "iai",
              "dai", "fai", "fci", "sai", "li", "sri", "bii", "byi",
              "cj", "uj", "int", "si", "io", "flg", "seg", "misc", "sr",
              "rng", "arr", "pmi", "pci", "mmxt", "mmxc", "mmxa",
              "mmxcmp", "mmxl", "mmxsr", "mmxsm", "sset", "ssea",
              "ssecmp", "ssel", "ssesu", "ssecvt", "fdt", "ftrdt", "flc",
              "tse", "ssebi", "vmx", "other"]

    labels_array = np.array(labels).reshape(-1, 1)
    hot_encoder = OneHotEncoder(sparse_output=False)
    encoded_labels = hot_encoder.fit_transform(labels_array) # 49x49

    encode_dict = {}
    for l, e in zip(labels, encoded_labels):
        encode_dict[l] = e   # les labels sont encodés et dic fait corresp entre nom et encodage

    del sequence[0]
    
    count = 0
    for s in sequence:
        if isinstance(s, str):
            count += 1
        else:
            count += s[1]
            
    steps = 128
    vect = 49
    data_array = np.zeros((int(count / steps) + 1, steps, vect), dtype='float32')
    length = steps
    i, j, k = (0, 0, 0)
    for s in sequence:
        if isinstance(s, str):
            data_array[i, j] = encode_dict[s] + 0.
            j += 1
            if j > length - 1:
                j = 0
                i += 1
        else:
            for _ in range(s[1]):
                data_array[i, j] = encode_dict[s[0]] + 0.
                j += 1
                if j > length - 1:
                    j = 0
                    i += 1
    return data_array

def extract_sequence(path):
    sequence = quick_disassemble(path) # sequence d'opcode avec redondance enlevé 'bai', ('udt', 3), 'bai'
    # Open a file and dump it in 
 
 
    # if sequence :
    #     encode_sequence(sequence)