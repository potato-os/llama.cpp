import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gguf import GGUFReader, GGMLQuantizationType


def text(value):
    raw = value.encode()
    return struct.pack('<Q', len(raw)) + raw


class TestCustomTensorTypes(unittest.TestCase):
    def read_fixture(self, declarations, tensor_type=47):
        # The vision format uses file-local ID47 for a 16-value, 10-byte block.
        metadata = text('general.tensor_types') + struct.pack('<IIQ', 9, 8, len(declarations))
        metadata += b''.join(text(item) for item in declarations)
        tensor = text('weight') + struct.pack('<IQQIQ', 2, 16, 1, tensor_type, 0)
        header = b'GGUF' + struct.pack('<IQQ', 3, 1, 1) + metadata + tensor
        data = header + b'\0' * (-len(header) % 32) + bytes(range(10))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.gguf'
            path.write_bytes(data)
            reader = GGUFReader(path)
            result = reader.tensors[0]
            return result.tensor_type, result.n_bytes, result.data.tobytes()

    def test_named_layout_overrides_numeric_id(self):
        kind, size, data = self.read_fixture(['47=Q4_sym_g16_f16'])
        self.assertEqual(kind, GGMLQuantizationType.Q4_SYM16F)
        self.assertEqual(size, 10)
        self.assertEqual(data, bytes(range(10)))

    def test_unused_unknown_layout_is_allowed(self):
        self.read_fixture(['47=Q4_sym_g16_f16', '99=unknown'])

    def test_used_unknown_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported custom tensor type'):
            self.read_fixture(['47=unknown'])

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate custom tensor type'):
            self.read_fixture(['47=Q4_sym_g16_f16', '47=unknown'])


if __name__ == '__main__':
    unittest.main()
