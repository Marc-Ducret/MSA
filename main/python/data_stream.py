import struct

"""
Reading from Java DataInputStream format.
"""

class DataInputStream:
    def __init__(self, stream):
        self.stream = stream

    def read_boolean(self):
        return struct.unpack('?', self.stream.read(1))[0]

    def read_byte(self):
        return struct.unpack('b', self.stream.read(1))[0]

    def read_unsigned_byte(self):
        return struct.unpack('B', self.stream.read(1))[0]
    
    def read_char(self):
        return chr(struct.unpack('>H', self.stream.read(2))[0])

    def read_double(self):
        return struct.unpack('>d', self.stream.read(8))[0]

    def read_float(self):
        return struct.unpack('>f', self.stream.read(4))[0]

    def read_short(self):
        return struct.unpack('>h', self.stream.read(2))[0]

    def read_unsigned_short(self):
        return struct.unpack('>H', self.stream.read(2))[0]

    def read_long(self):
        return struct.unpack('>q', self.stream.read(8))[0]

    def read_utf(self):
        utf_length = struct.unpack('>H', self.stream.read(2))[0]
        return self.stream.read(utf_length).decode('utf-8')

    def read_int(self):
        return struct.unpack('>i', self.stream.read(4))[0]
        
"""
Writing to Java DataInputStream format.
"""

class DataOutputStream:
    def __init__(self, stream):
        self.stream = stream

    def write_boolean(self, bool):
        self.stream.write(struct.pack('?', bool))

    def write_byte(self, val):
        self.stream.write(struct.pack('b', val))

    def write_unsigned_byte(self, val):
        self.stream.write(struct.pack('B', val))
    
    def write_char(self, val):
        self.stream.write(struct.pack('>H', ord(val)))

    def write_double(self, val):
        self.stream.write(struct.pack('>d', val))

    def write_float(self, val):
        self.stream.write(struct.pack('>f', val))

    def write_short(self, val):
        self.stream.write(struct.pack('>h', val))

    def write_unsigned_short(self, val):
        self.stream.write(struct.pack('>H', val))

    def write_long(self, val):
        self.stream.write(struct.pack('>q', val))

    def write_utf(self, string):
        self.stream.write(struct.pack('>H', len(string)))
        self.stream.write(string)

    def write_int(self, val):
        self.stream.write(struct.pack('>i', val))
        
    def flush(self):
        self.stream.flush()