import sys
from minecraft.data_stream import *
import socket

def bench(stream_in, stream_out):
    stream_in = DataInputStream(stream_in)
    stream_out = DataOutputStream(stream_out)

    N = stream_in.read_int()
    buf = [0] * N
    for i in range(N):
        buf[i] = stream_in.read_int()

    for i in range(N):
        stream_out.write_int(buf[i])
    stream_out.flush()

if sys.argv[1] == "pipe":
    bench(sys.stdin.buffer, sys.stdout.buffer)
elif sys.argv[1] == "socket":
    sok = socket.create_connection(('localhost', 1337))
    bench(sok.makefile(mode='rb'), sok.makefile(mode='wb'))
