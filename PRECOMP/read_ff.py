import numpy as np
import struct

def read_ff():
    with open('FFxy_single_cksum.bin', 'rb') as f:
        bs = f.read()
    idx = 0

    magic, = struct.unpack('i', bs[idx:idx+4])
    assert magic == 816968
    idx += 4

    n_step_x, = struct.unpack('i', bs[idx:idx+4])
    print(f'{n_step_x=}')
    idx += 4
    buf_size = n_step_x*8
    XX = np.frombuffer(bs[idx:idx+buf_size], dtype=np.float64)
    assert len(XX) == n_step_x
    idx += buf_size
    chksum = struct.unpack('II', bs[idx:idx+8])
    idx += 8

    n_step_y, = struct.unpack('i', bs[idx:idx+4])
    print(f'{n_step_y=}')
    idx += 4
    buf_size = n_step_y*8
    YY = np.frombuffer(bs[idx:idx+buf_size], dtype=np.float64)
    assert len(YY) == n_step_y
    idx += buf_size
    chksum = struct.unpack('II', bs[idx:idx+8])
    idx += 8

    print(f'{XX=}')
    print(f'{YY=}')


    nth_max = 128 # TODO
    nffa, = struct.unpack('i', bs[idx:idx+4])
    print(f'{nffa=}')
    idx += 4
    Ffm = np.zeros((nffa, n_step_x, n_step_y, nth_max), dtype=np.float64)
    Ffp = np.zeros((nffa, n_step_x, n_step_y, nth_max), dtype=np.float64)
    for a in range(nffa):
        print(f'{a+1} / {nffa}')
        nx, = struct.unpack('i', bs[idx:idx+4])
        idx += 4
        assert nx == n_step_x
        for i in range(n_step_x):
            ny, = struct.unpack('i', bs[idx:idx+4])
            idx += 4
            assert ny == n_step_y
            for j in range(n_step_y):
                nth, = struct.unpack('i', bs[idx:idx+4])
                idx += 4
                assert nth <= nth_max
                buf_size = nth*4
                Ffm[a,i,j,:nth] = np.frombuffer(bs[idx:idx+buf_size], dtype=np.float32)
                idx += buf_size
                Ffp[a,i,j,:nth] = np.frombuffer(bs[idx:idx+buf_size], dtype=np.float32)
                idx += buf_size

TX_LEN = 23
TY_LEN = 14
NYTAY = 810
NXTAY = 100
def read_tx():
    with open('taylorx_cksum.bin', 'rb') as f:
        bs = f.read()
    idx = 0

    magic, = struct.unpack('i', bs[idx:idx+4])
    assert magic == 816968
    idx += 4

    n_tay_y, = struct.unpack('i', bs[idx:idx+4])
    assert n_tay_y == TX_LEN
    idx += 4

    TX = np.zeros((TX_LEN, NYTAY), dtype=np.float64)
    for i in range(TX_LEN):
        ny_tay, = struct.unpack('i', bs[idx:idx+4])
        assert ny_tay == NYTAY
        idx += 4

        buf_size = ny_tay*8
        TX[i] = np.frombuffer(bs[idx:idx+buf_size], dtype=np.float64)
        idx += buf_size
    print(TX[0,:10])
    print(TX[1,:10])
    print(TX[2,:10])
    print(TX[3,:10])
    print(TX[4,:10])

def main():
    # read_ff()
    read_tx()

if __name__ == '__main__': main()
