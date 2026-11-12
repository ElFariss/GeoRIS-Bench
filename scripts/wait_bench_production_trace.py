"""One bounded Linux file-event wait for a durable pilot trace receipt."""
import argparse
import ctypes
import os
from pathlib import Path
import select
import time


def wait(trace, seconds):
    libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.inotify_init1(os.O_CLOEXEC)
    if fd<0:
        raise OSError(ctypes.get_errno(),"inotify_init1")
    try:
        end=time.monotonic()+seconds
        watching=None
        while True:
            # The trace worker may still be staging. Track its deepest
            # existing ancestor, moving the watch down as folders appear.
            ancestor=next(d for d in (trace,*trace.parents) if d.is_dir())
            if ancestor!=watching:
                if libc.inotify_add_watch(fd,os.fsencode(ancestor),0x8|0x80|0x100|0x400)<0:
                    raise OSError(ctypes.get_errno(),f"inotify_add_watch: {ancestor}")
                watching=ancestor
                if ancestor!=trace and (trace.is_dir() or any(
                        p.is_dir() for p in trace.parents if ancestor in p.parents)):
                    continue
            if (trace/"FAILED").is_file():
                raise ValueError("trace FAILED; no channel computation started")
            if (trace/"DONE").is_file() and (trace/"DONE").stat().st_size:
                return
            left=end-time.monotonic()
            if left<=0 or not select.select([fd],[],[],max(0.,left))[0]:
                raise TimeoutError("trace marker not delivered within the declared wait")
            os.read(fd,65536)
    finally:
        os.close(fd)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace",type=Path)
    p.add_argument("--seconds",type=int,default=3600)
    a=p.parse_args()
    wait(a.trace,a.seconds)
