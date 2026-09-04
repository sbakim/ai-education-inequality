"""Run-mode switch shared by all experiment cells."""
FAST_MODE = True   # set False for manuscript-grade full run

def RR(R):
    """Replication count actually used: 1/10 of R (min 3) in FAST_MODE, else R."""
    return max(3, R // 10) if FAST_MODE else R

def banner():
    if FAST_MODE:
        print('*** FAST_MODE: pipeline test only - not for manuscript reporting ***')
