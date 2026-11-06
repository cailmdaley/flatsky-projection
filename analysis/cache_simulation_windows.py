"""Warm the no-fit window cache used by spectra.load/load_field/load_scalar."""
from multiprocessing import Pool
from pipeline_windows import effective_window


def warm(case):
    spec,field,leg=case
    window=effective_window(spec,field,leg)
    print(spec,field,leg,window.shape,float(window.min()),float(window.max()),flush=True)


if __name__=='__main__':
    cases=[(s,f,0) for s in ('TT','WHITE','EE','BB') for f in (True,False)]
    with Pool(4) as pool:
        pool.map(warm,cases)
