"""Check correction broadcasting and preservation of raw production products."""
import numpy as np
import spectra as S

if __name__=='__main__':
    for field in (False,True):
        for spec in ('TT','white','EE','BB'):
            if spec in ('TT','white'):
                raw=S.load_scalar(spec,field=field,correct_windows=False)
                corrected=S.load_scalar(spec,field=field)
            else:
                loader=S.load_field if field else S.load
                raw=loader(spec,correct_windows=False)
                corrected=loader(spec)
            window=S.simulation_window(spec,field=field)
            broadcast=window[:,None,:] if field else window[:,:,None,:]
            np.testing.assert_allclose(corrected*broadcast,raw,rtol=2e-15,atol=0)
            assert np.isfinite(window).all() and (window>0).all()
            assert (window[...,S.ELL>=5400]==1).all(), 'Do not correct square-corner bins'
            print(field,spec,'roundtrip passed',raw.shape,flush=True)
    w=S.simulation_window('EE',field=True)
    k=np.argmin(abs(S.ELL-4775))
    assert np.ptp(w[:,k])>.02, 'A single projected-frame window is not sufficient'
    assert not np.array_equal(S.simulation_window('EE'),S.simulation_window('BB'))
    import field_1500d as FD
    cached=FD.load()
    np.testing.assert_allclose(cached['ee_mean'],S.load_field('EE').mean(1),rtol=1e-14)
    np.testing.assert_allclose(cached['bb_mean'],S.load_field('BB').mean(1),rtol=1e-14)
    print('field derived cache uses the corrected measurements',flush=True)
