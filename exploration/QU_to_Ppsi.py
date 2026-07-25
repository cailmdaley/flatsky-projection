P = np.sqrt(np.asarray(q)[flatsky_mask]**2 + np.asarray(u)[flatsky_mask]**2)
psi = np.arctan2(np.asarray(u)[flatsky_mask], np.asarray(q)[flatsky_mask]) / 2
np.asarray(q)[flatsky_mask] = P
np.asarray(u)[flatsky_mask] = psi

Pflat, psiflat = [healpix_to_flatsky(f, map_stub=CreateFieldMapStub(res=res*G3Units.arcmin), interp=interp)
                         for f in (q, u)]
plotFrame({r"$P$" : np.asarray(Pflat)[0:100,0:100], r"$\psi$" : np.asarray(psiflat)[0:200,0:200]})
flatsky_qu_frame = {'Q': Pflat.Clone(False), 'U': psiflat.Clone(False)}
np.asarray(flatsky_qu_frame['Q'])[:] = Pflat * np.cos(2*psiflat)
np.asarray(flatsky_qu_frame['U'])[:] = Pflat * np.sin(2*psiflat)
plotFrame({k : np.asarray(f)[0:100,0:100] for (k,f) in flatsky_qu_frame.items()})
flatten_pol(flatsky_qu_frame)
plotFrame({k : np.asarray(f)[0:100,0:100] for (k,f) in flatsky_qu_frame.items()})
eb_frame = constructEB(flatsky_qu_frame)
plotFrame(eb_frame, mult=mask)