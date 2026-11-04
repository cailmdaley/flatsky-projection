import numpy as np, healpy as hp
ns=1024; npix=hp.nside2npix(ns); a2=4*np.pi/npix
ip=np.arange(0,npix,211)
b=hp.boundaries(ns,ip,step=1,nest=True)          # (n,3,4) corners N,E,S,W
th,ph=hp.pix2ang(ns,ip,nest=True); z=np.cos(th)
n=np.stack([-z*np.cos(ph),-z*np.sin(ph),np.sqrt(1-z**2)],1); e=np.stack([-np.sin(ph),np.cos(ph),0*ph],1)
d0=b[:,:,0]-b[:,:,2]; d1=b[:,:,1]-b[:,:,3]       # image of the two chart diagonals on the tangent plane
p=lambda d:np.stack([(d*e).sum(1),(d*n).sum(1)],1)
A=np.stack([p(d0),p(d1)],2)                      # 2x2: chart diagonals -> sky
s=np.linalg.svd(A,compute_uv=False)
area=s[:,0]*s[:,1]/2                              # rhombus area = |d0 x d1|/2
g=(s[:,0]-s[:,1])/(s[:,0]+s[:,1])
print("area/expected: mean %.5f std %.1e"%((area/a2).mean(),(area/a2).std()))
print("g: mean %.4f max %.4f; <g^2> all sky %.5f"%(g.mean(),g.max(),(g**2).mean()))
print("  eq zone <g^2> %.5f, polar caps <g^2> %.5f"%((g[np.abs(z)<2/3]**2).mean(),(g[np.abs(z)>=2/3]**2).mean()))
F=((1+g**2)/(1-g**2)).mean()
print("predicted <(1+g^2)/(1-g^2)> = %.5f"%F)
for nn in (1024,2048,4096):
    w=hp.pixwin(nn,lmax=2*nn); l=np.arange(len(w)); x=l/nn
    for hi in (0.2,0.5):
        m=(x>0.05)&(x<hi)
        c=np.polyfit((l[m]/nn)**2,-np.log(w[m]),2)    # -ln w = c1 x^2 + c2 x^4...
        c1=np.polyfit((l[m]/nn)**2,-np.log(w[m])/(l[m]/nn)**2,1)[1]
        print(f"Nside {nn} x<{hi}: measured c={c1:.5f}  square pixel pi/72={np.pi/72:.5f}  sheared {np.pi/72*F:.5f}")
