# /// script
# dependencies = ["numpy"]
# ///
import numpy as np
exec(open('curl.py').read().split('print("== antisym')[0])
L=np.linspace(-50,50,161)*d2r; D=np.linspace(-70,-42,81)*d2r
LL,DD=np.meshgrid(L,D,indexing='ij'); w=np.cos(DD)
print("proj  <g^2>  2<g^2>  g_rms   omega_rms(own J)  kappa_rms(own,=1-(s1+s2)/2)")
for name,P in [("SFL",P_SFL),("BICEP",P_BIC),("SIN",P_SIN),("ZEA",P_ZEA),("STG",P_STG)]:
    G=np.zeros_like(LL); O=np.zeros_like(LL); K=np.zeros_like(LL)
    for i in range(LL.shape[0]):
        for j in range(LL.shape[1]):
            J=jac(P,LL[i,j],DD[i,j])
            s=np.linalg.svd(J,compute_uv=False)
            G[i,j]=(s[0]-s[1])/(s[0]+s[1]); K[i,j]=1-0.5*(s[0]+s[1])
            O[i,j]=invar(J)[3]
    g2=np.sum(w*G**2)/np.sum(w)
    print(f"{name:6s} {g2:.4e}  {2*g2:.4e}  {np.sqrt(g2):.4f}  {np.sqrt(np.sum(w*O**2)/np.sum(w)):.4f}  {np.sqrt(np.sum(w*K**2)/np.sum(w)):.4f}")
