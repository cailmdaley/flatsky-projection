# /// script
# dependencies = ["numpy"]
# ///
import numpy as np
d2r=np.pi/180; r2am=180*60/np.pi
lam0,dec0=0.0,-56*d2r

def sky(x): # (lam,dec)
    return x
# projections: return plane coords (x,y) given (lam,dec)
def P_SFL(l,d): return ((l-lam0)*np.cos(d), d-dec0)
def P_BIC(l,d): return ((l-lam0)*np.cos(dec0), d-dec0)
def zen(l,d):
    # angular radius r and azimuth from tangent point
    cosr=np.sin(dec0)*np.sin(d)+np.cos(dec0)*np.cos(d)*np.cos(l-lam0)
    r=np.arccos(np.clip(cosr,-1,1))
    # standard: xi = cos(d)sin(l-lam0); eta = sin(d)cos(dec0)-cos(d)sin(dec0)cos(l-lam0)
    xi=np.cos(d)*np.sin(l-lam0); eta=np.sin(d)*np.cos(dec0)-np.cos(d)*np.sin(dec0)*np.cos(l-lam0)
    n=np.hypot(xi,eta); n=np.where(n==0,1e-300,n)
    return r,xi/n,eta/n
def mk(R):
    def P(l,d):
        r,ux,uy=zen(l,d); Rr=R(r)
        return (Rr*ux, Rr*uy)
    return P
P_TAN=mk(np.tan); P_SIN=mk(np.sin); P_ZEA=mk(lambda r:2*np.sin(r/2)); P_STG=mk(lambda r:2*np.tan(r/2))
P_ARC=mk(lambda r:r)

def jac(P,l,d,h=1e-6):
    # d(plane)/d(E,N):  dE = cos d * dlam, dN = d dec
    xp,yp=P(l+h/np.cos(d),d); xm,ym=P(l-h/np.cos(d),d)
    c1=np.array([(xp-xm)/(2*h),(yp-ym)/(2*h)])
    xp,yp=P(l,d+h); xm,ym=P(l,d-h)
    c2=np.array([(xp-xm)/(2*h),(yp-ym)/(2*h)])
    return np.array([[c1[0],c2[0]],[c1[1],c2[1]]])

def invar(M):
    psi=M-np.eye(2)
    kap=-0.5*(psi[0,0]+psi[1,1]); g1=-0.5*(psi[0,0]-psi[1,1]); g2=-0.5*(psi[0,1]+psi[1,0]); om=0.5*(psi[0,1]-psi[1,0])
    return kap,g1,g2,om

print("== antisym part of J_proj (sky orthonormal -> plane), at offsets ==")
for name,P in [("SFL",P_SFL),("BICEP",P_BIC),("SIN",P_SIN),("ZEA",P_ZEA),("STG",P_STG),("TAN",P_TAN),("ARC",P_ARC)]:
    for (dl,dd) in [(10,0),(0,10),(15,15)]:
        l=lam0+dl*d2r; d=dec0+dd*d2r
        k,g1,g2,om=invar(jac(P,l,d))
        if (dl,dd)==(15,15): print(f"  {name:5s} off=({dl},{dd}): kappa={k:+.4f} g1={g1:+.4f} g2={g2:+.4f} omega={om:+.5f}")

print()
print("== omega of composite C = TAN o P^-1  (exact, numeric) ==")
for name,P in [("SFL",P_SFL),("BICEP",P_BIC),("SIN",P_SIN),("ZEA",P_ZEA),("STG",P_STG)]:
    out=[]
    for (dl,dd) in [(10,0),(0,10),(20,0),(0,20),(15,15),(25,25)]:
        l=lam0+dl*d2r; d=dec0+dd*d2r
        JT=jac(P_TAN,l,d); JP=jac(P,l,d)
        JC=JT@np.linalg.inv(JP)
        k,g1,g2,om=invar(JC)
        out.append(f"({dl},{dd}):om={om:+.2e},g={np.hypot(g1,g2):.2e}")
    print(f"  {name:5s} "+"  ".join(out))

print()
print("== SFL analytic check: omega ~ t/2, t=(lam-lam0)sin(dec) ==")
for (dl,dd) in [(10,0),(20,0),(10,10)]:
    l=lam0+dl*d2r; d=dec0+dd*d2r
    t=(l-lam0)*np.sin(d)
    JT=jac(P_TAN,l,d); JP=jac(P_SFL,l,d); JC=JT@np.linalg.inv(JP)
    print(f"  off=({dl},{dd}) t/2={t/2:+.5f}  omega_C={invar(JC)[3]:+.5f}  omega_JP={invar(JP)[3]:+.5f}")

print()
print("== radial deflection |d| for zenithal, relative to ARC (r-preserving) and TAN ==")
for rdeg in [10,15,20,25]:
    r=rdeg*d2r
    row=[]
    for nm,R in [("SIN",np.sin(r)),("ZEA",2*np.sin(r/2)),("STG",2*np.tan(r/2)),("TAN",np.tan(r))]:
        row.append(f"{nm}: vs ARC {abs(R-r)*r2am:6.1f}'  vs TAN {abs(np.tan(r)-R)*r2am:7.1f}'")
    print(f"  r={rdeg}deg  "+" | ".join(row))
print()
print("  chart-free: SIN vs ZEA displacement at r=25deg:", f"{abs(2*np.sin(25*d2r/2)-np.sin(25*d2r))*r2am:.1f} arcmin")
print("  chart-free: STG vs ZEA at r=25deg:", f"{abs(2*np.tan(25*d2r/2)-2*np.sin(25*d2r/2))*r2am:.1f} arcmin")

print()
print("== SPT 1500d field: RA +/-50deg, dec -70..-42; rms of composite (TAN o P^-1) invariants + deflection ==")
L=np.linspace(-50,50,161)*d2r; D=np.linspace(-70,-42,81)*d2r
LL,DD=np.meshgrid(L,D,indexing='ij')
w=np.cos(DD)
for name,P in [("SFL",P_SFL),("BICEP",P_BIC),("SIN",P_SIN),("ZEA",P_ZEA),("STG",P_STG)]:
    K=np.zeros_like(LL);G=np.zeros_like(LL);O=np.zeros_like(LL);Dd=np.zeros_like(LL)
    for i in range(LL.shape[0]):
        for j in range(LL.shape[1]):
            l,d=LL[i,j],DD[i,j]
            JT=jac(P_TAN,l,d); JP=jac(P,l,d); JC=JT@np.linalg.inv(JP)
            k,g1,g2,om=invar(JC)
            K[i,j]=k;G[i,j]=np.hypot(g1,g2);O[i,j]=om
            xp,yp=P(l,d); xt,yt=P_TAN(l,d)
            Dd[i,j]=np.hypot(xt-xp,yt-yp)
    def rms(a): return np.sqrt(np.sum(w*a**2)/np.sum(w))
    print(f"  {name:5s} kappa_rms={rms(K):.4f}  |gamma|_rms={rms(G):.4f}  omega_rms={rms(O):.4f}  "
          f"|d|_rms={rms(Dd)*180/np.pi:.2f}deg  |d|_max={Dd.max()*180/np.pi:.2f}deg")
