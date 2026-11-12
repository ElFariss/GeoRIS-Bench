"""Explicit dual-polarization rays to a terminated NEC wire channel."""
import numpy as np

from atomwise_ris.hybrid.embedded import C0
from atomwise_ris.hybrid.p4_rt_bridge import FREQ, NEC_C, NEC_ETA
from atomwise_ris.hybrid.ray_coupling import _schur_operators
from atomwise_ris.physics.nec import WireGeometry

LOADS=np.array([1.-10.79906419239132j,1.+129.2009358076087j])
K=2*np.pi*FREQ/NEC_C
FIELD_FACTOR=-1j*K*NEC_ETA/(C0/FREQ)


def wire_context(centers,segments):
    return WireGeometry(centers,np.full(len(centers),.024),radius=.00005,frequency=FREQ).context(segments)


class RisBasis:
    """One resident PyNEC factorization, reused for feeds and frame RHSs."""
    def __init__(self, centers, mesh, archived_z):
        self.n=len(centers)
        self.mesh=mesh
        self.length=.024/mesh
        self.offset=(np.arange(mesh)-mesh//2)*self.length
        self.ctx=wire_context(centers,mesh)
        self.index=0
        self.middle=np.arange(self.n)*mesh+mesh//2
        feed=[]
        for i in range(self.n):
            self.ctx.ex_card(0,i+1,mesh//2+1,0,1.,0.,0.,0.,0.,0.)
            feed.append(self.solve())
        self.iy=np.column_stack(feed)
        self.z=np.asarray(archived_z,dtype=complex)
        identity=np.eye(self.n)
        self.archive_error=float(np.linalg.norm(self.iy[self.middle]@self.z-identity)/np.linalg.norm(identity))
        if self.archive_error>2e-5:
            raise ValueError(f"wire current basis and archived Z_RIS disagree: {self.archive_error:.6g}")
        self.iz=self.iy@self.z

    def solve(self):
        self.ctx.xq_card(0)
        r=self.ctx.get_structure_currents(self.index)
        if r is None or abs(r.get_frequency()-FREQ)>1:
            raise ValueError(f"NEC excitation {self.index} did not produce expected current")
        self.index+=1
        result=np.asarray(r.get_current(),dtype=np.complex128).copy()
        if result.shape!=(self.n*self.mesh,) or not np.isfinite(result).all():
            raise ValueError("invalid PyNEC structure currents")
        return result

    def open_gap(self, e_z):
        v=self.length*e_z
        current=[]
        for column in v.T:
            # No incident wave means exactly zero induced current; do not ask
            # NEC to execute an empty excitation card.
            if not np.any(column):
                current.append(np.zeros(len(column),dtype=complex))
                continue
            for j,x in enumerate(column):
                self.ctx.ex_card(0,j//self.mesh+1,j%self.mesh+1,0,
                                 float(x.real),float(x.imag),0.,0.,0.,0.)
            current.append(self.solve())
        open_current=np.column_stack(current)-self.iz@(self.iy.T@v)
        dz=-v.T@open_current
        return (dz+dz.T)/2


def ray_z_projection(a, theta_t, theta_r, path_len, offsets, isolated_feed_moment):
    """Return centre Ez and reverse-arrival direction z under absolute P4 units.

    Colocated V/H isotropic patterns are e_theta/e_phi in global coordinates.
    The vertical wires only project onto the V component at both ends; Sionna
    has already composed the full V/H Jones transport along the path, including
    any polarization mixing. Outward receive angle is the reverse propagation
    direction. The expression is the z projection of the Cartesian P4 map.
    """
    c_t=np.cos(theta_t)
    feed=np.exp(1j*K*c_t[:,None]*offsets[None,:])@isolated_feed_moment
    e0=FIELD_FACTOR*np.exp(-1j*K*path_len)*a[:,0,0]*feed*np.sin(theta_t)*np.sin(theta_r)
    return e0,np.cos(theta_r)


def project_rays(t,basis, ext, n_bs):
    rx=np.asarray(t["rx_xyz"],float)
    tx=np.asarray(t["tx_xyz"],float)
    e=len(ext)
    if tx.shape!=(e,3) or rx.shape!=(basis.n+e,3) or not np.allclose(tx,ext,rtol=0,atol=1e-6):
        raise ValueError("trace port geometry mismatches selected external geometry")
    if not np.allclose(rx[basis.n:],tx,rtol=0,atol=1e-6):
        raise ValueError("trace receiver port order mismatch")
    m=basis.mesh
    d=basis.length
    # An isolated vertical wire defines the exact transmit/receive basis.
    single=wire_context(np.array([[0.,0.,1.]]),m)
    single.ex_card(0,1,m//2+1,0,1.,0.,0.,0.,0.,0.)
    single.xq_card(0)
    i=np.asarray(single.get_structure_currents(0).get_current(),dtype=complex)
    feed=i/i[m//2]*d
    source=np.asarray(t["link_tx"],int)
    dest=np.asarray(t["link_rx"],int)
    kind=np.asarray(t["path_type"],int)
    jt=np.asarray(t["path_jones"])
    th_t=np.asarray(t["path_theta_t"],float)
    th_r=np.asarray(t["path_theta_r"],float)
    delay=np.asarray(t["path_delay_s"],float)
    length=np.asarray(t["path_length_m"] if "path_length_m" in t else delay*C0,float)
    p=len(source)
    if (jt.shape!=(p,2,2) or any(x.shape!=(p,) for x in (dest,kind,th_t,th_r,length,delay))
        or not all(np.isfinite(x).all() for x in (jt,th_t,th_r,length,delay))
        or (length<=0).any() or not np.isin(kind,(1,2,3,4)).all()
        or (source<0).any() or (source>=e).any() or (dest<0).any() or (dest>=len(rx)).any()):
        raise ValueError("invalid path archive")
    if np.any(dest==basis.n+source):
        raise ValueError("unexpected self path")
    e0,vr=ray_z_projection(jt,th_t,th_r,length,basis.offset,feed)
    e_segments=e0[:,None]*np.exp(1j*K*vr[:,None]*basis.offset[None,:])
    ris_path=dest<basis.n
    u=np.zeros((basis.n,m,e),dtype=complex)
    np.add.at(u,(dest[ris_path],slice(None),source[ris_path]),e_segments[ris_path])
    fields=u.reshape(basis.n*m,e)
    z_re=-(basis.iz*d).T@fields

    # A colocated BS array is outside the far-field P4 approximation.
    # Keep NEC near mutual terms for traced LOS pairs <0.3 m. Add any
    # reflection/diffraction on top; omit environmental self-reflections.
    near=np.linalg.norm(ext[:,None,:]-ext[None,:,:],axis=-1)<.3
    z_near,_=WireGeometry(ext,np.full(e,.024),frequency=FREQ).impedance(m)
    directed=np.diag(np.diag(z_near)).copy()
    for j in np.flatnonzero(~ris_path):
        r=dest[j]-basis.n
        s=source[j]
        if near[r,s] and kind[j]==1:
            directed[r,s]+=z_near[r,s]
        else:
            directed[r,s]-=np.dot(feed,e_segments[j])
    z_ee=(directed+directed.T)/2
    delta=basis.open_gap(fields)
    cin,cout,h,a=_schur_operators(basis.z,z_re,z_ee+delta,n_bs)
    _,_,h_uncorrected,_=_schur_operators(basis.z,z_re,z_ee,n_bs)
    if not all(np.isfinite(x).all() for x in (cin,cout,h,a,delta)):
        raise ValueError("nonfinite channel result")
    return dict(C_in=cin,C_out=cout,H_direct=h,Z_effective=a,H_direct_uncorrected=h_uncorrected,
                delta_Z_open_gap=delta, Z_EE_without_open_gap=z_ee), (source,dest,kind)


def aggregate_classes(source,dest,kind,n,n_bs,k):
    def label(src_ids,dst_ids):
        values=set(kind[np.isin(source,src_ids)&np.isin(dest,dst_ids)].tolist())
        if 1 in values:
            return "los"
        if values=={2}:
            return "reflection_only"
        if values=={3}:
            return "diffraction_only"
        return "mixed" if values else "blocked"
    bs=np.arange(n_bs)
    ue=n_bs+np.arange(k)
    ris=np.arange(n)
    return {"bs_to_user":[label(bs,[n+i]) for i in ue],
            "ris_to_user":[label([i],ris) for i in ue],
            "bs_to_ris":label(bs,ris)}
