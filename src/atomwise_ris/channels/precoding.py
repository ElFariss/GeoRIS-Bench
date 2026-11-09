import numpy as np
C0=299792458.0


def precoder(direct, power_w=0.1, regularization=1e-3):
    """Reference-channel RZF, computed once per snapshot, shared by every method.

    Relative regularization is fixed, not retuned against RIS configurations.
    Total radiated power is sum(abs(W)**2), including all users.
    """
    k=direct.shape[0]
    gram=direct @ direct.conj().T
    reg=max(float(np.trace(gram).real/k)*regularization,1e-30)
    w=direct.conj().T @ np.linalg.inv(gram + reg*np.eye(k))
    if np.linalg.norm(w) < 1e-30:
        w=np.eye(direct.shape[1], k, dtype=complex)
    return w*np.sqrt(power_w)/np.linalg.norm(w)
