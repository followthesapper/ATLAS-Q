"""
NTT-based integer periodicity features.

Implements a simple Cooley–Tukey style NTT for prime modulus p with primitive root g.
We provide:
 - find_prime_with_primitive_root(n): pick a suitable prime and generator
 - ntt(x, p, g): forward NTT
 - top_period_ntt(x): pick peak frequency and map to period
"""

import numpy as np


def _is_prime(p):
    if p < 2:
        return False
    if p % 2 == 0:
        return p == 2
    i = 3
    while i * i <= p:
        if p % i == 0:
            return False
        i += 2
    return True


def _powmod(a, e, m):
    r = 1
    a %= m
    while e:
        if e & 1:
            r = (r * a) % m
        a = (a * a) % m
        e >>= 1
    return r


def _primitive_root(p):
    # Find primitive root modulo p (p must be prime)
    phi = p - 1
    # factor phi (naive)
    fac = []
    x = phi
    d = 2
    while d * d <= x:
        if x % d == 0:
            fac.append(d)
            while x % d == 0:
                x //= d
        d += 1
    if x > 1:
        fac.append(x)
    for g in range(2, p):
        ok = True
        for q in fac:
            if _powmod(g, phi // q, p) == 1:
                ok = False
                break
        if ok:
            return g
    return None


def find_prime_with_primitive_root(n):
    # find a prime p where n | (p-1) and get a primitive root
    p = max(257, n + 1)
    while True:
        # ensure p-1 divisible by n for convenience
        if (p - 1) % n == 0 and _is_prime(p):
            g = _primitive_root(p)
            if g is not None:
                return p, g
        p += 1


def ntt(x, p, g):
    # simple DFT-like NTT: X[k] = sum x[n] * g^{n*k} mod p
    x = np.asarray(x, dtype=int) % p
    N = len(x)
    X = np.zeros(N, dtype=int)
    for k in range(N):
        acc = 0
        pow_ = 1
        # g^{n*k} = (g^k)^n
        wk = _powmod(g, k, p)
        for n in range(N):
            acc = (acc + x[n] * pow_) % p
            pow_ = (pow_ * wk) % p
        X[k] = acc
    return X


def top_period_ntt(x):
    N = len(x)
    p, g = find_prime_with_primitive_root(N)
    X = ntt(x, p, g)
    # ignore DC
    X = X.copy()
    X[0] = 0
    k = int(np.argmax(X))
    # map frequency index k to period ~ N/k (integer)
    period = N // k if k > 0 else N
    return period, (p, g), X
