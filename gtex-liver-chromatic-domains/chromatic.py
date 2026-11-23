"""Chromatic six-packs and pair counts of one ball.

For one ball's cells (planar positions in level-0 px, class codes 0..3):

- ``ball_sixpacks``: the chromatic Delaunay–Čech filtration of the cells and,
  for each class t present, the six-pack of the inclusion (class t -> all
  cells) in degrees 0 and 1. Values are squared smallest-enclosing-ball radii
  in px², topocyte's α² convention.
- ``pair_counts``: the ball's Delaunay edges per unordered class pair
  (``common.PAIRS``), and ``pair_expectation``: their exact expectation when
  the classes are shuffled among the same cells.

Credit. The filtration is a port of the chromatic Delaunay–Čech filtration of
chalc (Abhinav Natarajan, https://github.com/abhinavnatarajan/Chalc,
GPL-3.0-or-later), which implements the chromatic alpha complexes of
di Montesano, Draganov, Edelsbrunner and Saghafian (arXiv:2212.03128). The
numba port is copied from chalc-scaling commit 8f370c6
(``report/port/chromatic_port.py``), where it was checked against chalc
15.0.1 on random and degenerate clouds. The match holds up to bars shorter
than about 1e-12 of the scale: on exactly tied clouds, such as a lattice, the
port's floating-point radii and chalc's exact arithmetic pair such near-zero
bars differently (progress.md, Task 2), so ``verify_results.py`` drops them
before it compares bar counts. chalc 15.0.1 stays the oracle of
``test_chromatic.py`` and ``verify_results.py``.

The port's pipeline:

1. the chromatic lift: a cell of the r-th class present goes to (x, y, e_r)
   in R^(2 + s - 1), with e_0 = 0, as chalc does;
2. GUDHI's ``DelaunayComplex`` of the lift (CGAL, exact predicates);
3. the faces of its maximal cells, deduplicated by a numba hash table, so no
   lexicographic simplex label (and no comb(n, d + s) ceiling) is ever formed;
4. each simplex's value: the radius of the smallest enclosing ball of its
   vertices in the plane (Welzl, numba), raised to the largest value of its
   facets;
5. the boundary matrix in filtration order, and phimaker's six-pack.

Changes from the port:

- **Values squared**, after the facet-max pass. Squaring is monotone, so the
  filtration order is the port's.
- **Faces only up to dimension 2 by default.** Degrees 0 and 1 need no more:
  H_0 and H_1 of a complex are those of its 2-skeleton, for the domain and the
  codomain alike, so kernel, image and cokernel follow, and the relative
  modules by the five lemma. A private ``_max_face_dim=3`` route, the port's
  max_degree + 2, is kept for the test that checks it.
- **One boundary matrix per complex,** reused for every class present.
- **Classes keep their codes 0..3:** the class masks use the codes, and the
  domain of class t is ``(mask & ~(1 << t)) == 0``. The lift uses only the
  classes present, in code order.
- **Exact duplicates in the lift** (cells of one class at one position) are
  merged before the triangulation, which would merge them anyway. A merged
  cell would add only a zero-length H_0 bar, which is dropped. Cells of
  different classes at one position are distinct points of the lift and stay.
- **Planar only:** the port's 3D miniball, its Qhull engine and its value
  snapping (off in the port) are left out.

Threads: the numba kernels are serial. phimaker is passed ``num_threads=1``,
yet phimaker 0.5.0 runs up to 10 threads inside each six-pack call
(progress.md, Task 5); the pairing, and so every result, does not depend on
them. numba's default threading layer aborts when ``parallel=True`` code is
entered from several Python threads, so a run spreads balls over processes.
"""

from __future__ import annotations

from itertools import combinations

import gudhi
import numba as nb
import numpy as np
import phimaker

from common import CLASSES, MODULES, PAIRS

# The modules of a class's six-pack kept with ``full`` (``MODULES`` by default); the codomain is kept once per ball.
FULL_MODULES = ("kernel", "domain", "image", "cokernel", "relative")

# ---------------------------------------------------------------- lift


def chromatic_lift(X: np.ndarray, codes: np.ndarray, s: int) -> np.ndarray:
    """(n, d) points with colour codes 0..s-1 -> (n, d + s - 1): colour 0 at the origin of the extra axes, colour j at e_j."""
    n, d = X.shape
    Y = np.zeros((n, d + s - 1))
    Y[:, :d] = X
    nz = codes > 0
    Y[np.flatnonzero(nz), d + codes[nz] - 1] = 1.0
    return Y


# ------------------------------------------------------- triangulation


def delaunay_cells(Y: np.ndarray) -> np.ndarray:
    """Maximal cells of the Delaunay triangulation of Y (GUDHI, exact predicates), rows ascending."""
    st = gudhi.DelaunayComplex(points=Y).create_simplex_tree()
    D = st.dimension()
    top = [sx for sx, _ in st.get_skeleton(D) if len(sx) == D + 1]
    cells = np.array(top, dtype=np.int64).reshape(-1, D + 1)
    cells.sort(axis=1)
    return cells


# ------------------------------------------------------------- faces


@nb.njit(cache=True)
def _row_hash(rows, r):
    h = np.uint64(1469598103934665603)
    for q in range(rows.shape[1]):
        h ^= np.uint64(rows[r, q])
        h *= np.uint64(1099511628211)
    return h


@nb.njit(cache=True)
def _dedupe_rows(rows):
    """Unique rows of ``rows`` (first-seen order) and each row's unique id."""
    R, w = rows.shape
    cap = 16
    while cap < 2 * R:
        cap *= 2
    mask = np.uint64(cap - 1)
    table = np.full(cap, -1, np.int64)
    uniq = np.empty((R, w), np.int64)
    inv = np.empty(R, np.int64)
    M = 0
    for r in range(R):
        slot = _row_hash(rows, r) & mask
        while True:
            u = table[slot]
            if u == -1:
                table[slot] = M
                for q in range(w):
                    uniq[M, q] = rows[r, q]
                inv[r] = M
                M += 1
                break
            same = True
            for q in range(w):
                if uniq[u, q] != rows[r, q]:
                    same = False
                    break
            if same:
                inv[r] = u
                break
            slot = (slot + np.uint64(1)) & mask
    return uniq[:M].copy(), inv


@nb.njit(cache=True)
def _drop_one(simplices):
    """All facets (drop vertex i, i = 0..k) of each k-simplex, rows stay sorted."""
    N, w = simplices.shape
    out = np.empty((N * w, w - 1), np.int64)
    for j in range(N):
        for i in range(w):
            r = j * w + i
            c = 0
            for q in range(w):
                if q != i:
                    out[r, c] = simplices[j, q]
                    c += 1
    return out


@nb.njit(cache=True)
def _subsets(cells, combos):
    """Rows cells[:, combo] for every combo (sorted, so rows stay sorted)."""
    m = cells.shape[0]
    nc, w = combos.shape
    out = np.empty((m * nc, w), np.int64)
    for j in range(m):
        for c in range(nc):
            for q in range(w):
                out[j * nc + c, q] = cells[j, combos[c, q]]
    return out


def faces(cells: np.ndarray, kmax: int):
    """simplices[k] (N_k, k+1) and facets[k] (N_k, k+1) ids into simplices[k-1]."""
    D = cells.shape[1] - 1
    kmax = min(kmax, D)
    if kmax == D:
        top = cells
    else:
        combos = np.array(list(combinations(range(D + 1), kmax + 1)), dtype=np.int64)
        top, _ = _dedupe_rows(_subsets(cells, combos))
    simplices = [None] * (kmax + 1)
    facets = [None] * (kmax + 1)
    simplices[kmax] = top
    for k in range(kmax, 0, -1):
        lower, inv = _dedupe_rows(_drop_one(simplices[k]))
        simplices[k - 1] = lower
        facets[k] = inv.reshape(-1, k + 1)
    return simplices, facets


# ------------------------------------------------- smallest enclosing ball


@nb.njit(cache=True)
def _d2(X, a, b):
    s = 0.0
    for q in range(X.shape[1]):
        t = X[a, q] - X[b, q]
        s += t * t
    return s


@nb.njit(cache=True)
def _ball2(X, a, b, c_out):
    for q in range(X.shape[1]):
        c_out[q] = 0.5 * (X[a, q] + X[b, q])
    return 0.25 * _d2(X, a, b)


@nb.njit(cache=True)
def _ball3(X, a, b, c, c_out):
    """Circumball (centre in the plane of a, b, c) of three points, d = 2 or 3."""
    d = X.shape[1]
    u = np.zeros(3)
    v = np.zeros(3)
    for q in range(d):
        u[q] = X[b, q] - X[a, q]
        v[q] = X[c, q] - X[a, q]
    w0 = u[1] * v[2] - u[2] * v[1]
    w1 = u[2] * v[0] - u[0] * v[2]
    w2 = u[0] * v[1] - u[1] * v[0]
    ww = w0 * w0 + w1 * w1 + w2 * w2
    uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2]
    vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    if ww == 0.0:  # collinear: the farthest pair
        ab, ac, bc = uu, vv, _d2(X, b, c)
        if ab >= ac and ab >= bc:
            return _ball2(X, a, b, c_out)
        if ac >= bc:
            return _ball2(X, a, c, c_out)
        return _ball2(X, b, c, c_out)
    # x = (|u|^2 (v x w) + |v|^2 (w x u)) / (2 |w|^2)
    vw0 = v[1] * w2 - v[2] * w1
    vw1 = v[2] * w0 - v[0] * w2
    vw2 = v[0] * w1 - v[1] * w0
    wu0 = w1 * u[2] - w2 * u[1]
    wu1 = w2 * u[0] - w0 * u[2]
    wu2 = w0 * u[1] - w1 * u[0]
    x0 = (uu * vw0 + vv * wu0) / (2.0 * ww)
    x1 = (uu * vw1 + vv * wu1) / (2.0 * ww)
    x2 = (uu * vw2 + vv * wu2) / (2.0 * ww)
    x = (x0, x1, x2)
    for q in range(d):
        c_out[q] = X[a, q] + x[q]
    return x0 * x0 + x1 * x1 + x2 * x2


_TOL = 1e-12


@nb.njit(cache=True)
def _outside(X, p, c, r2):
    s = 0.0
    for q in range(X.shape[1]):
        t = X[p, q] - c[q]
        s += t * t
    return s > r2 * (1.0 + _TOL) + 1e-300


@nb.njit(cache=True)
def _before(X, a, b):
    """Whether point a precedes point b lexicographically by coordinates."""
    for q in range(X.shape[1]):
        if X[a, q] < X[b, q]:
            return True
        if X[a, q] > X[b, q]:
            return False
    return False


@nb.njit(cache=True)
def _sort_by_coordinates(X, a):
    for i in range(1, a.shape[0]):
        x = a[i]
        j = i - 1
        while j >= 0 and _before(X, x, a[j]):
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = x


@nb.njit(cache=True)
def _canonical_r2(X, sup, ns, c_out):
    """Radius^2 of the ball with support sup[:ns], points sorted by coordinates.

    Sorting by coordinates, not by vertex id, makes coincident points
    interchangeable: the same geometric support always gives the same bits.
    In the plane a support has at most three points.
    """
    s = sup[:ns].copy()
    _sort_by_coordinates(X, s)
    if ns == 1:
        return 0.0
    if ns == 2:
        return _ball2(X, s[0], s[1], c_out)
    return _ball3(X, s[0], s[1], s[2], c_out)


@nb.njit(cache=True)
def _meb_r2_2d(X, v, c, sup):
    """Welzl, d = 2: returns r^2 of the smallest enclosing circle of X[v]."""
    k = v.shape[0]
    for q in range(2):
        c[q] = X[v[0], q]
    r2 = 0.0
    sup[0] = v[0]
    ns = 1
    for i in range(1, k):
        if _outside(X, v[i], c, r2):
            for q in range(2):
                c[q] = X[v[i], q]
            r2 = 0.0
            sup[0] = v[i]
            ns = 1
            for j in range(i):
                if _outside(X, v[j], c, r2):
                    r2 = _ball2(X, v[i], v[j], c)
                    sup[0] = v[i]
                    sup[1] = v[j]
                    ns = 2
                    for l in range(j):
                        if _outside(X, v[l], c, r2):
                            r2 = _ball3(X, v[i], v[j], v[l], c)
                            sup[0] = v[i]
                            sup[1] = v[j]
                            sup[2] = v[l]
                            ns = 3
    return _canonical_r2(X, sup, ns, c)


@nb.njit(cache=True)
def cech_values(X, simplices):
    """Radius of the smallest enclosing ball of each simplex (rows of vertex ids), planar points.

    Serial on purpose: numba's default (workqueue) threading layer aborts the
    process when parallel=True code is entered from several Python threads.
    The values are ~1 % of the run time.
    """
    N, w = simplices.shape
    out = np.empty(N)
    for j in range(N):
        c = np.empty(3)
        sup = np.empty(4, np.int64)
        v = simplices[j]
        if w == 1:
            out[j] = 0.0
        elif w == 2:
            out[j] = np.sqrt(0.25 * _d2(X, v[0], v[1]))
        else:
            out[j] = np.sqrt(_meb_r2_2d(X, v, c, sup))
    return out


# ----------------------------------------------------------- pipeline


def colour_masks(codes: np.ndarray, simplices_k: np.ndarray) -> np.ndarray:
    """Each simplex's set of class codes, as the bit mask OR_v (1 << code_v)."""
    return np.bitwise_or.reduce(np.left_shift(1, codes[simplices_k]), axis=1)


def chromatic_delaunay_cech(xy: np.ndarray, codes: np.ndarray, max_face_dim: int):
    """Facets, values (px²) and class masks of the simplices of dimension <= ``max_face_dim``, and the classes present.

    Entry k of each list is about the k-simplices; facets[k] holds row ids into
    the (k-1)-simplices, and facets[0] is None.
    """
    present = np.unique(codes)
    Y = chromatic_lift(xy, np.searchsorted(present, codes), len(present))
    # Triangulate each distinct point of the lift once (its first cell), then map the cells back to cell ids.
    _, first = np.unique(Y, axis=0, return_index=True)
    first.sort()
    cells = first[delaunay_cells(Y[first])]
    simplices, facets = faces(cells, max_face_dim)
    values = [cech_values(xy, sk) for sk in simplices]
    # In exact arithmetic a face's smallest enclosing ball is never larger than
    # its coface's. Floating-point Welzl can break that by an ulp on degenerate
    # input (cocircular lattice points), which would put a coface before its
    # facet. Taking the max over the facets restores the order and is a no-op
    # whenever the values are exact.
    for k in range(1, len(values)):
        values[k] = np.maximum(values[k], values[k - 1][facets[k]].max(axis=1))
    values = [v * v for v in values]  # radii -> squared radii (px²)
    masks = [colour_masks(codes, sk) for sk in simplices]
    return facets, values, masks, present


def boundary_matrix(facets, values):
    """Filtration order (value, dim) and phimaker's column lists."""
    sizes = [len(v) for v in values]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    allv = np.concatenate(values)
    dims = np.concatenate([np.full(sz, k, np.int64) for k, sz in enumerate(sizes)])
    order = np.lexsort((dims, allv))
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    cols: list = [None] * order.size
    for k, sz in enumerate(sizes):
        rk = rank[offsets[k] : offsets[k + 1]].tolist()
        if k == 0:
            for r in rk:
                cols[r] = []
        else:
            rows = np.sort(rank[offsets[k - 1] + facets[k]], axis=1).tolist()
            for r, row in zip(rk, rows):
                cols[r] = row
    return order, rank, allv, dims, cols


def _diagrams(pairs: dict, sv: np.ndarray, sd: np.ndarray, kernel: bool) -> tuple[np.ndarray, np.ndarray]:
    """phimaker's (birth index -> death index or None) pairs as bars of degrees 0 and 1, death > birth."""
    b = np.fromiter(pairs.keys(), np.int64, len(pairs))
    dd = np.fromiter((-1 if x is None else x for x in pairs.values()), np.int64, len(pairs))
    birth = sv[b]
    death = np.where(dd < 0, np.inf, sv[np.maximum(dd, 0)])
    deg = sd[b] - (1 if kernel else 0)
    keep = death > birth
    return tuple(np.column_stack([birth[keep & (deg == p)], death[keep & (deg == p)]]) for p in (0, 1))


def _checked(xy: np.ndarray, cls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``xy`` as contiguous float64 (n, 2) and ``cls`` as int64 class codes (n,); ``ValueError`` otherwise."""
    xy = np.ascontiguousarray(xy, dtype=np.float64)
    codes = np.asarray(cls).astype(np.int64)
    if xy.ndim != 2 or xy.shape[1] != 2 or codes.shape != (len(xy),):
        raise ValueError(f"need planar positions (n, 2) and n class codes, got {xy.shape} and {codes.shape}")
    if codes.size and (codes.min() < 0 or codes.max() >= len(CLASSES)):
        raise ValueError(f"class codes must lie in 0..{len(CLASSES) - 1}, got {np.unique(codes).tolist()}")
    return xy, codes


def ball_sixpacks(xy: np.ndarray, cls: np.ndarray, *, full: bool = False, _max_face_dim: int = 2) -> dict[tuple[str, int, int], np.ndarray]:
    """The six-pack of the inclusion (class t -> all cells) for each class t of one ball, degrees 0 and 1.

    Parameters
    ----------
    xy : (n, 2) cell positions in level-0 px, n >= 1.
    cls : (n,) class codes 0..3 (``common.CLASSES``).
    full : keep the kernel, cokernel and relative modules too (for checks).

    Returns
    -------
    dict
        ``(module, t, p)`` for module in ``MODULES`` (with ``full``,
        ``FULL_MODULES``), class t in 0..3 and degree p in (0, 1), plus
        ``("codomain", -1, p)``: the alpha PH of all the ball's cells, the same
        for every class, taken from the first class present. The kernel's
        degree is that of the domain cycles it holds. Each value is an (m, 2)
        float64 array of bars (birth, death) in px², death > birth, essential
        bars with death ``inf``, in no particular order. A class absent from
        the ball has no six-pack: its keys hold (0, 2) arrays.
    """
    xy, codes = _checked(xy, cls)
    if len(xy) == 0:
        raise ValueError("a ball holds at least one cell")
    modules = FULL_MODULES if full else MODULES
    out = {(m, t, p): np.empty((0, 2)) for m in modules for t in range(len(CLASSES)) for p in (0, 1)}

    facets, values, masks, present = chromatic_delaunay_cech(xy, codes, _max_face_dim)
    order, rank, allv, dims, cols = boundary_matrix(facets, values)
    allm = np.concatenate(masks)
    degrees = dims[order].tolist()
    sv, sd = allv[order], dims[order]
    for i, t in enumerate(present.tolist()):
        in_dom = (allm & ~(1 << t)) == 0
        cols_in_domain = np.sort(rank[in_dom]).tolist()
        pack = phimaker.sixpack_from_inclusion(cols, degrees, cols_in_domain, get_representatives=False, num_threads=1)
        for m in modules + (("codomain",) if i == 0 else ()):
            key_t = -1 if m == "codomain" else t
            out[m, key_t, 0], out[m, key_t, 1] = _diagrams(getattr(pack, m), sv, sd, kernel=m == "kernel")
    return out


# ------------------------------------------------------- pair counts


def pair_counts(xy: np.ndarray, cls: np.ndarray) -> tuple[np.ndarray, int]:
    """The ball's Delaunay edges per unordered class pair (``common.PAIRS``), and their number.

    GUDHI's ``DelaunayComplex`` (exact predicates) triangulates the distinct
    positions. The edges of the cells are every pair of cells at the two ends
    of a Delaunay edge, and every pair of cells at one position. The edge set
    does not depend on the classes, so ``pair_expectation`` is exact. With
    n < 2 there are no edges.
    """
    xy, codes = _checked(xy, cls)
    counts = np.zeros(len(PAIRS), dtype=np.int64)
    if len(codes) < 2:
        return counts, 0
    positions, at = np.unique(xy, axis=0, return_inverse=True)
    per = np.zeros((len(positions), len(CLASSES)), dtype=np.int64)  # cells per position and class
    np.add.at(per, (at.reshape(-1), codes), 1)
    m = per.sum(axis=1)
    n_edges = int(np.sum(m * (m - 1) // 2))
    across = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)  # [a, b]: an a cell at one end, a b cell at the other
    if len(positions) > 1:
        st = gudhi.DelaunayComplex(points=positions).create_simplex_tree()
        ends = np.array([sx for sx, _ in st.get_skeleton(1) if len(sx) == 2], dtype=np.int64).reshape(-1, 2)
        across = per[ends[:, 0]].T @ per[ends[:, 1]]
        n_edges += int(np.sum(m[ends[:, 0]] * m[ends[:, 1]]))
    at_one = per.T @ per  # [a, b], a != b: pairs of an a and a b cell at one position
    for k, (a, b) in enumerate(PAIRS):
        if a == b:
            counts[k] = across[a, a] + np.sum(per[:, a] * (per[:, a] - 1) // 2)
        else:
            counts[k] = across[a, b] + across[b, a] + at_one[a, b]
    return counts, n_edges


def pair_expectation(class_counts: np.ndarray, n_edges: int) -> np.ndarray:
    """The expected ``pair_counts`` when the classes are shuffled among the ball's cells.

    Each of the ``n_edges`` edges joins two distinct cells, and a shuffle
    makes every pair of cells equally likely to hold a given pair of classes:
    ``n_edges·n_a·n_b / C(n, 2)`` for a != b and ``n_edges·C(n_a, 2) / C(n, 2)``
    for a = b, with n_a the cells of class a. Integer arithmetic up to one
    correctly rounded division. With n < 2 there are no edges.
    """
    n_class = [int(c) for c in np.asarray(class_counts).reshape(-1)]
    n = sum(n_class)
    n_pairs = n * (n - 1) // 2
    out = np.zeros(len(PAIRS))
    if n_pairs == 0:
        return out
    for k, (a, b) in enumerate(PAIRS):
        same = n_class[a] * (n_class[a] - 1) // 2 if a == b else n_class[a] * n_class[b]
        out[k] = int(n_edges) * same / n_pairs
    return out
