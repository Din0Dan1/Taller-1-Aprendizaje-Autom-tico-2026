import numpy as np
from src.modelos import (OLS, Ridge, RegresionLinealGD, LassoProximal, LogisticaGD,
                         sigmoide, perdida_logistica, _aumentar)

def make_data(task, N=80, D=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D))
    w0 = np.array([1.5, -2.0, 0.0, 0.7])[:D]
    if task == "regresion":
        y = 1.0 + X @ w0 + 0.1 * rng.standard_normal(N)
    else:
        y = (X @ w0 + rng.standard_normal(N) > 0).astype(float)
    return X, y

# ---------- 1. gradientes por diferencias centrales (b y L2 incluidos) ----------
def grad_check(modelo, X, y, n_points=3, h=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    Xa = _aumentar(X)
    deltas = []
    for _ in range(n_points):
        theta = 0.5 * rng.standard_normal(Xa.shape[1])     # theta = (b, w)
        g_an = modelo._gradiente_suave(Xa, y, theta)
        g_num = np.zeros_like(theta)
        for i in range(theta.size):
            e = np.zeros_like(theta); e[i] = h
            g_num[i] = (modelo._objetivo_theta(Xa, y, theta + e)
                        - modelo._objetivo_theta(Xa, y, theta - e)) / (2 * h)
        deltas.append(np.linalg.norm(g_an - g_num)
                      / max(1.0, np.linalg.norm(g_an), np.linalg.norm(g_num)))
    return deltas

def test_gradientes():
    for nombre, clase, task in [("regresion", RegresionLinealGD, "regresion"),
                                ("logistica", LogisticaGD, "logistica")]:
        X, y = make_data(task)
        for lam2 in [0.0, 0.1]:
            d = max(grad_check(clase(lambda2=lam2), X, y))
            print(f"{nombre:10s} lambda2={lam2:<4} delta_g = {d:.2e}")
            assert d < 1e-5          # L1 no se diferencia: se prueba con KKT

# ---------- 2. estabilidad numérica ----------
def test_estabilidad():
    zs = np.array([-1000, -100, -1, 0, 1, 100, 1000], dtype=float)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        p = sigmoide(zs)
        assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        for y in [0.0, 1.0]:
            for z in zs:
                l = perdida_logistica(np.array([z]), np.array([y]))
                assert np.isfinite(l) and l >= 0
        assert abs(perdida_logistica(np.array([1000.0]), np.array([0.0])) - 1000.0) < 1e-9
        assert abs(perdida_logistica(np.array([-1000.0]), np.array([1.0])) - 1000.0) < 1e-9
        assert perdida_logistica(np.array([1000.0]), np.array([1.0])) < 1e-12

# ---------- 3. residuos de optimalidad ----------
def test_ols_columna_duplicada():
    X, y = make_data("regresion")
    X = np.column_stack([X, X[:, 0]])
    m = OLS().fit(X, y)
    print("OLS duplicada: rango", m.rango_, "de", X.shape[1] + 1, "| residuo", m.residuo_normal(X, y))
    assert not m.solucion_unica_
    assert m.residuo_normal(X, y) < 1e-10

def test_ridge_ecuaciones_normales():
    X, y = make_data("regresion")
    X = np.column_stack([X, X[:, 0]])                  # dependiente: Ridge sigue siendo único
    m = Ridge(lambda2=0.1).fit(X, y)
    print("Ridge: residuo normal =", m.residuo_normal(X, y))
    assert m.residuo_normal(X, y) < 1e-10
    assert abs(np.mean(y - m.predict(X))) < 1e-10      # 1^T r = 0: b no penalizado

def test_lasso_optimalidad():
    X, y = make_data("regresion")
    X[:, 1] = 0.8 * X[:, 0] + 0.2 * X[:, 1]            # con correlación
    N, D = X.shape
    Z = np.random.default_rng(1).standard_normal((N, D)); Z -= Z.mean(0)
    Q, _ = np.linalg.qr(Z)
    Xo = np.sqrt(N) * Q                                # X^T X / N = I, columnas centradas
    for nombre, XX in [("correlacionada", X), ("ortogonal", Xo)]:
        m = LassoProximal(lambda1=0.05, lambda2=0.0, tol=1e-12, max_iter=200_000).fit(XX, y)
        print(nombre, m.n_iter_, m.motivo_, "ceros:", int(np.sum(m.w_ == 0)),
              "| optimalidad:", m.optimalidad(XX, y))
        assert m.motivo_ == "convergencia" and m.optimalidad(XX, y) < 1e-6
    # caso ortogonal: forma cerrada w_j = S_lambda1(x_j^T y / N)
    m = LassoProximal(lambda1=0.05, tol=1e-12, max_iter=200_000).fit(Xo, y)
    w_cerrada = np.sign(Xo.T @ y / N) * np.maximum(np.abs(Xo.T @ y / N) - 0.05, 0)
    assert np.allclose(m.w_, w_cerrada, atol=1e-6)

# ---------- 4. predicciones, guardado y tau ----------
def test_predicciones_y_tau():
    ruta = "tmp_modelo.npz"
    X, y = make_data("regresion")
    m = Ridge(lambda2=0.1).fit(X, y)
    m.guardar(ruta)
    assert np.array_equal(m.predict(X), Ridge.cargar(ruta).predict(X))

    X, y = make_data("logistica")
    m = LogisticaGD(lambda2=0.01, tol=1e-8, max_iter=50_000).fit(X, y)
    P = m.predict_proba(X)
    assert P.shape == (X.shape[0], 2) and (P >= 0).all() and (P <= 1).all()
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)
    m.guardar(ruta)
    m2 = LogisticaGD.cargar(ruta)
    assert np.array_equal(P, m2.predict_proba(X))

    obj0 = m.objetivo(X, y)
    l05, l09 = m.predict(X, tau=0.5), m.predict(X, tau=0.9)
    assert np.array_equal(P, m.predict_proba(X)) and obj0 == m.objetivo(X, y)
    assert l09.sum() <= l05.sum()

    # segunda llamada a fit sin warm_start: mismo resultado
    m3 = LogisticaGD(lambda2=0.01, tol=1e-8, max_iter=50_000).fit(X, y)
    m3.fit(X, y)
    assert np.allclose(m3.w_, m.w_, atol=1e-12) and m3.n_iter_ == m.n_iter_

def run_all():
    test_gradientes(); test_estabilidad(); test_ols_columna_duplicada()
    test_ridge_ecuaciones_normales(); test_lasso_optimalidad(); test_predicciones_y_tau()
    print("Todas las pruebas del bloque (b) pasaron")

run_all()