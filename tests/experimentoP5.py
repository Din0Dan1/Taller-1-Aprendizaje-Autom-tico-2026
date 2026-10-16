"""Problema 5 (b): ajustes OLS/Ridge/Lasso, descomposición sesgo-varianza y comparación teórica de Ridge.

Convención de los modelos (la de modelos.py): F = 1/(2N)||y - Xw||^2 + penalización.
Con ella, Ridge(lambda2 = lam) resuelve (X^T X + N*lam*I) w = X^T y, es decir, el lambda "clásico" es N*lam.
En todo el problema b = 0 es conocido: se usa fit_intercept=False.
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

raiz = next((d for d in [Path.cwd(), *Path.cwd().parents] if (d / "src" / "simulacionP5.py").exists()), None)
if raiz is None:
    raise FileNotFoundError("No se encontró src/simulacionP5.py en esta carpeta ni en sus carpetas superiores.")
if str(raiz) not in sys.path:
    sys.path.insert(0, str(raiz))

from src.simulacionP5 import generar_todos, semillas, registro_semillas
from tests.experimentoP5 import (correr_experimento, ajustar_replicas,
                                ridge_teorico, comparar_ridge_teoria, LAMBDAS)

LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
SIGMA2 = 1.0


def crear_modelo(nombre, lam, tol=1e-10, max_iter=200_000):
    """Cada regularizador activa solo su penalización."""
    if nombre == "OLS":
        return OLS(fit_intercept=False)
    if nombre == "Ridge":
        return Ridge(lambda2=lam, fit_intercept=False)
    if nombre == "Lasso":
        return LassoProximal(lambda1=lam, lambda2=0.0, fit_intercept=False,
                             tol=tol, max_iter=max_iter)
    raise ValueError(nombre)


def ajustar_replicas(esc, nombre, lam=None, **kw):
    """Ajusta el modelo a cada y^(r). Devuelve W (R, D) y un resumen de convergencia."""
    R = esc.Y.shape[1]
    W = np.zeros((R, esc.X.shape[1]))
    n_iter, no_conv = [], 0
    for r in range(R):
        m = crear_modelo(nombre, lam, **kw).fit(esc.X, esc.Y[:, r])
        W[r] = m.w_
        if hasattr(m, "motivo_"):
            n_iter.append(m.n_iter_)
            no_conv += int(m.motivo_ != "convergencia")
    info = {"iter_max": max(n_iter) if n_iter else 0, "no_convergio": no_conv}
    return W, info


def descomposicion(W, esc):
    """B^2, V y E_f con las fórmulas del enunciado (1/R en la varianza)."""
    F = W @ esc.Xstar.T                                 # f_hat_rm, forma (R, M)
    fbar = F.mean(axis=0)                               # \bar f_m
    B2 = np.mean((fbar - esc.f_star) ** 2)
    V = np.mean(np.mean((F - fbar) ** 2, axis=0))       # 1/M sum_m 1/R sum_r
    E = np.mean((F - esc.f_star) ** 2)                  # 1/(MR) sum_{m,r}
    return B2, V, E


def error_respuesta_nueva(W, esc, seed=0):
    """E[(y_* - f_hat)^2] con y_* = f_* + eps_*, eps_* ~ N(0, sigma^2) independiente: debe ser E_f + sigma^2."""
    rng = np.random.default_rng(seed)
    F = W @ esc.Xstar.T
    y_nueva = esc.f_star[None, :] + np.sqrt(SIGMA2) * rng.standard_normal(F.shape)
    return np.mean((y_nueva - F) ** 2)


def correr_experimento(escenarios, lambdas=LAMBDAS, **kw):
    """Tabla por escenario, modelo y lambda; además guarda los coeficientes (para el punto (c))."""
    filas, coefs = [], {}
    for rho, esc in escenarios.items():
        combos = [("OLS", None)] + [(m, l) for m in ("Ridge", "Lasso") for l in lambdas]
        for nombre, lam in combos:
            W, info = ajustar_replicas(esc, nombre, lam, **kw)
            B2, V, E = descomposicion(W, esc)
            assert abs(E - (B2 + V)) < 1e-10 * max(1.0, E), "falla Ê_f = B̂² + V̂"
            coefs[(rho, nombre, lam)] = W
            filas.append({"rho": rho, "modelo": nombre, "lambda": lam,
                          "B2": B2, "V": V, "E_f": E, "B2+V": B2 + V,
                          "dif_identidad": abs(E - (B2 + V)),
                          "E_nueva(E_f+s2)": E + SIGMA2,
                          "E_nueva_MC": error_respuesta_nueva(W, esc),
                          "no_nulos_medio": float(np.mean(np.sum(W != 0, axis=1))),
                          **info})

    return pd.DataFrame(filas), coefs


# ---------- Ridge: media y covarianza empíricas frente a las fórmulas teóricas ----------
def ridge_teorico(esc, lam):
    """E[w_hat] y Cov(w_hat) teóricos. Con la convención de modelos.py, el lambda clásico es N*lam."""
    X, N, D = esc.X, esc.X.shape[0], esc.X.shape[1]
    A = np.linalg.inv(X.T @ X + N * lam * np.eye(D))
    media = A @ X.T @ X @ esc.w_star
    cov = SIGMA2 * A @ X.T @ X @ A
    return media, cov


def comparar_ridge_teoria(esc, lam, W):
    """Compara media y covarianza empíricas de W (R, D) con las teóricas, en coeficientes y en predicciones."""
    R = W.shape[0]
    media_t, cov_t = ridge_teorico(esc, lam)
    media_e = W.mean(axis=0)
    cov_e = np.cov(W.T, bias=True)                      # 1/R, igual que V̂
    Xs = esc.Xstar
    B2_t = np.mean((Xs @ media_t - esc.f_star) ** 2)
    V_t = np.mean(np.einsum("md,de,me->m", Xs, cov_t, Xs))
    return {"rho": esc.rho, "lambda": lam,
            "err_rel_media": np.linalg.norm(media_e - media_t) / np.linalg.norm(media_t),
            "err_rel_cov": np.linalg.norm(cov_e - cov_t) / np.linalg.norm(cov_t),
            "B2_empirico": descomposicion(W, esc)[0], "B2_teorico": B2_t,
            "V_empirico": descomposicion(W, esc)[1], "V_teorico": V_t,
            "V_teorico*(R-1)/R": V_t * (R - 1) / R}