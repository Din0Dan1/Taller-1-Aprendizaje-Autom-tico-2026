"""Simulación del Problema 5 (a): diseños con bloques correlacionados y R respuestas.

Convenciones:
    - D = 20 variables en 4 bloques de 5; varianza 1, correlación rho dentro del bloque y 0 entre bloques.
    - N = 120 filas de entrenamiento (X) y M = 1000 de auditoría (X_*), generadas independientemente.
    - w* tiene valores en las variables 1, 6, 11, 16 (indexación de 1 a D) y ceros en el resto.
    - y^(r) = X w* + eps^(r), eps^(r) ~ N(0, I_N). El intercepto es 0 y conocido.
    - X y X_* quedan FIJAS en cada escenario; solo cambia eps entre réplicas.

Semillas: una semilla maestra (2026) y subsemillas obtenidas con SeedSequence.spawn.
    Se usan números aleatorios comunes: el mismo Z y los mismos eps en ambos valores de rho,
    de modo que lo único que cambia entre escenarios es la correlación (X = Z L^T, con L L^T = Sigma_rho).
"""
from dataclasses import dataclass
import numpy as np

SEMILLA_MAESTRA = 2026
N_TRAIN, M_AUDIT, D, R = 120, 1000, 20, 40
N_BLOQUES, TAM_BLOQUE = 4, 5
RHOS = (0.0, 0.9)
SOPORTE = np.array([1, 6, 11, 16]) - 1               # S* en indexación base 0
VALORES = np.array([3.0, -2.0, 1.5, -1.0])


def coeficientes_verdaderos(d=D):
    w = np.zeros(d)
    w[SOPORTE] = VALORES

    return w


def matriz_covarianza(rho, n_bloques=N_BLOQUES, tam=TAM_BLOQUE):
    """Sigma_rho: diagonal por bloques, con bloque (1 - rho) I + rho 1 1^T."""
    bloque = (1.0 - rho) * np.eye(tam) + rho * np.ones((tam, tam))
    return np.kron(np.eye(n_bloques), bloque)


def semillas(maestra=SEMILLA_MAESTRA, r=R):
    """Subsemillas registradas: dos para los diseños (X, X_*) y R para los ruidos."""
    raiz = np.random.SeedSequence(maestra)
    ss_X, ss_Xs, ss_ruido = raiz.spawn(3)
    ss_eps = ss_ruido.spawn(r)
    return {"maestra": maestra, "X": ss_X, "Xstar": ss_Xs, "eps": ss_eps}


def registro_semillas(sem):
    """Tabla (diccionario) con la entropía y la clave de cada subsemilla, para el informe."""
    reg = {"maestra": sem["maestra"],
           "X": sem["X"].spawn_key, "Xstar": sem["Xstar"].spawn_key}
    reg["eps"] = [s.spawn_key for s in sem["eps"]]
    return reg


@dataclass
class Escenario:
    rho: float
    Sigma: np.ndarray
    X: np.ndarray            # (N, D)   entrenamiento, fija
    Xstar: np.ndarray        # (M, D)   auditoría, fija
    w_star: np.ndarray       # (D,)
    f_star: np.ndarray       # (M,)     X_* w*
    eps: np.ndarray          # (N, R)   ruidos
    Y: np.ndarray            # (N, R)   columna r = y^(r)


def generar_escenario(rho, sem=None, n=N_TRAIN, m=M_AUDIT, d=D, r=R):
    """Genera X, X_* y las R respuestas de un escenario."""
    sem = semillas(r=r) if sem is None else sem
    Sigma = matriz_covarianza(rho, d // TAM_BLOQUE, TAM_BLOQUE)
    L = np.linalg.cholesky(Sigma)                       # Sigma = L L^T

    # Z ~ N(0, I) con las mismas semillas para todo rho; X = Z L^T tiene covarianza Sigma
    Z = np.random.default_rng(sem["X"]).standard_normal((n, d))
    Zs = np.random.default_rng(sem["Xstar"]).standard_normal((m, d))
    X, Xstar = Z @ L.T, Zs @ L.T

    eps = np.column_stack([np.random.default_rng(s).standard_normal(n) for s in sem["eps"]])
    w = coeficientes_verdaderos(d)
    Y = (X @ w)[:, None] + eps                           # intercepto 0 y conocido
    return Escenario(rho, Sigma, X, Xstar, w, Xstar @ w, eps, Y)


def generar_todos(rhos=RHOS, **kw):
    sem = semillas()
    return {rho: generar_escenario(rho, sem, **kw) for rho in rhos}