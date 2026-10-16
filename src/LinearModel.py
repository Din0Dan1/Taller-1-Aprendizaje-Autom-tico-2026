import numpy as np
import math
# ---------- utilidades ----------
def sigmoid(z):
    """Función para calcular la sigmoide para cada predicción z_i"""
    z = np.asarray(z, dtype=float)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1/(1+e), e/(1+e))

def soft_threshold(v, t):
    """ Función que hace que Lasso lleve los coef a cero.
        Si |v| <= t el resultado es cero
        si v > t el resultado es v-t (Se acerca t unidades a cero)
        si v < -t el resultado es v + t (Se acerca t unidades a cero)
    """

    return  np.sign(v) * np.maximum(np.abs(v) - t, 0.0)

def validate_inputs(X, y, task, lam1, lam2, eta, tol, max_iter):
    """ Función que valida las dimensiones de X,y.
        También valida valores de hiperparámetros.  """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y,dtype = float)
    if X.ndim != 2:
        raise ValueError(f"X debe ser de 2D (N,D), pero tiene {X.ndim} dimensiones")
    if y.ndim != 1:
        raise ValueError(f"y debe ser de 1D (N,), pero tiene {y.ndim} dimensiones")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"La dim de filas de  X y la longitud de y deben ser ambos N\n X tiene {X.shape[0]} filas \n y tiene {y.shape[0]} entradas ")

    if not np.isfinite(X).all():
        raise ValueError("X tiene valores nulos o infinitos")

    if not np.isfinite(y).all():
        raise ValueError("y tiene valores nulos o infinitos")



    if lam1 < 0:
        raise ValueError("lam1 es menor a cero")
    if lam2 < 0:
        raise ValueError("lam2 es menor a cero")
    if eta is not None and eta <= 0:
        raise ValueError("eta debe ser mayor a cero")
    if tol <= 0:
        raise ValueError("tol debe ser mayor a cero")
    if not isinstance(max_iter,int) or max_iter < 1:
        raise ValueError("max iter debe ser mayor a cero y entero")
    if task not in ("logistic", "regression"):
        raise ValueError("task debe ser 'logistic' o 'regression'")
    if task == "logistic" and not np.isin(y, [0, 1]).all():
        raise ValueError("En clasificación, y debe tener solo valores 0 o 1")

    return X,y



# ---------- pérdidas y gradientes (suave) ----------
def smooth_loss(w, b, X, y, task):
    """Parte suave J del objetivo en regresión y en logística.
    No incluye las penalizaciones L1 ni L2."""
    N = X.shape[0]
    z = X @ w + b

    if task == "regression":
        res = z - y
        return res @ res / (2 * N)

    if task == "logistic":
        return np.mean(np.logaddexp(0, z) - y * z)


def smooth_grad(w, b, X, y, task):
    """Gradiente de la parte suave J respecto a (w, b).
    Devuelve (grad_w, grad_b)."""
    N = X.shape[0]
    z = X @ w + b

    if task == "regression":
        res = z - y
    elif task == "logistic":
        res = sigmoid(z) - y
    else:
        raise ValueError("task debe ser 'regression' o 'logistic'")

    grad_w = X.T @ res / N      # forma (D,)
    grad_b = np.mean(res)       # escalar
    return grad_w, grad_b


def full_objective(w, b, X, y, task, lam1, lam2):
    """F(w, b) = J(w, b) + lam1 * ||w||_1 + (lam2 / 2) * ||w||_2^2.
    El intercepto b no entra en las penalizaciones."""
    J = smooth_loss(w, b, X, y, task)
    l1 = lam1 * np.sum(np.abs(w))
    l2 = (lam2 / 2) * (w @ w)
    return J + l1 + l2

# ---------- modelo ----------
class LinearModel:
    def __init__(self, method="ols", task="regression", lam1=0.0, lam2=0.0,
             eta=None, tol=1e-6, max_iter=10_000, warm_start=False):
        self.method = method
        self.task = task
        self.lam1 = lam1
        self.lam2 = lam2
        self.eta = eta
        self.tol = tol
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.w_ = None
        self.b_ = None
        self.history_ = []
        self.n_iter_ = 0
        self.stop_reason_ = None

    def fit(self, X, y):
        X, y = validate_inputs(X, y, self.task, self.lam1, self.lam2,self.eta, self.tol, self.max_iter)
        # TODO: si NO warm_start o w_ es None: reiniciar w_, b_, history_, n_iter_
        #       si warm_start: partir de w_, b_ actuales
        if self.method == "ols":
            self._fit_ols(X, y)
        elif self.method == "ridge":
            self._fit_ridge(X, y)
        elif self.method == "gd":
            self._fit_gd(X, y)
        elif self.method == "prox":
            self._fit_prox(X, y)
        return self

    def _fit_ols(self, X, y):
        """OLS por álgebra lineal numérica (mínimos cuadrados con intercepto)."""
        N = X.shape[0]
        Xt = np.column_stack([np.ones(N), X])        # X̃ = [1_N  X], forma (N, D+1)
        theta, *_ = np.linalg.lstsq(Xt, y, rcond=None)
        self.b_ = theta[0]                           # intercepto
        self.w_ = theta[1:]                          # pesos, forma (D,)
        self.n_iter_ = 0
        self.stop_reason_ = "closed_form"
        self.history_ = [full_objective(self.w_, self.b_, X, y, self.task, 0.0, 0.0)]

    def _fit_ridge(self, X, y):
        N, D = X.shape
        Xt = np.column_stack([np.ones(N), X])
        P = np.diag(np.r_[0.0, np.ones(D)])
        M = Xt.T @ Xt + N * self.lam2 * P
        theta = np.linalg.solve(M, Xt.T @ y)
        self.b_ = theta[0]
        self.w_ = theta[1:]
        self.n_iter_ = 0
        self.stop_reason_ = "closed_form"
        self.history_ = [full_objective(self.w_, self.b_, X, y, self.task, 0.0, self.lam2)]

    def _fit_gd(self, X, y):
        """Descenso por gradiente para regresión o logística, con opción L2.
        El intercepto b se actualiza solo con el gradiente de J (sin penalización)."""
        N, D = X.shape

        # Paso eta: 1/L, o el que dé el usuario
        if self.eta is None:
            Xt = np.column_stack([np.ones(N), X])
            lmax = np.linalg.eigvalsh(Xt.T @ Xt)[-1]
            factor = 1.0 if self.task == "regression" else 0.25   # sigmoide: sigma' <= 1/4
            L = factor * lmax / N + self.lam2
            eta = 1.0 / L
        else:
            eta = self.eta
        self.eta_ = eta

        # Arranque: en frío
        if not self.warm_start or self.w_ is None:
            w = np.zeros(D)
            b = 0.0
            self.history_ = []
            self.n_iter_ = 0
        else:
            w = self.w_.copy()
            b = self.b_

        self.stop_reason_ = "max_iter"
        for k in range(self.max_iter):
            gw, gb = smooth_grad(w, b, X, y, self.task)
            gw = gw + self.lam2 * w          # opción L2 (b no se penaliza)

            # Criterio de parada: norma del gradiente completo (w y b)
            if np.linalg.norm(np.r_[gb, gw]) < self.tol:
                self.stop_reason_ = "tol"
                break

            w = w - eta * gw
            b = b - eta * gb
            self.history_.append(full_objective(w, b, X, y, self.task, 0.0, self.lam2))
            self.n_iter_ += 1

        self.w_ = w
        self.b_ = b

    def _fit_prox(self, X, y):
        """Gradiente proximal (ISTA) para F = J + lam1*||w||_1 + (lam2/2)*||w||_2^2.
        El paso suave es de gradiente sobre J + L2; el umbral suave solo se aplica a w.
        El intercepto b se actualiza solo con el gradiente de J (sin umbral)."""
        N, D = X.shape

        # Paso eta = 1/L (L de la parte suave), o el que dio el usuario
        if self.eta is None:
            Xt = np.column_stack([np.ones(N), X])
            lmax = np.linalg.eigvalsh(Xt.T @ Xt)[-1]
            factor = 1.0 if self.task == "regression" else 0.25
            L = factor * lmax / N + self.lam2
            eta = 1.0 / L
        else:
            eta = self.eta
        self.eta_ = eta

        # Arranque: en frío, salvo warm_start explícito
        if not self.warm_start or self.w_ is None:
            w = np.zeros(D)
            b = 0.0
            self.history_ = []
            self.n_iter_ = 0
        else:
            w = self.w_.copy()
            b = self.b_

        self.stop_reason_ = "max_iter"
        for k in range(self.max_iter):
            gw, gb = smooth_grad(w, b, X, y, self.task)
            gw = gw + self.lam2 * w                       # parte suave: J + L2 (b sin penalizar)

            w_new = soft_threshold(w - eta * gw, eta * self.lam1)   # prox solo sobre w
            b_new = b - eta * gb                                    # b: sin umbral

            # Residuo proximal sobre theta = (b, w) completo
            residuo = np.linalg.norm(np.r_[b_new - b, w_new - w]) / eta

            w, b = w_new, b_new
            self.history_.append(full_objective(w, b, X, y, self.task, self.lam1, self.lam2))
            self.n_iter_ += 1

            if residuo < self.tol:
                self.stop_reason_ = "tol"
                break

        self.w_ = w
        self.b_ = b


    def predict(self, X):
        """Regresión: z = Xw + b. Logística: probabilidad p = sigmoid(z)."""
        if self.w_ is None:
            raise RuntimeError("El modelo no está ajustado: llama a fit primero")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.w_.shape[0]:
            raise ValueError(f"X debe ser (N, {self.w_.shape[0]})")
        if not np.isfinite(X).all():
            raise ValueError("X tiene valores nulos o infinitos")
        z = X @ self.w_ + self.b_
        return z if self.task == "regression" else sigmoid(z)