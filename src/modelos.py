"""Modelos lineales implementados desde cero (Taller 1, Problema 4).

Convención de pérdidas (enunciado del taller):
    z = X w + b 1
    J_reg = 1/(2N) ||z - y||^2
    J_log = -(1/N) sum[y log p + (1-y) log(1-p)],  p = sigma(z)
    F     = J + lambda1 ||w||_1 + (lambda2/2) ||w||_2^2      (b no se penaliza)
"""
import numpy as np


# ---------------------------------------------------------------------------
# Utilidades numéricas
# ---------------------------------------------------------------------------
def sigmoide(z):
    """Sigmoide estable: el exponente evaluado nunca es positivo."""
    z = np.asarray(z, dtype=float)
    e = np.exp(-np.abs(z))
    # Para estabilizar z < 0, multiplica numerador y denominador de 1/(1+e^-z) por e^z de modo que el exponente nunca sea postivo, i.e. no desborda.
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def log1pexp(z):
    """log(1 + e^z) estable (softplus)."""
    z = np.asarray(z, dtype=float)
    # log(1 + e^z) = max(z,0) + log(1 + e^(-|z|)); usa np.log1p = np.log(1 + x) para el segundo término.
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def umbral_suave(v, t):
    """Operador proximal de t*|.|: sign(v) * max(|v| - t, 0)."""
    # Es la regla del Problema 2a), con t en lugar de lambda1.
    return np.sign(v) * np.maximum(0.0, np.abs(v) - t)


def perdida_logistica(z, y):
    """J_log evaluada desde z, sin calcular log(p)."""
    #  l_i = log(1 + e^z_i) - y_i z_i, con log(1 + e^z) calculado de forma estable.
    z = np.asarray(z, dtype=float)
    return float(np.mean(log1pexp(z) - y * z))


def _aumentar(X, fit_intercept=True):
    """X~ = [1_N  X] si hay intercepto; X sin cambios si b = 0 es conocido."""
    if not fit_intercept:
        return X
    return np.column_stack([np.ones(X.shape[0]), X])


def _validar_X(X, D=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X debe ser una matriz 2D (N x D).")
    if X.shape[0] == 0:
        raise ValueError("X no tiene filas.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contiene valores no finitos.")
    if D is not None and X.shape[1] != D:
        raise ValueError(f"X tiene {X.shape[1]} columnas; se esperaban {D}.")
    return X


def _validar_Xy(X, y, binaria=False):
    X = _validar_X(X)
    y = np.asarray(y, dtype=float).ravel()
    if y.shape[0] != X.shape[0]:
        raise ValueError("X e y tienen distinto número de filas.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contiene valores no finitos.")
    if binaria and not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("y debe contener solo 0 y 1.")
    return X, y


def _no_negativo(nombre, valor):
    if not np.isfinite(valor) or valor < 0:
        raise ValueError(f"{nombre} debe ser finito y >= 0.")


# ---------------------------------------------------------------------------
# Base común
# ---------------------------------------------------------------------------
class _ModeloLineal:
    """Guarda w, b y ofrece predicción, guardado y carga."""

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.w_ = None
        self.b_ = None

    @property
    def _i0(self):
        """Posición del primer peso en theta: 1 si theta = (b, w), 0 si theta = w."""
        return 1 if self.fit_intercept else 0

    def _separar(self, theta):
        """theta -> (b, w). Sin intercepto, b = 0 exacto."""
        if self.fit_intercept:
            return float(theta[0]), theta[1:]
        return 0.0, theta.copy()

    def _comprobar_ajustado(self):
        if self.w_ is None:
            raise RuntimeError("El modelo no ha sido ajustado.")

    def decision(self, X):
        self._comprobar_ajustado()
        X = _validar_X(X, D=self.w_.shape[0])
        return X @ self.w_ + self.b_

    def get_params(self):
        return {k: v for k, v in vars(self).items() if not k.endswith("_")}

    def guardar(self, ruta):
        self._comprobar_ajustado()
        np.savez(ruta, w=self.w_, b=self.b_, clase=type(self).__name__,
                 **{f"hp_{k}": (np.nan if v is None else v)
                    for k, v in self.get_params().items()})

    @classmethod
    def cargar(cls, ruta):
        datos = np.load(ruta, allow_pickle=False)
        if str(datos["clase"]) != cls.__name__:
            raise ValueError(f"El archivo es de {datos['clase']}, no de {cls.__name__}.")
        params = {}
        for k in datos.files:
            if k.startswith("hp_"):
                v = datos[k].item()
                params[k[3:]] = None if isinstance(v, float) and np.isnan(v) else v
        modelo = cls(**params)
        modelo.w_ = datos["w"].astype(float)
        modelo.b_ = float(datos["b"])
        return modelo


class _RegresionMixin:
    def predict(self, X):
        return self.decision(X)

    def objetivo(self, X, y):
        X, y = _validar_Xy(X, y)
        r = self.decision(X) - y
        lam1 = getattr(self, "lambda1", 0.0)
        lam2 = getattr(self, "lambda2", 0.0)
        return (0.5 * np.mean(r ** 2) + lam1 * np.sum(np.abs(self.w_))
                + 0.5 * lam2 * np.sum(self.w_ ** 2))


# ---------------------------------------------------------------------------
# Soluciones por álgebra lineal
# ---------------------------------------------------------------------------
class OLS(_RegresionMixin, _ModeloLineal):
    """Mínimos cuadrados con intercepto, mediante lstsq (norma mínima si no es única)."""

    def __init__(self, fit_intercept=True):
        super().__init__(fit_intercept=fit_intercept)

    def fit(self, X, y):
        X, y = _validar_Xy(X, y)
        Xa = _aumentar(X, self.fit_intercept)
        theta, _, self.rango_, _ = np.linalg.lstsq(Xa, y, rcond=None)
        self.b_, self.w_ = self._separar(theta)
        self.solucion_unica_ = self.rango_ == Xa.shape[1]
        return self

    def residuo_normal(self, X, y):
        """||X~^T (y - X~ theta)|| / N: debe ser ~0 en el óptimo."""
        X, y = _validar_Xy(X, y)
        Xa = _aumentar(X, self.fit_intercept)
        return float(np.linalg.norm(Xa.T @ (y - self.decision(X))) / X.shape[0])


class Ridge(_RegresionMixin, _ModeloLineal):
    """Ridge con intercepto no penalizado: (X~^T X~/N + lambda2 P) theta = X~^T y / N."""

    def __init__(self, lambda2=1.0, fit_intercept=True):
        super().__init__(fit_intercept=fit_intercept)
        self.lambda2 = lambda2

    def _matriz_P(self, D):
        # P = diag(0, 1, …, 1) con intercepto; P = I si b = 0 es conocido.
        return np.diag(np.r_[0.0, np.ones(D)]) if self.fit_intercept else np.eye(D)

    def fit(self, X, y):
        _no_negativo("lambda2", self.lambda2)
        X, y = _validar_Xy(X, y)
        N, D = X.shape
        Xa = _aumentar(X, self.fit_intercept)
        # P = diag(0, 1, …, 1): el cero deja el intercepto sin penalizar.
        P = self._matriz_P(D)
        # Matriz de las ecuaciones normales de Ridge con la convención (4) (X̃ᵀX̃/N + λ₂P)θ = X̃ᵀy/N. El 1/N viene de J_reg.
        A = Xa.T @ Xa / N + self.lambda2 * P
        c = Xa.T @ y / N
        theta = np.linalg.solve(A, c)
        self.b_, self.w_ = self._separar(theta)
        return self

    def residuo_normal(self, X, y):
        X, y = _validar_Xy(X, y)
        N, D = X.shape
        Xa = _aumentar(X, self.fit_intercept)
        theta = np.r_[self.b_, self.w_] if self.fit_intercept else self.w_
        P = self._matriz_P(D)
        res = (Xa.T @ Xa / N + self.lambda2 * P) @ theta - Xa.T @ y / N
        return float(np.linalg.norm(res))


# ---------------------------------------------------------------------------
# Métodos iterativos
# ---------------------------------------------------------------------------
class _Iterativo(_ModeloLineal):
    """Bucle común: historial, criterio de parada, motivo de terminación y arranque en caliente."""

    def __init__(self, lambda2=0.0, eta=None, tol=1e-8, max_iter=10_000, warm_start=False,
                 fit_intercept=True):
        super().__init__(fit_intercept=fit_intercept)
        self.lambda2 = lambda2
        self.eta = eta
        self.tol = tol
        self.max_iter = max_iter
        self.warm_start = warm_start

    # --- a definir en cada subclase ---
    def _objetivo_theta(self, Xa, y, theta):
        raise NotImplementedError

    def _gradiente_suave(self, Xa, y, theta):
        raise NotImplementedError

    def _constante_lipschitz(self, Xa):
        raise NotImplementedError

    def _paso(self, theta, grad, eta):
        """Descenso por gradiente simple; Lasso lo redefine con el paso proximal."""
        return theta - eta * grad

    def _residuo_parada(self, theta, theta_nuevo, grad, eta):
        return float(np.linalg.norm(grad))

    # --- bucle ---
    def _validar_hiper(self):
        _no_negativo("lambda2", self.lambda2)
        _no_negativo("lambda1", getattr(self, "lambda1", 0.0))
        if not (np.isfinite(self.tol) and self.tol > 0):
            raise ValueError("tol debe ser > 0.")
        if int(self.max_iter) != self.max_iter or self.max_iter < 1:
            raise ValueError("max_iter debe ser un entero >= 1.")
        if self.eta is not None and not (np.isfinite(self.eta) and self.eta > 0):
            raise ValueError("eta debe ser > 0.")

    def _ajustar(self, Xa, y):
        self._validar_hiper()
        D1 = Xa.shape[1]
        if self.warm_start and self.w_ is not None and self.w_.shape[0] == D1 - self._i0:
            theta = np.r_[self.b_, self.w_] if self.fit_intercept else self.w_.copy()
        else:
            theta = np.zeros(D1)

        L = self._constante_lipschitz(Xa)
        eta = self.eta if self.eta is not None else 1.0 / L
        self.eta_usado_ = eta
        self.historial_ = [self._objetivo_theta(Xa, y, theta)]
        self.residuos_ = []
        self.motivo_ = "max_iter"

        for k in range(1, int(self.max_iter) + 1):
            grad = self._gradiente_suave(Xa, y, theta)
            theta_nuevo = self._paso(theta, grad, eta)
            res = self._residuo_parada(theta, theta_nuevo, grad, eta)
            theta = theta_nuevo
            obj = self._objetivo_theta(Xa, y, theta)
            self.historial_.append(obj)
            self.residuos_.append(res)
            if not np.isfinite(obj):
                self.motivo_ = "divergencia"
                break
            if res < self.tol:
                self.motivo_ = "convergencia"
                break

        self.n_iter_ = k
        self.b_, self.w_ = self._separar(theta)
        return self


class RegresionLinealGD(_RegresionMixin, _Iterativo):
    """Regresión lineal por descenso por gradiente, con opción L2."""

    def _objetivo_theta(self, Xa, y, theta):
        r = Xa @ theta - y
        return 0.5 * np.mean(r ** 2) + 0.5 * self.lambda2 * np.sum(theta[self._i0:] ** 2)

    def _gradiente_suave(self, Xa, y, theta):
        N = Xa.shape[0]
        # Gradiente de 1/(2N)||X~ theta - y||^2 respecto a theta.
        grad = Xa.T @ (Xa @ theta - y) / N
        # Derivada de (lambda2/2)||w||^2; el intercepto no se penaliza.
        grad[self._i0:] += self.lambda2 * theta[self._i0:]
        return grad

    def _constante_lipschitz(self, Xa):
        N = Xa.shape[0]
        return np.linalg.norm(Xa, 2) ** 2 / N + self.lambda2

    def fit(self, X, y):
        X, y = _validar_Xy(X, y)
        return self._ajustar(_aumentar(X, self.fit_intercept), y)


class LassoProximal(RegresionLinealGD):
    """Lasso (con L2 opcional) por gradiente proximal (ISTA); b no se penaliza."""

    def __init__(self, lambda1=1.0, lambda2=0.0, eta=None, tol=1e-8, max_iter=10_000,
                 warm_start=False, fit_intercept=True):
        super().__init__(lambda2=lambda2, eta=eta, tol=tol, max_iter=max_iter,
                         warm_start=warm_start, fit_intercept=fit_intercept)
        self.lambda1 = lambda1

    def _objetivo_theta(self, Xa, y, theta):
        return super()._objetivo_theta(Xa, y, theta) + self.lambda1 * np.sum(np.abs(theta[self._i0:]))

    def _paso(self, theta, grad, eta):
        theta_nuevo = theta - eta * grad
        # El operador proximal de eta*lambda1*||w||_1; b no pasa por él.
        i0 = self._i0
        theta_nuevo[i0:] = umbral_suave(theta_nuevo[i0:], eta * self.lambda1)
        return theta_nuevo

    def _residuo_parada(self, theta, theta_nuevo, grad, eta):
        # Residuo de la actualización proximal del enunciado (incluye b).
        return np.linalg.norm(theta_nuevo - theta) / eta

    def optimalidad(self, X, y):
        """Máxima violación de las condiciones de optimalidad (debe ser ~0)."""
        X, y = _validar_Xy(X, y)
        N = X.shape[0]
        r = y - self.decision(X)
        c = X.T @ r / N - self.lambda2 * self.w_
        activo = self.w_ != 0
        viol = np.zeros_like(self.w_)
        viol[activo] = np.abs(c[activo] - self.lambda1 * np.sign(self.w_[activo]))
        viol[~activo] = np.maximum(np.abs(c[~activo]) - self.lambda1, 0.0)
        viol_b = abs(np.mean(r)) if self.fit_intercept else 0.0
        return float(max(viol.max(initial=0.0), viol_b))


class LogisticaGD(_Iterativo):
    """Regresión logística por descenso por gradiente, con opción L2."""

    def _objetivo_theta(self, Xa, y, theta):
        return (perdida_logistica(Xa @ theta, y)
                + 0.5 * self.lambda2 * np.sum(theta[self._i0:] ** 2))

    def _gradiente_suave(self, Xa, y, theta):
        N = Xa.shape[0]
        p = sigmoide(Xa @ theta)
        # dl_i/dz_i, obtenido en el Problema 3a).
        grad = Xa.T @ (p - y) / N
        grad[self._i0:] += self.lambda2 * theta[self._i0:]
        return grad

    def _constante_lipschitz(self, Xa):
        N = Xa.shape[0]
        # La constante de Lipschitz de la gradiente de J_log es <= 1/4 * ||X~||_2^2 / N; la de J_reg es ||X~||_2^2 / N. La suma de ambas es una constante de Lipschitz para F.
        return 0.25 * np.linalg.norm(Xa, 2) ** 2 / N + self.lambda2

    def fit(self, X, y):
        X, y = _validar_Xy(X, y, binaria=True)
        return self._ajustar(_aumentar(X, self.fit_intercept), y)

    def predict_proba(self, X):
        p1 = sigmoide(self.decision(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, tau=0.5):
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau debe estar en [0, 1].")
        return (self.predict_proba(X)[:, 1] >= tau).astype(int)

    def objetivo(self, X, y):
        X, y = _validar_Xy(X, y, binaria=True)
        return (perdida_logistica(self.decision(X), y)
                + 0.5 * self.lambda2 * np.sum(self.w_ ** 2))