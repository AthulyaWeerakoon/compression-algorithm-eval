from typing import Sequence
import numpy as np
from scipy.stats import norm, beta as sp_beta, gamma as sp_gamma, invgamma
from compression_evaluation.types import ParametricDistribution


class Gaussian(ParametricDistribution):
    """Normal distribution N(μ, σ²)"""

    def __init__(self, mean: float, variance: float, weight: float = None):
        if weight is None:
            self._weight = 1
        else:
            self._weight = weight
        self.mean = mean
        self.variance = variance

    def parameters(self) -> Sequence[float]:
        return self.mean, self.variance

    def mode(self) -> float:
        return self.mean

    def pdf(self, x: float) -> float:
        return norm.pdf(x, loc=self.mean, scale=np.sqrt(self.variance))

    @property
    def name(self) -> str:
        return "Gaussian"

    @property
    def weight(self) -> float:
        return self._weight


class Beta(ParametricDistribution):
    """Beta(α, β) distribution"""

    def __init__(self, alpha: float, beta: float, weight: float = None):
        if weight is None:
            self._weight = 1
        else:
            self._weight = weight
        self.alpha = alpha
        self.beta = beta

    def parameters(self) -> Sequence[float]:
        return self.alpha, self.beta

    def mode(self) -> float:
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        elif self.alpha <= 1 < self.beta:
            return 0.0
        elif self.alpha > 1 >= self.beta:
            return 1.0
        else:
            return 0.5

    def pdf(self, x: float) -> float:
        return sp_beta.pdf(x, self.alpha, self.beta)

    @property
    def name(self) -> str:
        return "Beta"

    @property
    def weight(self) -> float:
        return self._weight


class Gamma(ParametricDistribution):
    """Gamma(α, β) distribution (shape α, rate β)"""

    def __init__(self, alpha: float, beta: float, weight: float = None):
        if weight is None:
            self._weight = 1
        else:
            self._weight = weight
        self.alpha = alpha
        self.beta = beta

    def parameters(self) -> Sequence[float]:
        return self.alpha, self.beta

    def mode(self) -> float:
        return (self.alpha - 1) / self.beta if self.alpha >= 1 else 0.0

    def pdf(self, x: float) -> float:
        return sp_gamma.pdf(x, a=self.alpha, scale=1/self.beta)

    @property
    def name(self) -> str:
        return "Gamma"

    @property
    def weight(self) -> float:
        return self._weight


class InverseGamma(ParametricDistribution):
    """Inverse-Gamma(α, β) distribution"""

    def __init__(self, alpha: float, beta: float, weight: float = None):
        if weight is None:
            self._weight = 1
        else:
            self._weight = weight
        self.alpha = alpha
        self.beta = beta

    def parameters(self) -> Sequence[float]:
        return self.alpha, self.beta

    def mode(self) -> float:
        if self.alpha <= 1:
            raise ValueError("Mode undefined for α ≤ 1")
        return self.beta / (self.alpha + 1)

    def pdf(self, x: float) -> float:
        return invgamma.pdf(x, a=self.alpha, scale=self.beta)

    @property
    def name(self) -> str:
        return "Inverse-Gamma"

    @property
    def weight(self) -> float:
        return self._weight


class ScaledInvChiSquared(ParametricDistribution):
    """Scaled Inverse Chi-squared(ν, σ²) distribution"""

    def __init__(self, nu: float, sigma_sq: float, weight: float = None):
        if weight is None:
            self._weight = 1
        else:
            self._weight = weight
        self.nu = nu
        self.sigma_sq = sigma_sq

    def parameters(self) -> Sequence[float]:
        return self.nu, self.sigma_sq

    def mode(self) -> float:
        if self.nu <= 2:
            raise ValueError("Mode undefined for ν ≤ 2")
        return (self.nu * self.sigma_sq) / (self.nu + 2)

    def pdf(self, x: float) -> float:
        return invgamma.pdf(x, a=self.nu/2, scale=(self.nu * self.sigma_sq)/2)

    @property
    def name(self) -> str:
        return "Scaled-Inverse-Chi-Squared"

    @property
    def weight(self) -> float:
        return self._weight
