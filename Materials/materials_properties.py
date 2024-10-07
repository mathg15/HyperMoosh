import numpy as np


class Material:
    def __init__(self, permittivity, permeability):
        self.permittivity = permittivity
        self.permeability = permeability

    def get_permittivity(self, freq):
        return self.permittivity

    def get_permeability(self, freq):
        return self.permeability


class ExpData(Material):

    def __init__(self, range_freq, permittivities, permeabilities):
        self.range_freq = range_freq
        self.permittivities = permittivities
        self.permeabilities = permeabilities

    def get_permittivity(self, freq):
        try:
            return np.interp(freq, self.range_freq, self.permittivities)
        except:
            return self.permittivities

    def get_permeability(self, freq):
        try:
            return np.interp(freq, self.range_freq, self.permeabilities)
        except:
            return self.permeabilities


def epsilon_maxwell_loss_angle(freq, epsilon_1, epsilon_2, conductivity):
    omega = 2 * np.pi * freq
    delta = np.arctan((epsilon_2 + conductivity / (omega * 8.854e-12) / epsilon_1))
    return epsilon_1 * (1 - 1j * np.tan(delta))


def epsilon_maxwell(freq, epsilon1, epsilon2, conductivity):
    omega = 2 * np.pi * freq
    return epsilon1 - 1j * epsilon2 - 1j * conductivity / (8.854e-12 * omega)


def epsilon_conductivity(freq, conductivity):
    omega = 2 * np.pi * freq
    return -1j * conductivity / (8.854e-12 * omega)


def mu_landau(freq, freq_reso, Mu_Dc):
    omega = 2 * np.pi * freq
    omega_0 = 2 * np.pi * freq_reso
    gamma = 0.7 * omega_0
    return 1 + (Mu_Dc - 1) * (omega_0 ** 2 + 1j * omega * gamma) / (
            omega_0 ** 2 - omega ** 2 + 2j * np.pi * omega * gamma)


def epsilon_lorentz(freq, freq_plasma, omega_0, gamma):
    omega = 2 * np.pi * freq
    return 1 + freq_plasma ** 2 / (omega_0 ** 2 - omega ** 2 + 1j * gamma * omega)


def epsilon_drude(freq, freq_plasma, gamma):
    omega = 2 * np.pi * freq
    return 1 + (freq_plasma ** 2) / (omega ** 2 + gamma ** 2) + 1j * (freq_plasma ** 2 * gamma) / (
            omega * (omega ** 2 + gamma ** 2))
