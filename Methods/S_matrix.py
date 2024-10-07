import numpy as np
import copy


class Scattering_matrix_method:

    def __init__(self, structure):
        self.structure = structure

    def cascade(self, A, B):
        t = 1 / (1 - B[0, 0] * A[1, 1])
        S = np.array([[A[0, 0] + A[0, 1] * B[0, 0] * A[1, 0] * t, A[0, 1] * B[0, 1] * t],
                      [B[1, 0] * A[1, 0] * t, B[1, 1] + A[1, 1] * B[0, 1] * B[1, 0] * t]], dtype=complex)
        return S

    def calculation(self, angle, freq, polarization):
        permittivity, permeability = self.structure.layers(freq)
        thickness = copy.deepcopy(self.structure.thickness)
        eps_0 = 8.85418782e-12
        mu_0 = 1.25663706e-6
        c = int(np.sqrt(1 / (eps_0 * mu_0)))
        thickness[0] = 0
        g = len(permittivity)
        omega = 2 * np.pi * freq
        k_0 = omega / c
        kx = np.sqrt(permittivity[0] * permeability[0]) * k_0 * np.sin(angle * (np.pi / 180))
        kz = np.sqrt(permittivity * permeability * (k_0 ** 2) - np.ones(g) * (kx ** 2))

        if polarization == 'TE':
            f = permeability
        elif polarization == 'TM':
            f = permittivity

        if np.real(permittivity[0]) < 0 and np.real([0]) < 0:
            kz[0] = -kz[0]

        if g > 2:
            kz[1:g - 2] = kz[1:g - 2] * (
                    1 - 2 * (np.imag(kz[1:g - 2]) < 0))
        if np.real(permittivity[g - 1]) < 0 and np.real(
                [g - 1]) < 0 and np.real(np.sqrt(permittivity[g - 1] * [
            g - 1] * k_0 ** 2 - kx ** 2)) != 0:
            kz[g - 1] = -np.sqrt(
                permittivity[g - 1] * [g - 1] * k_0 ** 2 - kx ** 2)
        else:
            kz[g - 1] = np.sqrt(
                permittivity[g - 1] * permeability[g - 1] * k_0 ** 2 - kx ** 2)

        T = np.zeros((2 * g, 2, 2), dtype=complex)
        T[0] = [[0, 1], [1, 0]]

        for k in range(g - 1):
            # Définition de la matrice de couche
            t = np.exp(- 1j * kz[k] * thickness[k])
            T[2 * k + 1] = np.array([[0, t], [t, 0]])

            # Définition de la matrice d'interface
            b1 = kz[k] / f[k]
            b2 = kz[k + 1] / f[k + 1]
            T[2 * k + 2] = [[(b1 - b2) / (b1 + b2), (2 * b2) / (b1 + b2)],
                            [(2 * b1) / (b1 + b2), (b2 - b1) / (b1 + b2)]]

        t = np.exp(-1j * kz[g - 1] * thickness[g - 1])
        T[2 * g - 1] = [[0, t], [t, 0]]

        A = np.zeros((2 * g - 1, 2, 2), dtype=complex)
        A[0] = T[0]

        for j in range(len(T) - 2):
            A[j + 1] = self.cascade(A[j], T[j + 1])

        r = A[len(A) - 1][0, 0]
        t = A[len(A) - 1][1, 0]
        R = np.abs(r) ** 2
        T = np.real(
            abs(t) ** 2 * kz[g - 1] * f[0] / (kz[0] * f[g - 1]))

        return r, R, t, T

    def Coef(self, polarization, freq_range, nb_points, incidence_angle):
        r = np.ones((nb_points, 1), dtype=complex)
        R = np.ones((nb_points, 1))
        t = np.ones((nb_points, 1), dtype=complex)
        T = np.ones((nb_points, 1))

        for i in range(nb_points):
            freq = freq_range[i]
            r[i], R[i], t[i], T[i] = self.calculation(incidence_angle, freq, polarization)
        return r, R, t, T

    def Coef_Log(self, polarization, freq_range, nb_points, incidence_angle):
        r = np.ones((nb_points, 1), dtype=complex)
        R = np.ones((nb_points, 1))
        t = np.ones((nb_points, 1), dtype=complex)
        T = np.ones((nb_points, 1))

        for i in range(nb_points):
            freq = freq_range[i]
            r[i], R[i], t[i], T[i] = self.calculation(incidence_angle, freq, polarization)
        return 10 * np.log10(R), 10 * np.log10(T)

    def Phase(self, polarization, freq_range, nb_points, incidence_angle):
        r = np.ones((nb_points, 1), dtype=complex)
        R = np.ones((nb_points, 1))
        t = np.ones((nb_points, 1), dtype=complex)
        T = np.ones((nb_points, 1))

        for i in range(nb_points):
            freq = freq_range[i]
            r[i], R[i], t[i], T[i] = self.calculation(incidence_angle, freq, polarization)
        return np.angle(r), np.angle(t)

    def SE(self, polarization, freq_range, nb_points, incidence_angle):  # Pour calculer l'efficacité de blindage
        r = np.ones((nb_points, 1), dtype=complex)
        R = np.ones((nb_points, 1))
        t = np.ones((nb_points, 1), dtype=complex)
        T = np.ones((nb_points, 1))

        for i in range(nb_points):
            freq = freq_range[i]
            r[i], R[i], t[i], T[i] = self.calculation(incidence_angle, freq, polarization)

        SE = -10 * np.log10(np.abs(T))
        max_SE = np.max(SE)
        min_SE = np.min(SE)
        return SE, max_SE, min_SE