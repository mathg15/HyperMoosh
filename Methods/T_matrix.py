import numpy as np
import copy


class Transfer_matrix_method:

    def __init__(self, structure):
        self.structure = structure

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

        # Changing the determination of the square root to achieve perfect stability
        if g > 2:
            kz[1:g - 2] = kz[1:g - 2] * (
                    1 - 2 * (np.imag(kz[1:g - 2]) < 0))
        # Outgoing wave condition for the last medium
        if np.real(permittivity[g - 1]) < 0 and np.real(
                [g - 1]) < 0 and np.real(np.sqrt(permittivity[g - 1] * [
            g - 1] * k_0 ** 2 - kx ** 2)) != 0:
            kz[g - 1] = -np.sqrt(
                permittivity[g - 1] * [g - 1] * k_0 ** 2 - kx ** 2)
        else:
            kz[g - 1] = np.sqrt(
                permittivity[g - 1] * permeability[g - 1] * k_0 ** 2 - kx ** 2)

        T = np.zeros((2 * g - 2, 2, 2), dtype=complex)
        gamma = kz / f
        sum = (gamma[1:] + gamma[:-1]) / (2 * gamma[1:])
        dif = (gamma[1:] - gamma[:-1]) / (2 * gamma[1:])
        phases = np.exp(1.j * kz[1:] * thickness[1:])
        for k in range(g - 1):
            # Layer transfer matrix
            T[2 * k] = [[sum[k], dif[k]],
                        [dif[k], sum[k]]]

            # Layer propagation matrix
            T[2 * k + 1] = [[phases[k], 0],
                            [0, 1 / phases[k]]]
        # Once the scattering matrixes have been prepared, now let us combine them

        A = np.empty((2, 2), dtype=complex)
        A = T[0]
        for i in range(1, T.shape[0]):
            A = A @ T[i]
        # reflection coefficient of the whole structure
        r = A[1, 0] / A[0, 0]
        # transmission coefficient of the whole structure
        t = 1 / A[0, 0]
        # Energy reflexion coefficient;
        R = np.real(abs(r) ** 2)
        # Energy transmission coefficient;
        T = np.real(abs(t) ** 2)

        return r, t, R, T

    def SE(self, polarization, freq_range, nb_points, incidence_angle):  # Pour calculer l'efficacité de blindage
        r = np.ones((nb_points, 1), dtype=complex)
        R = np.ones((nb_points, 1), dtype=complex)
        t = np.ones((nb_points, 1), dtype=complex)
        T = np.ones((nb_points, 1), dtype=complex)

        for i in range(nb_points):
            freq = freq_range[i]
            r[i], R[i], t[i], T[i] = self.calculation(incidence_angle, freq, polarization)

        SE = -10 * np.log10(np.abs(T))
        max_SE = np.max(SE)
        min_SE = np.min(SE)
        return SE, max_SE, min_SE

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