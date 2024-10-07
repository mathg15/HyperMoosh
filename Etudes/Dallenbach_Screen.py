import numpy as np
import matplotlib.pyplot as plt

from Methods import S_matrix
from Materials import materials_properties, multilayer

n_points = 1000
range_freq_mats = np.logspace(7, 11, n_points)
range_freq = np.linspace(1e9, 12e9, 1000)

permittivities_air = materials_properties.epsilon_maxwell(range_freq_mats, 1, 0, 0)
material_air = materials_properties.ExpData(range_freq, permittivities_air, 1)
permittivities_A = materials_properties.epsilon_maxwell(range_freq_mats, 10, 5, 0)
material_A = materials_properties.ExpData(range_freq, permittivities_A, 1)
permittivities_B = materials_properties.epsilon_maxwell(range_freq_mats, 1, 0, 1e7)
material_B = materials_properties.ExpData(range_freq, permittivities_B, 1)

Dallenbach = np.array([material_A, material_B])
thickness_Dallenbach = np.array([3e-3, 5e-6])
structure_Dallenbach = multilayer.Structure(Dallenbach.flatten(), thickness_Dallenbach)

Simu_Smatrix_Dallenbach = S_matrix.Scattering_matrix_method(structure_Dallenbach)
# range_freq = np.array([5e9])
Coef_S_Dallenbach = Simu_Smatrix_Dallenbach.Coef_Log(polarization='TE', freq_range=range_freq, nb_points=n_points,
                                                 incidence_angle=0)

plt.plot(range_freq, Coef_S_Dallenbach[0], label='Reflexion')
plt.xlabel('Frequency (GHz)')
plt.xticks(np.array([0.2, 0.4, 0.6, 0.8, 1, 1.2]) * 1e10, np.array([2, 4, 6, 8, 10, 12]))
plt.ylabel('Return loss (dB)')
plt.show()

