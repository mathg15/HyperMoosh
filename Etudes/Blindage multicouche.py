import numpy as np
import matplotlib.pyplot as plt

from Methods import S_matrix
from Materials import materials_properties, multilayer

n_pts = 1000
range_freq = np.linspace(1e7, 1e11, n_pts)


permittivities_cond = materials_properties.epsilon_maxwell(range_freq, 1, 0, 1e2)
material_cond = materials_properties.ExpData(range_freq, permittivities_cond, 1)

Blindage = np.array([material_cond])
thickness_Blindage = np.array([1e-3])
structure_Blindage = multilayer.Structure(Blindage.flatten(), thickness_Blindage)

Simu_Smatrix_Blindage = S_matrix.Scattering_matrix_method(structure_Blindage)
# range_freq = np.array([5e9])
SE_Blindage = Simu_Smatrix_Blindage.SE(polarization='TE', freq_range=range_freq, nb_points=n_pts,
                                                 incidence_angle=0)

plt.semilogx(range_freq, SE_Blindage[0], label='Efficacité de blindage')
plt.xlabel('Frequency (Hz)')
# plt.xticks(np.array([0.2, 0.4, 0.6, 0.8, 1, 1.2]) * 1e10, np.array([2, 4, 6, 8, 10, 12]))
plt.ylabel('Efficacité de blindage (dB)')
plt.show()

