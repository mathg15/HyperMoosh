import numpy as np


class Structure:
    def __init__(self, materials, thickness):
        self.materials = materials
        thickness = np.insert(thickness, 0, 10e-2)  # Couche d'air semi inf
        thickness = np.append(thickness, 10e-2)  # Couche d'air semi inf
        self.thickness = thickness

    def layers(self, freq):
        permittivities = np.ones_like(self.materials, dtype=complex)
        permeabilities = np.ones_like(self.materials, dtype=complex)

        for i in range(len(self.materials)):
            material = self.materials[i]
            permittivities[i] = material.get_permittivity(freq)
            permeabilities[i] = material.get_permeability(freq)

        permeabilities = np.insert(permeabilities, 0, 1)  # Couche d'air semi inf
        permittivities = np.insert(permittivities, 0, 1)  # Couche d'air semi inf
        permeabilities = np.append(permeabilities, 1)
        permittivities = np.append(permittivities, 1)

        return permittivities, permeabilities
