import sys
import re

import photon_canon as pc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from photon_canon.hardware import ID


def get_parameters_from_issue():
    # Read issue body
    with open('issue_body.txt', 'r') as f:
        issue_data = f.read()

    # Extract and float params
    match = re.search(r"Index of refraction\s*([\d.]+)", issue_data)
    n = float(match.group(1) if match else 1.5)
    match = re.search(r"Absorption Coefficient\s*([\d.]+)", issue_data).group(1)
    mua = float(match.group(1) if match else 15)
    match = re.search(r"Scattering Coefficient\s*([\d.]+)", issue_data).group(1)
    mus = float(match.group(1) if match else 4)
    match = re.search(r"Scattering anisotropy\s*([\d.-]+)", issue_data).group(1)
    g = float(match.group(1) if match else 0.75)

    return n, mus, mua, g

def make_all_plots(photon, tissue_start=0.217, into_tissue_step=1):
    fig = plt.figure(figsize=(28, 35))
    ax = np.array([
        fig.add_subplot(3, 4, 1, projection='3d'), fig.add_subplot(3, 4, 2), fig.add_subplot(3, 4, 3),
        fig.add_subplot(3, 4, 4),  # 0-3: Paths
        fig.add_subplot(3, 2, 3, projection='3d'), fig.add_subplot(3, 2, 4, projection='3d'),
        # 4-5: Illumination pattern (beam in water and beam into tissue)
        fig.add_subplot(3, 2, 5), fig.add_subplot(3, 2, 6)
        # 6-7: Incidence/detection at tissue and locations at objective
    ])

    photon.plot_path(axes=ax[0])
    photon.plot_path(project_onto='xy', axes=[ax[1]])
    photon.plot_path(project_onto='xz', axes=[ax[2]])
    photon.plot_path(project_onto='yz', axes=[ax[3]])

    # First move
    x, y, z = [photon.location_history[:, i, 0:2] for i in range(3)]
    for i in range(250):
        ax[4].plot(x[i], y[i], z[i])

    # Second move
    x, y, z = [photon.location_history[:, i, into_tissue_step:into_tissue_step + 2] for i in range(3)]
    for i in range(250):
        ax[5].plot(x[i], y[i], z[i])

    # Tissue positions
    # Starting point
    ax[6].scatter(*photon.location_history[:, :2, 1].T, color='b', label='Incident locations')
    # Exit locations
    out_of_tissue = photon.location_history[:, 2] == tissue_start
    out_of_tissue[:, into_tissue_step] = False  # Ignore entry case
    x, y = np.where(out_of_tissue, [photon.location_history[:, 0], photon.location_history[:, 1]], np.nan)
    ax[6].scatter(x[not np.isnan], y[not np.isnan], color='r', label='Exit locations')

    # Objective positions
    # Starting point
    ax[7].scatter(*photon.location_history[:, :2, 0].T, color='b', label='Starting locations')
    # Exit location
    x, y, _ = photon.exit_location.T if photon.exit_location is not None else [(), (), ()]
    ax[7].scatter(x, y, color='r', label='Exit locations')

    ax[0].set(xlim=[-0.5, 0.5], ylim=[-0.5, 0.5], zlim=[0.4, -0.01])
    ax[1].set(xlim=[-0.5, 0.5], ylim=[-0.5, 0.5])
    ax[2].set(xlim=[-0.5, 0.5], ylim=[0.8, -0.01])
    ax[3].set(xlim=[-0.5, 0.5], ylim=[0.8, -0.010])
    ax[4].set(xlim=[-0.5, 0.5], ylim=[-0.5, 0.5], zlim=[0, 0.22])
    ax[4].set_title("Photons' move to tissue")
    ax[4].invert_zaxis()
    ax[5].set(xlim=[-0.2, 0.2], ylim=[-0.2, 0.2], zlim=[0.42, 0.2])
    ax[5].set_title("Photons' first step into tissue")
    ax[6].set(xlim=[-0.05, 0.05], ylim=[-0.05, 0.05])
    ax[6].set_title("Photons' incident and detection locations at tissue interface")
    ax[6].legend()
    ax[7].set(xlim=[-0.5, 0.5], ylim=[-0.5, 0.5])
    ax[7].set_title("Photons' incident and detection locations at objective")
    detector_circle = patches.Circle((0, 0), ID, edgecolor='black', facecolor='none', label='Detector region')
    ax[7].add_patch(detector_circle)
    ax[7].set_title('Incident and Exit locations')
    ax[7].legend()
    fig.tight_layout()

    return fig

def simulate_asset(*args):
    dw = pc.Medium(n=1.33, mu_a=0, mu_s=0, desc='water', display_color='aqua')
    g = pc.Medium(n=1.523, mu_a=0, mu_s=0, desc='glass', display_color='gray')
    t = pc.Medium(n=args[0], mu_a=args[1], mu_s=args[2], g=args[3], desc='user_defined', display_color='lightpink')
    surroundings_n = 1.33

    OD = pc.hardware.OD
    ID = pc.hardware.ID
    theta = np.arctan(-OD / pc.hardware.WD)

    sampler = pc.hardware.ring_pattern((ID, OD), theta)
    LED = pc.Illumination(pattern=sampler)
    detector = pc.Detector(pc.hardware.cone_of_acceptance(ID))

    n = 250

    s = pc.System(
        dw, 0.2,
        g, 0.017,
        t, float('inf'),
        surrounding_n=surroundings_n,
        illuminator=LED,
        detector=(detector, 0)
    )

    detector.reset()
    photon = s.beam(batch_size=n, recurse=False, russian_roulette_constant=20, tir_limit=100)
    photon.simulate()
    fig = make_all_plots(photon, tissue_start=0.217, into_tissue_step=2)

    fig.savefig('assets/simulation.png')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = [float(arg) for arg in sys.argv[1:]]
    else:
        args = get_parameters_from_issue()
    simulate_asset(*args)