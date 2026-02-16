import numpy as np
import matplotlib.pyplot as plt

# Costanti fisiche
M_MU = 105.66
X0_SILICIO = 9.36
DENSITA_SILICIO = 2.33
SPESSORE_CM = 0.04
SPESSORE_GCM2 = SPESSORE_CM * DENSITA_SILICIO
PIXEL_CM = 0.01


def theta_mcs(p, beta, x_over_X0):
    if x_over_X0 <= 0:
        return 0.0
    return (13.6 / p) * np.sqrt(x_over_X0) * (1 + 0.038 * np.log(x_over_X0))


def propaga_muone(E, n_piani, distanza, pixel, x0=0.0, y0=0.0):
    p = np.sqrt(E**2 - M_MU**2)
    beta = p / E

    dx, dy, dz = 0.0, 0.0, 1.0
    x, y, z = x0, y0, 0.0
    sigma = pixel / np.sqrt(12)

    misure = []

    for i in range(n_piani):
        z_piano = i * distanza
        dz_step = z_piano - z

        x += dx * dz_step
        y += dy * dz_step
        z = z_piano

        t0 = theta_mcs(p, beta, SPESSORE_GCM2 / X0_SILICIO)
        dx += np.random.normal(0, t0)
        dy += np.random.normal(0, t0)

        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        dx /= norm
        dy /= norm
        dz /= norm

        xm = np.random.normal(x, sigma)
        ym = np.random.normal(y, sigma)

        misure.append((xm, ym, z))

    return np.array(misure)


def ricostruisci_direzione(misure):
    z = misure[:,2]
    x = misure[:,0]
    y = misure[:,1]

    ax, bx = np.polyfit(z, x, 1)
    ay, by = np.polyfit(z, y, 1)

    dx = ax
    dy = ay
    dz = 1.0

    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    return dx/norm, dy/norm, dz/norm


def simula_fascio(E, N, sigma_beam, n_piani, distanza, pixel):
    hits = []
    dirs = []
    tracce = []

    for _ in range(N):
        x0, y0 = np.random.normal(0, sigma_beam, 2)
        mis = propaga_muone(E, n_piani, distanza, pixel, x0, y0)
        hits.append(mis[-1,:2])
        dirs.append(ricostruisci_direzione(mis))
        tracce.append(mis)

    return np.array(hits), np.array(dirs), tracce


print("CONFIGURAZIONE TRACCIATORE")

while True:
    n_piani = int(input("Numero di piani (consigliato: 6): "))
    if n_piani < 2:
        print("Errore: servono almeno 2 piani.")
    else:
        break

while True:
    distanza = float(input("Distanza tra i piani in cm (consigliato: 10.0): "))
    if distanza <= 0:
        print("Errore: la distanza deve essere positiva.")
    else:
        break

print("\nCONFIGURAZIONE FASCIO")

while True:
    N = int(input("Numero di muoni da simulare (consigliato: 5000): "))
    if N < 50:
        print("Errore: servono almeno 50 muoni.")
    else:
        break

while True:
    sigma_beam = float(input("Sigma iniziale del fascio in cm (consigliato: 0.1): "))
    if sigma_beam < 0:
        print("Errore: sigma deve essere >= 0.")
    else:
        break

print("\nENERGIE")

while True:
    energie_input = input("Inserisci energie in MeV separate da virgola (esempio: 200, 500, 1000, 5000): ")

    try:
        energie = [float(e.strip()) for e in energie_input.split(",")]
    except:
        print("Errore: inserisci solo numeri separati da virgola.")
        continue

    if any(E <= M_MU for E in energie):
        print(f"Errore: ogni energia deve essere > {M_MU} MeV (massa del muone).")
        print("Reinserisci la lista.")
        continue

    break


sigma_x = []
sigma_y = []
ris_ang = []

for E in energie:
    print("\nSimulo", E, "MeV")
    hits, dirs, tracce = simula_fascio(E, N, sigma_beam, n_piani, distanza, PIXEL_CM)

    np.savetxt(f"direzioni_{int(E)}MeV.csv", dirs, delimiter=",",
               header="dx,dy,dz", comments="")

    with open(f"tracce_{int(E)}MeV.csv", "w") as f:
        f.write("muone,x,y,z\n")
        for i, t in enumerate(tracce):
            for (x,y,z) in t:
                f.write(f"{i},{x},{y},{z}\n")

    sx = np.std(hits[:,0])
    sy = np.std(hits[:,1])
    sigma_x.append(sx)
    sigma_y.append(sy)
    ra = np.mean(np.sqrt(dirs[:,0]**2 + dirs[:,1]**2))
    ris_ang.append(ra)

    if sx == 0 and sy == 0:
        print("ATTENZIONE: tutti i muoni cadono nello stesso pixel.")

    if sx < PIXEL_CM/np.sqrt(12) or sy < PIXEL_CM/np.sqrt(12):
        print("ATTENZIONE: dispersione del fascio più piccola della risoluzione del pixel.")

    if np.std(dirs[:,0]) == 0 and np.std(dirs[:,1]) == 0:
        print("ATTENZIONE: tutte le direzioni ricostruite sono identiche.")

    plt.figure(figsize=(7, 6))
    plt.scatter(hits[:,0], hits[:,1], s=5, alpha=0.5)
    plt.xlabel("x ultimo piano [cm]")
    plt.ylabel("y ultimo piano [cm]")
    plt.title(f"Dispersione del fascio all’ultimo piano — {E} MeV")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plt.figure()
plt.loglog(energie, sigma_x, "-o", label="sigma_x")
plt.loglog(energie, sigma_y, "-o", label="sigma_y")
plt.xlabel("Energia [MeV]")
plt.ylabel("Sigma fascio [cm]")
plt.title("Allargamento del fascio all'ultimo piano")
plt.legend()
plt.grid(True)

plt.figure()
plt.loglog(energie, ris_ang, "-o")
plt.xlabel("Energia [MeV]")
plt.ylabel("Risoluzione angolare (rad)")
plt.title("Risoluzione angolare ricostruita")
plt.grid(True)

plt.show()

