import math
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

#Zufaellige Konfiguration
def initialisation(N):
    state = 2*np.random.randint(2, size=(N,N))-1

    return state

#Monte-Carlo-Schritte
def mcstep(config, beta):
    N = len(config)

    for i in range(N):
        for j in range(N):
            x = np.random.randint(0, N)
            y = np.random.randint(0, N)
            s = config[x,y]
            neighbours = config[(x+1)%N, j] + config[(x-1)%N, j] + config[i, (y+1)%N] + config[i, (y-1)%N]
            cost = 2*s*neighbours

            if cost < 0:
                s *= -1
            elif rand() < np.exp(-cost * beta):
                s *= -1
            
            config[x, y] = s

#Berechnet die Energy einer gegebenen Konfiguration
def calcEnergy(config):
    energy = 0
    N = len(config)

    for i in range(N):
        for j in range(N):
            S = config[i,j]
            neighbours = config[(i+1)%N, j] + config[(i-1)%N, j] + config[i, (j+1)%N] + config[i, (j-1)%N]
            energy += -neighbours*S/8.

    return energy

#Berechnet die Magnetisierung einer gegebenen Konfiguration
def calcMag(config):
    mag = np.sum(config)

    return mag


#Anfangswerte
T = np.array([1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.])
L_0 = 4 #Groesse des kleinsten Gitters
N = 20 #Anzahl der Daten zur Berechnung des Mittelwertes
eqSteps = 1000 #MC-Schritte fuers Gleichgewicht
mcSteps = 30000 #MC-Schritte fuer die Berechnung

for l in range(7):
    L = L_0 + l
    
    file = open('Plots/Gitter' + str(L) + '.csv', 'w')
    file.write('Temperatur' + ';' + 'Magnetisierung' + ';' + 'statistischer Fehler' + '\n')
    print('----------------------', 'Start Gitter', L, '----------------------')
    
    #Variablen
    n1 = 1.0/(mcSteps * L**2)
    m1 = 1.0/N
    NT = np.size(T)

    Magnetisation = np.zeros(NT)
    MagStat = np.zeros(NT)

    for n in range(NT):
        print('-----------', 'Temperatur', T[n], '-----------')
        Mag = np.zeros(N)
        iT = 1.0/T[n]

        for m in range(N):
            print(m/N*100, '%', end=(', '))
            
            M1 = 0
            config = initialisation(L)

            #Gleichgewicht durch MC-Schritte
            for i in range(eqSteps):
                mcstep(config, iT)

            for i in range(mcSteps):
                mcstep(config, iT)
                M = calcMag(config) #Magnetisierung

                M1 += n1*abs(M)
            
            Mag[m] += M1
            
        Magnetisation[n] = np.sum(m1*Mag)
        MagStat[n] = np.std(Mag)
        
        file.write(str(T[n]) + ';' + str(Magnetisation[n]) + ';' + str(MagStat[n]) + '\n')

    file.close()

    fig = plt.figure(figsize=(18, 10))
    plt.errorbar(T, Magnetisation, yerr=MagStat, fmt='o')
    plt.title('Gitter mit L =' + str(L), fontsize=20)
    plt.xlabel("Temperatur (T) in J/k", fontsize=14)
    plt.ylabel("Magnetization", fontsize=14)
    fig.savefig('Plots/Gitter' + str(L) + '.png')

    print('----------------------', 'Ende Gitter', L, '----------------------')
