import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

# Parametros
m = 30 # Kg
k = 15000 # N/m
c = 0.0001 
wn = np.sqrt(k/m)
print(f'wn: {wn}')

A = 1 # Amplitude
phi = 0 # Fase
delta = 0.01 
time = np.arange(0.0, 5.0, delta) # Tempo

# t = np.linspace(0, 5 , 300, endpoint=True)
Tp = 2*np.pi/wn # Periodo

# Deslocamento
u = [A*np.cos(wn*t - phi) for t in time]
v = [-A*np.sin(wn*t - phi)*wn for t in time]

# Obtencao do deslocamento e velocidade por 
# solucao numerica odeint 

# Eq do sistema (no espaco de estados)
def systemFreeVibration(y,t):
    xp = y[1]
    xpp = -k/m * y[0] - c/m * y[1] # - F/m
    dy = [xp, xpp]
    return dy

# Condicoes iniciais
ui = 1.0
vi = 0

# Solucionando o problema com um odeint
uvnum = odeint(systemFreeVibration, [ui,vi], time)

# Plotando os resultados
plt.figure()
plt.plot(time,uvnum[:, 0])
plt.plot(time,u, linestyle="dashed", color="red")
plt.xlabel('Time(s)')
plt.ylabel('Deslocamento(m)')
plt.show()

plt.figure()
plt.plot(time,uvnum[:, 1], "red")
plt.plot(time,v, linestyle="dashed", color="green" )
plt.xlabel('Time(s)')
plt.ylabel('Velocidade(m/s)')
plt.show()
