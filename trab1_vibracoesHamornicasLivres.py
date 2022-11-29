import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
from scipy.integrate import odeint

# Parametros gerais (não importa se o movimento é amortecido
# ou não)

# massa do oscilador em Kg
m = 50 

# rigidez em N/m
k = 10 

# frequência natural em rad/s
wn = np.sqrt(k/m) 

# amplitude 
A = 1

# tempo
delta = 0.01
time =  np.arange(0.0, 180.0, delta)

# Solução analítica da equação do movimento
def systemFreeVibration(y,t):
    xp = y[1]
    xpp = -k/m * y[0] - c/m * y[1] 
    dy = [xp, xpp]
    return dy

# Condições iniciais
u0 = 1.0
v0 = 0

# -----------------------------------------------------------
# Movimento não amortecido
# -----------------------------------------------------------

ksi = 0 # critério de amortecimento
wd = wn # frequencia natural de amortecimento
c = 2*ksi*wn*m 
phi = np.arctan(v0/(u0*wn)) # fase

A = np.sqrt(u0**2 + (v0**2)/wn**2)

# Solução numérica 
u = [A*np.cos(wn*t - phi) for t in time]
v = [-A*np.sin(wn*t - phi)*wn for t in time]

# Solução analítica
uv_num = odeint(systemFreeVibration, [u0,v0], time )

# Plotagem dos gráficos
plt.figure('Grafico 1 - ksi = 0')
plt.title("Deslocamento de Movimento Não Amortecido")
plt.plot(time, uv_num[:,0])
plt.plot(time, u, linestyle="dashed", color="red")
plt.xlabel('Tempo(s)')
plt.ylabel('Deslocamento(m)')

plt.figure('Grafico 2 - ksi = 0')
plt.title("Velocidade de Movimento Não Amortecido")
plt.plot(time, uv_num[:,1], "orange")
plt.plot(time, v, linestyle="dashed", color="green")
plt.xlabel('Tempo(s)')
plt.ylabel('Velocidade(m/s)')
plt.show()


# -----------------------------------------------------------
# Movimento criticamente amortecido
# -----------------------------------------------------------

ksi = 1 # critério de amortecimento
wd = wn*np.sqrt(1- ksi**2) # frequencia natural de amortecimento
c = 2*ksi*wn*m 
phi = 0 # fase

A1 = u0
A2 = v0 + u0*wn

# Solução numérica 
u = [np.exp(-wn*t)*(A1 + A2*t) for t in time]
v = [np.exp(-wn*t)*A2 -wn*np.exp(-wn*t)*(A1 + A2*t) for t in time]

# Solução analítica
uv_num = odeint(systemFreeVibration, [u0,v0], time )

# Plotagem dos gráficos
plt.figure('Grafico 3 - ksi = 1')
plt.title("Deslocamento de Movimento Criticamente Amortecido")
plt.plot(time, uv_num[:,0])
plt.plot(time, u, linestyle="dashed", color="red")
plt.xlabel('Tempo(s)')
plt.ylabel('Deslocamento(m)')

plt.figure('Grafico 4 - ksi = 1')
plt.title("Velocidade de Movimento Criticamente Amortecido")
plt.plot(time, uv_num[:,1], "orange")
plt.plot(time, v, linestyle="dashed", color="green")
plt.xlabel('Tempo(s)')
plt.ylabel('Velocidade(m/s)')
plt.show()

# -----------------------------------------------------------
# Movimento superamortecido
# -----------------------------------------------------------

ksi = 2 # critério de amortecimento
c = 2*ksi*wn*m 
phi = 0 # fase

C1 = u0
C2 = (v0 + u0*ksi*wn)/(wn*np.sqrt(ksi**2 - 1))

# Solução numérica 
u = [np.exp(-wn*ksi*t)*(C1*np.cosh(wn*np.sqrt(ksi**2 - 1)*t) + 
C2*np.sinh(wn*np.sqrt(ksi**2 - 1)*t)) for t in time]
v = [
-ksi*wn*np.exp(-ksi*wn*t)*(C1*np.cosh(wn*np.sqrt(ksi**2 - 1)*t) + 
C2*np.sinh(wn*np.sqrt(ksi**2 - 1)*t)) + 
np.exp(-wn*ksi*t)*(C1*wn*np.sqrt(ksi**2 - 1)
*np.sinh(wn*np.sqrt(ksi**2 - 1)*t) + C2*wn*np.sqrt(ksi**2 - 1)
*np.cosh(wn*np.sqrt(ksi**2 - 1)*t))
    for t in time]

# Solução analítica
uv_num = odeint(systemFreeVibration, [u0,v0], time )

# Plotagem dos gráficos
plt.figure('Grafico 5 - ksi > 1')
plt.title("Deslocamento de Movimento Superamortecido")
plt.plot(time, uv_num[:,0])
plt.plot(time, u, linestyle="dashed", color="red")
plt.xlabel('Tempo(s)')
plt.ylabel('Deslocamento(m)')

plt.figure('Grafico 6 - ksi > 1')
plt.title("Velocidade de Movimento Superamortecido")
plt.plot(time, uv_num[:,1], "orange")
plt.plot(time, v, linestyle="dashed", color="green")
plt.xlabel('Tempo(s)')
plt.ylabel('Velocidade(m/s)')
plt.show()

# -----------------------------------------------------------
# Movimento subamortecido
# -----------------------------------------------------------

ksi = 0.08 # critério de amortecimento
wd = wn*np.sqrt(1- ksi**2) # frequencia natural de amortecimento
c = 2*ksi*wn*m 
phi = np.arctan(ksi*wn/wd) # fase

C1 = u0
C2 = (v0 + u0*ksi*wn)/wd

# Solução numérica 
u = [np.exp(-ksi*wn*t)*(C1*np.cos(wd*t) + C2*np.sin(wd*t)) for t in time]
v = [-ksi*wn*np.exp(-ksi*wn*t)*(C1*np.cos(wd*t) + C2*np.sin(wd*t)) +
np.exp(-ksi*wn*t)*(-C1*wd*np.sin(wd*t) + C2*wd*np.cos(wd*t) ) for t in time]

# Solução analítica
uv_num = odeint(systemFreeVibration, [u0,v0], time )

# Plotagem dos gráficos
plt.figure('Grafico 7 - 0 < ksi < 1')
plt.title("Deslocamento de Movimento Criticamente Amortecido")
plt.plot(time, uv_num[:,0])
plt.plot(time, u, linestyle="dashed", color="red")
plt.xlabel('Tempo(s)')
plt.ylabel('Deslocamento(m)')

plt.figure('Grafico 8 - 0 < ksi < 1')
plt.title("Velocidade de Movimento Criticamente Amortecido")
plt.plot(time, uv_num[:,1], "orange")
plt.plot(time, v, linestyle="dashed", color="green")
plt.xlabel('Tempo(s)')
plt.ylabel('Velocidade(m/s)')
plt.show()

fig = px.line(x=time, y=u)
fig.show()

nT = 5
y_d = np.array([0.6, 0.0483])
decrement = np.log(y_d[0]/y_d[1])/nT
print(f"Decremento exponencial {decrement}")

ksi_est = decrement/np.sqrt(4*np.pi**2 + decrement**2)
print(f"Fator de amortecimento estimado {ksi_est}")
print(f"Fator de amortecimento do grafico {ksi}")







