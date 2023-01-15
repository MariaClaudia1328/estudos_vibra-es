import numpy as np 
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import plotly.express as px

# Parametros do sistema
k = 720
m = 5
wn = np.sqrt(k/m)

# Fator de amortecimento
# xi = 0.05
# xi = 0.1
# xi = 0.25
# xi = 0.5

# Caso sem dissipação (vibração não amortecida)
xi = 0

# Parametros do forçamento
f0 = 1 
w = 5


# Tempo
delta = 0.01
time =  np.arange(0.0, 30.0, delta)

# Parametros para a resposta
phi = 0
A = 1

# Resposta homogênea
uh = [np.exp(-xi*wn*t)*( A*np.cos(wn*t - phi)) for t in time]

# Amplitude da resposta particular
Up = f0/np.sqrt((1-(w/wn)**2)**2+(2*xi*w/wn)**2)
# Resposta particular
up = [(Up*np.cos(w*t - phi)) for t in time]

# Soma das resposta homogênea com particular
u = np.add(uh, up)

# Condições iniciais
u01 = A*np.cos(phi)+Up*np.cos(phi)
u02 = 0.0

# Solução numérica com as condições iniciais
def numericSolution(y, t):
    F = f0*k*np.cos(w*t-phi)
    xp = y[1]
    xpp = -k/m * y[0] - 2*xi*wn* y[1] + F/m
    dy = [xp, xpp]
    return dy

uv = odeint(numericSolution, [u01, u02], time)


plt.figure("Resposta homogênea e particular - caso subamortecido")
plt.plot(time, uh, time, up)

plt.figure("Soma das respostas homogênea e particular - caso subamortecido")
plt.plot(time,u, color='green')
plt.plot(time, uv[:,0], linestyle="--", color="red")


# Definição do dominio da frequência
omf = np.arange(0,5,0.01)*2*np.pi

# Ganho no domínio da frequência
Gom = [1/np.sqrt((1-(om1/wn)**2)**2+(2*xi*om1/wn)**2) for om1 in omf]

# Fase no domínio da frequência
phase = [ np.arctan2((2*xi*(om1/wn)),(1-(om1/wn)**2)) for om1 in omf ]

# Ganho máximo
Gmax = np.max(Gom)/np.sqrt(2)

# Frequência do ganho máximo
x_gmax = np.ones(omf.size)*Gmax

plt.figure("Ganho e fase no domínio da frequência - caso subamortecido")
plt.plot(omf,x_gmax, color="orange")
plt.plot(omf,Gom, color='green')
plt.plot(omf, phase, color="red")


# Gráfico para auxiliar no calculo da largura de banda
fig = px.line(x=omf/wn, y=[Gom, x_gmax])
# fig.show()

# Largura de banda obtida ksi = 0.05
w2 = 1.05294
w1 = 0.95294
bandwidth = w2 - w1 
# Largura de banda obtida ksi = 0.1
# w2 = 1.089085
# w1 = 0.88488
# bandwidth = w2 - w1 
#Largura de banda obtida ksi = 0.25
# w2 = 1.167625
# w1 = 0.6283185
# bandwidth = w2 - w1 
#Largura de banda obtida ksi = 0.5
# w2 = 1.172861
# w1 = 0
# bandwidth = w2 - w1 

# Estimação do fator de amortecimento a partir da largura de banda
xi_estimado = bandwidth/2
print("Fator de amortecimento estimado: ", "{:.5f}".format(xi_estimado))

# Exemplo 4.1 do livro da Prof Aline 
md = 2
e = 1
Omega = wn*1.0001

U_Omega = (md*e*Omega)/(m*wn**2 - m*Omega**2)

#Assumindo condições iniciais nulas
u = [U_Omega * (np.cos(wn*t) + np.cos(Omega*t)) for t in time]

# Para melhor visualização do gráfico quando Omega = wn/2 e Omega = wn/3
# u = [500*U_Omega * (np.cos(wn*t) + np.cos(Omega*t)) for t in time]

F0 = [md*e*Omega**2 * np.cos(Omega*t) for t in time]

plt.figure("Exercício 4.1")
plt.plot(time, F0)
plt.plot(time, u)


plt.show()