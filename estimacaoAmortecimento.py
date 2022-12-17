# Para executar no terminal: 
# $ python3 estimacaoAmortecimento.py
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

data_exp = np.loadtxt('massa4/test3-4/tps.txt')

time = data_exp[:,0]
deslocamento1 = data_exp[:,2]

# acesso a pontos no gráfico
# fig = px.line(x=time, y=deslocamento1)
# fig.show()
nT = 20
yd = np.array([2.93, 0.036 ])
decrement = np.log(yd[0]/yd[1])/nT

xi_estimado = decrement/np.sqrt(4*np.pi**2 + decrement**2)
print(f"ksi estimado: {xi_estimado}")

T =  0.5 - 0.46
wd = 2*np.pi/T
wn = (wd/np.sqrt(1 - xi_estimado**2))/(2*np.pi)

print(f'wd estimado [rad/s]: {wd}')
print(f'wn estimado [Hz]: {wn}')

plt.figure()
plt.plot(time, deslocamento1)
plt.show()

# Calculo de wn teorico
E = 210*10**9
b = 25*10**-3
h = 6*10**-3
I = (b*h**3)/12
me = 0.054 
mn = 0.128
mb = 0.445
L = 350*10**-3

wn_anali = np.sqrt((3*E*I)/((L**3)*(me + mn + 0.23*mb)) )/(2*np.pi)
print('wn teórico [Hz]:', wn_anali )