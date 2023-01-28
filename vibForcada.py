import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

signal_1Hz = np.loadtxt('massa4-vibForcada/1Hz_tmp_001.txt')
signal_5Hz = np.loadtxt('massa4-vibForcada/5Hz_tmp_001.txt')
signal_9Hz = np.loadtxt('massa4-vibForcada/9Hz_tmp_001.txt')
signal_13Hz = np.loadtxt('massa4-vibForcada/13Hz_tmp_001.txt')
signal_15Hz = np.loadtxt('massa4-vibForcada/15Hz_tmp_001.txt')
signal_16Hz = np.loadtxt('massa4-vibForcada/16Hz_tmp_001.txt')
signal_17Hz = np.loadtxt('massa4-vibForcada/17Hz_tmp_001.txt')
signal_18Hz = np.loadtxt('massa4-vibForcada/18Hz_tmp_001.txt')
signal_19Hz = np.loadtxt('massa4-vibForcada/19Hz_tmp_001.txt')
signal_20Hz = np.loadtxt('massa4-vibForcada/20Hz_tmp_001.txt')
signal_21Hz = np.loadtxt('massa4-vibForcada/21Hz_tmp_001.txt')
signal_23Hz = np.loadtxt('massa4-vibForcada/23Hz_tmp_001.txt')
signal_27Hz = np.loadtxt('massa4-vibForcada/27Hz_tmp_001.txt')
signal_29Hz = np.loadtxt('massa4-vibForcada/29Hz_tmp_001.txt')

time1Hz = signal_1Hz[:,0]
signalAcc1Hz = signal_1Hz[:,1]
signalF1Hz = signal_1Hz[:,2]

time5Hz = signal_5Hz[:,0]
signalAcc5Hz = signal_5Hz[:,1]
signalF5Hz = signal_5Hz[:,2]

time9Hz = signal_9Hz[:,0]
signalAcc9Hz = signal_9Hz[:,1]
signalF9Hz = signal_9Hz[:,2]

time13Hz = signal_13Hz[:,0]
signalAcc13Hz = signal_13Hz[:,1]
signalF13Hz = signal_13Hz[:,2]

time15Hz = signal_15Hz[:,0]
signalAcc15Hz = signal_15Hz[:,1]
signalF15Hz = signal_15Hz[:,2]

time16Hz = signal_16Hz[:,0]
signalAcc16Hz = signal_16Hz[:,1]
signalF16Hz = signal_16Hz[:,2]

time17Hz = signal_17Hz[:,0]
signalAcc17Hz = signal_17Hz[:,1]
signalF17Hz = signal_17Hz[:,2]

time18Hz = signal_18Hz[:,0]
signalAcc18Hz = signal_18Hz[:,1]
signalF18Hz = signal_18Hz[:,2]

time19Hz = signal_19Hz[:,0]
signalAcc19Hz = signal_19Hz[:,1]
signalF19Hz = signal_19Hz[:,2]

time20Hz = signal_20Hz[:,0]
signalAcc20Hz = signal_20Hz[:,1]
signalF20Hz = signal_20Hz[:,2]

time21Hz = signal_21Hz[:,0]
signalAcc21Hz = signal_21Hz[:,1]
signalF21Hz = signal_21Hz[:,2]

time23Hz = signal_23Hz[:,0]
signalAcc23Hz = signal_23Hz[:,1]
signalF23Hz = signal_23Hz[:,2]

time27Hz = signal_27Hz[:,0]
signalAcc27Hz = signal_27Hz[:,1]
signalF27Hz = signal_27Hz[:,2]

time29Hz = signal_29Hz[:,0]
signalAcc29Hz = signal_29Hz[:,1]
signalF29Hz = signal_29Hz[:,2]

fig1Hz = px.line(x=signalAcc1Hz, y = signalF1Hz)
fig5Hz = px.line(x=signalAcc5Hz, y = signalF5Hz)
fig9Hz = px.line(x=signalAcc9Hz, y = signalF9Hz)
fig13Hz = px.line(x=signalAcc13Hz, y = signalF13Hz)
fig15Hz = px.line(x=signalAcc15Hz, y = signalF15Hz)
fig16Hz = px.line(x=signalAcc16Hz, y = signalF16Hz)
fig17Hz = px.line(x=signalAcc17Hz, y = signalF17Hz)
fig18Hz = px.line(x=signalAcc18Hz, y = signalF18Hz)
fig19Hz = px.line(x=signalAcc19Hz, y = signalF19Hz)
fig20Hz = px.line(x=signalAcc20Hz, y = signalF20Hz)
fig21Hz = px.line(x=signalAcc21Hz, y = signalF21Hz)
fig23Hz = px.line(x=signalAcc23Hz, y = signalF23Hz)
fig27Hz = px.line(x=signalAcc27Hz, y = signalF27Hz)
fig29Hz = px.line(x=signalAcc29Hz, y = signalF29Hz)

# fig1Hz.show()

A1Hz = np.abs(1.460881 - 0.106166)
C1Hz = np.abs(1.055074 - 0.52959)
Amp1Hz = A1Hz/2
phi1Hz = np.arcsin(C1Hz/A1Hz)

A5Hz = np.abs(-0.322118 - 1.538033)
C5Hz = np.abs(0.307247 - 0.965165)
Amp5Hz = A5Hz/2
phi5Hz = np.arcsin(C5Hz/A5Hz)

A9Hz = np.abs(-2.557094 - 3.510571)
C9Hz = np.abs(-1.32449 - (-0.246789))
Amp9Hz = A9Hz/2
phi9Hz = np.arcsin(C9Hz/A9Hz)

A13Hz = np.abs(-7.211725 - 8.453762)
C13Hz = np.abs(-0.340951 - (1.853323))
Amp13Hz = A13Hz/2
phi13Hz = np.arcsin(C13Hz/A13Hz)

A15Hz = np.abs(-20.23278 - 22.26943)
C15Hz = np.abs(-3.238097 - (4.64841))
Amp15Hz = A15Hz/2
phi15Hz = np.arcsin(C15Hz/A15Hz)

A16Hz = np.abs(-61.71865 - 64.61065)
C16Hz = np.abs(-59.00497 - (61.84958))
Amp16Hz = A16Hz/2
phi16Hz = np.arcsin(C16Hz/A16Hz)

A17Hz = np.abs(-29.33367 - 30.07817)
C17Hz = np.abs(-17.53429 - (16.21998))
Amp17Hz = A17Hz/2
phi17Hz = np.arcsin(C17Hz/A17Hz)

A18Hz = np.abs(-14.44274 - 15.19878)
C18Hz = np.abs(-7.946795 - (18.361422))
Amp18Hz = A18Hz/2
phi18Hz = np.arcsin(C18Hz/A18Hz)

A19Hz = np.abs(-7.481453 - 7.792199)
C19Hz = np.abs(-7.299204 - (7.582006))
Amp19Hz = A19Hz/2
phi19Hz = np.arcsin(C19Hz/A19Hz)

A20Hz = np.abs(-46.48813 - 47.14454)
C20Hz = np.abs(-35.32054 - (36.4101))
Amp20Hz = A20Hz/2
phi20Hz = np.arcsin(C20Hz/A20Hz)

A21Hz = np.abs(-60.03893 - 61.12241)
C21Hz = np.abs(-17.0325 - (17.50848))
Amp21Hz = A21Hz/2
phi21Hz = np.arcsin(C21Hz/A21Hz)

A23Hz = np.abs(-21.42773 - 21.40314)
C23Hz = np.abs(-0.768628 - (1.248257))
Amp23Hz = A23Hz/2
phi23Hz = np.arcsin(C23Hz/A23Hz)

A27Hz = np.abs(-12.77213 - 13.18129)
C27Hz = np.abs(-0.199404 - (0.866143))
Amp27Hz = A27Hz/2
phi27Hz = np.arcsin(C27Hz/A27Hz)

A29Hz = np.abs(-11.71144 - 12.08476)
C29Hz = np.abs(-0.627689 - (1.073906))
Amp29Hz = A29Hz/2
phi29Hz = np.arcsin(C29Hz/A29Hz)

print("Amplitude 1Hz: ", Amp1Hz)
print("Fase 1Hz: ", phi1Hz)

print("Amplitude 5Hz: ", Amp5Hz)
print("Fase 5Hz: ", phi5Hz)

print("Amplitude 9Hz: ", Amp9Hz)
print("Fase 9Hz: ", phi9Hz)

print("Amplitude 13Hz: ", Amp13Hz)
print("Fase 13Hz: ", phi13Hz)

print("Amplitude 15Hz: ", Amp15Hz)
print("Fase 15Hz: ", phi15Hz)

print("Amplitude 16Hz: ", Amp16Hz)
print("Fase 16Hz: ", phi16Hz)

print("Amplitude 17Hz: ", Amp17Hz)
print("Fase 17Hz: ", phi17Hz)

print("Amplitude 18Hz: ", Amp18Hz)
print("Fase 18Hz: ", phi18Hz)

print("Amplitude 19Hz: ", Amp19Hz)
print("Fase 19Hz: ", phi19Hz)

print("Amplitude 20Hz: ", Amp20Hz)
print("Fase 20Hz: ", phi20Hz)

print("Amplitude 21Hz: ", Amp21Hz)
print("Fase 21Hz: ", phi21Hz)

print("Amplitude 23Hz: ", Amp23Hz)
print("Fase 23Hz: ", phi23Hz)

print("Amplitude 27Hz: ", Amp27Hz)
print("Fase 27Hz: ", phi27Hz)

print("Amplitude 29Hz: ", Amp29Hz)
print("Fase 29Hz: ", phi29Hz)

C_signal = np.array([C1Hz, C5Hz, C9Hz, C13Hz, C15Hz, C16Hz, C17Hz, C18Hz, C19Hz, C20Hz, C21Hz, C23Hz, C27Hz, C29Hz])

A_signal = np.array([A1Hz, A5Hz, A9Hz, A13Hz, A15Hz,A16Hz,A17Hz, A18Hz, A19Hz, A20Hz, A21Hz, A23Hz, A27Hz, A29Hz])

phi = 180 - np.arcsin(C_signal[0:6]/A_signal[0:6])*180/np.pi
phi2 = np.arcsin(C_signal[6:14]/A_signal[6:14])*180/np.pi
phaset = np.append(phi,phi2)
freq = np.array([1,3,9,13,15,16,17,18,19,20,21,23,27,29])


fig, axs = plt.subplots(2)
fig.suptitle('Ganho e fase')
axs[0].plot(freq, A_signal)
plt.ylabel('Amplitude |G(omega)|', fontsize=12)
axs[1].plot(freq, phaset)
plt.xlabel('Frequência(Hz)', fontsize=12)
plt.ylabel('Fase(grau)', fontsize=12)
plt.show()

sig_freq=np.loadtxt('massa4/test1-4/mfcmassa4.txt')

freqRFviga=sig_freq[1:60,0]
signalRFviga=sig_freq[1:60,1]
phaseviga=sig_freq[1:60,2]

plt.figure()
plt.plot(freq, 20*np.log10(np.array(A_signal)/50), freqRFviga, signalRFviga)
plt.xlabel('Frequência(Hz)', fontsize=12)
plt.ylabel('Amplitude |G(omega)|', fontsize=12)
plt.show()

plt.figure()
plt.plot(freq, phaset, freqRFviga, phaseviga)
plt.xlabel('Frequência(Hz)', fontsize=12)
plt.ylabel('Fase (grau)', fontsize=12)
plt.show()