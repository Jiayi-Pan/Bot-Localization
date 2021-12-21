# %% [markdown]
# # Import

# %%
import json
from matplotlib import pyplot as plt
import numpy as np


def vis():
# %%
	with open("Data/KalmanFilteredPath.json") as fp:
		KalmanFilteredPath = json.load(fp)
	with open("Data/KalmanRealPath.json") as fp:
		KalmanRealPath = json.load(fp)
	with open("Data/KalmanSensePath.json") as fp:
		KalmanSensePath = json.load(fp)
	K = [KalmanRealPath,KalmanSensePath, KalmanFilteredPath]

	# %%
	with open("Data/ParticleFilteredPath10.json") as fp:
		ParticleFilteredPath10 = json.load(fp)
	with open("Data/ParticleRealPath10.json") as fp:
		ParticleRealPath10 = json.load(fp)
	with open("Data/ParticleSensePath10.json") as fp:
		ParticleSensePath10 = json.load(fp)
	P10 = [ParticleRealPath10, ParticleSensePath10, ParticleFilteredPath10]

	# %%
	with open("Data/ParticleFilteredPath100.json") as fp:
		ParticleFilteredPath100 = json.load(fp)
	with open("Data/ParticleRealPath100.json") as fp:
		ParticleRealPath100 = json.load(fp)
	with open("Data/ParticleSensePath100.json") as fp:
		ParticleSensePath100 = json.load(fp)
	P100 = [ParticleRealPath100, ParticleSensePath100, ParticleFilteredPath100]

	# %%
	with open("Data/ParticleFilteredPath1000.json") as fp:
		ParticleFilteredPath1000 = json.load(fp)
	with open("Data/ParticleRealPath1000.json") as fp:
		ParticleRealPath1000 = json.load(fp)
	with open("Data/ParticleSensePath1000.json") as fp:
		ParticleSensePath1000 = json.load(fp)
	P1000 = [ParticleRealPath1000, ParticleSensePath1000, ParticleFilteredPath1000]

	# %%
	with open("Data/ParticleFilteredPath10000.json") as fp:
		ParticleFilteredPath10000 = json.load(fp)
	with open("Data/ParticleRealPath10000.json") as fp:
		ParticleRealPath10000 = json.load(fp)
	with open("Data/ParticleSensePath10000.json") as fp:
		ParticleSensePath10000 = json.load(fp)
	P10000 = [ParticleRealPath10000, ParticleSensePath10000, ParticleFilteredPath10000]


	# %%
	CutStep = [0,44,75,106,129,145,177]
	CutName = ["At A","At B","At C","At D","At E","At F","At G"]

	# %% [markdown]
	MethodList = [K,P10,P100,P1000,P10000]
	NameList = ["Kalman Filter","Particle Filter + 10","Particle Filter + 100","Particle Filter + 1000","Particle Filter + 10000"]

	for i in range(len(MethodList)):
		method = MethodList[i]
		plt.plot(np.array(method[0])[:,0],np.array(method[0])[:,1])
		plt.plot(np.array(method[1])[:,0],np.array(method[1])[:,1], 'o', markersize=1)
		plt.plot(np.array(method[2])[:,0],np.array(method[2])[:,1])
		plt.legend(["Real Path", "Sensor Input", "Filtered Path"])
		print("Real Path, Sensor Path and Filtered Path for "+NameList[i])
		plt.show()
		# input("Press Enter for next data visualization")
		plt.close()

	# %%
	for i in range(len(MethodList)):
		method = MethodList[i]
		real = [np.array([pos]) for pos in method[0]]
		sensed = [np.array([pos]) for pos in method[1]]
		filtered = [np.array([pos]) for pos in method[2]]
		
		err_sen = []
		err_filt = []
		for j in range(len(real)):
			err_sen.append(np.linalg.norm(real[j]-sensed[j]))
			err_filt.append(np.linalg.norm(real[j]-filtered[j]))
		x = list(range(len(err_sen)))
		plt.plot(x,err_sen,)
		plt.plot(x,err_filt)
		for xc,c in zip(CutStep,CutName):
			plt.axvline(x=xc,linestyle='dashed',color='y')
			plt.text(xc,1.45,c)
		plt.title(NameList[i])
		plt.ylim([0,2])
		plt.legend(["Sensor Error", "Filter Error"])
		plt.xlabel("step number")
		plt.ylabel("error")
		print("Error for "+NameList[i])
		print("Sum of error: ", sum(err_filt))
		plt.show()
		# input("Press Enter for next data visualization")
		plt.close()

	# %%