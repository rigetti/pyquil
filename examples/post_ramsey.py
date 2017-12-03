from pyquil import api
qpu = api.QPUConnection("Z12-13-C4a2")
res = qpu.ramsey(qubit_id=3, start=0.01, stop=10, step=0.2, detuning=0.5)
print(res.success, res.result)
