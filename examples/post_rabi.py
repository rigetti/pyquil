from pyquil import api
qpu = api.QPUConnection("Z12-13-C4a2")
res = qpu.rabi(qubit_id=3, start=0.01, stop=0.33, step=0.03, the_time=160)
print(res.success, res.result)
