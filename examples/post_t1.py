from pyquil import api
qpu = api.QPUConnection("Z12-13-C4a2")
res = qpu.t1(qubit_id=3, start=0.01, stop=60.0, num_pts=31)
print res.success, res.result
