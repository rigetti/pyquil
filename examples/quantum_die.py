# ----------------------------
# Quantum Die
# ----------------------------
# Emily Stamm
# Dec 30, 2018

from pyquil import *
import numpy as np
from pyquil.quil import *

def throw_polyhedral_die_helper(p,qc, num_qubits):
	result_dict = qc.run_and_measure(p, trials = 1)
	result = 1
	for i in range(num_qubits):
		result += (2**i)*result_dict[i][0] 
	return result

# throw_polyhedral_die(num_sides)
# ----------------------------
# return the result of throwing a num_sides sided die by running a quantum program

def throw_polyhedral_die(num_sides):
	# Number of qubits needed for computation
	num_qubits = int(np.ceil(np.log2(num_sides)))
	# Initialize program
	p  = Program()
	p.inst(H(i) for i in range(num_qubits)) 


	qc_string = str(num_qubits) + 'q-qvm'
	qc = get_qc(qc_string)

	result = num_sides + 1
	# Throw dice until get a number <= number of sides of dice
	# Note- will get anywhere from 1 to ceiling of log_2(num_sides) 
	#  >= log_2(num_sides)
	while result > num_sides:
		result = throw_polyhedral_die_helper(p,qc,num_qubits)
	return result

def input_for_die():
    number_of_sides = int(input("Please enter number of sides: "))
    print("The result is: ", throw_polyhedral_die(number_of_sides))

input_for_die()