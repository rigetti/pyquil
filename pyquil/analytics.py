#!/usr/bin/python
##############################################################################
# Copyright 2016-2017 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################
"""
A collection of methods for analyzing pyquil programs (gate count, etc...)
"""

import pyquil.gates as gates
from grove.qft.fourier import qft
from grove.unitary_circuit.arbitrary_state import create_arbitrary_state
import numpy as np
from pyquil.quil import Program
from grove.teleport.teleportation import make_bell_pair
import plotly
import plotly.graph_objs as go
from copy import deepcopy
import time
import random

def get_total_instruction_count(program):
    '''
    Returns the total number of instructions within the given program
    :param program: The program to analyze
    :return: The max number of instructions executed by this program
    '''
    
    return len(program.actions)

def get_qubit_count(program):
    '''
    Returns the total number of qubits needed for this program
    :param program: The program to analyze
    :return: The total number of qubits allocated for this program
    '''
    return len(program.get_qubits())

def get_gate_types(program):
    '''
    Returns a dictionary of gate types to their counts within the program
    :param program: The program to analyze
    :return: The mapping of gates to their counts within program
    '''
    counts = {}
    for gate in get_all_gate_types():
        counts[gate] = 0
    
    # Count how many gates there are for each gate type
    for action in program.actions:
        counts[action[1].operator_name] += 1
        
    return counts

def get_qubit_num_gates(gate_counts):
    '''
    Returns a dictionary with the original gate_counts, but with
    extra entries 'Single Qubit Gates', 'Two Qubit Gates',
    'Control Instructions', 'Classical Instructions',
    'Three+ Qubit Gates', and 'Elements in DEFGATES'
    '''
    
    results = deepcopy(gate_counts)
    results['Single Qubit Gates'] = \
                gate_counts['I'] + \
                gate_counts['H'] + \
                gate_counts['X'] + \
                gate_counts['Y'] + \
                gate_counts['Z'] + \
                gate_counts['S'] + \
                gate_counts['T'] + \
                gate_counts['PHASE']
    results['Two Qubit Gates'] = \
                gate_counts['RX'] + \
                gate_counts['RY'] + \
                gate_counts['RZ'] + \
                gate_counts['CNOT'] + \
                gate_counts['CPHASE00'] + \
                gate_counts['CPHASE01'] + \
                gate_counts['CPHASE10'] + \
                gate_counts['CPHASE'] + \
                gate_counts['SWAP'] + \
                gate_counts['ISWAP'] + \
                gate_counts['PSWAP']
    results['Control Instructions'] = \
                gate_counts['RESET'] + \
                gate_counts['NOP'] + \
                gate_counts['WAIT'] + \
                gate_counts['HALT']
    results['Classical Instructions'] = \
                gate_counts['TRUE'] + \
                gate_counts['FALSE'] + \
                gate_counts['NOT'] + \
                gate_counts['AND'] + \
                gate_counts['OR'] + \
                gate_counts['MOVE'] + \
                gate_counts['EXCHANGE'] + \
                gate_counts['JUMP-WHEN'] + \
                gate_counts['JUMP-UNLESS']
    results['Three+ Qubit Gates'] = \
                gate_counts['CCNOT'] + \
                gate_counts['CSWAP']
    results['Elements in DEFGATES'] = \
                gate_counts['DEFGATE']
    return results

def analyze(program_creator, input_list, should_range=False, to_include=None, remove_zeros=False):
    '''
    Analyzes a program by compiling properties of the pyQuil program for different inputs. Provides
    a wide range of configurations to assist in analyzing specific programs
    :param program_creator: A method to accept program parameters and create the Quil program (i.e
                            a method such as qft from grove.fourier).
    :param input_list: A list of inputs to be inputted into the program creator. These may be tuples
                       of parameters, or something as simple as a list [0, 1, 2 ...] indicating the
                       number of qubits to use in a program.
    :param should_range: (Optional) If true, takes each value in n_list, assumes that it is an integer,
                         and replaces it with range(that integer). This is useful in situations such as
                         using 1, 2, 3, ... qubits without indicating the range of indices for these
                         qubits. Default is False.
    :param to_include: (Optional) A list of properties to analyze with your program creator. Default is
                       to include every parameter.
    :param remove_zeros: (Optional) If set to True, any properties where the value is 0 for all inputs
                         of the program are left out of the final result. Default is False.
    :return: A tuple containing multiple analysis results:
                0) The input list used to generate each program
                1) A structure of arrays, with indices corresponding to the index of the input in input_list,
                and keys representing the properties measured for those program compilations
                2) A dictionary from properties to possible complexities for that property (wrt input)
    '''

    results = []
    for val in input_list:
        # Space efficient option to pass in sizes of a list
        # rather than a list of increasing lists. This is helpful
        # for items such as qubit ranges
        if should_range:
            val = range(val)
        
        start_time = time.time() / 1000.0
        program = program_creator(val)
        properties = get_qubit_num_gates(get_gate_types(program))
        properties["Total Instructions"] = get_total_instruction_count(program)
        properties["Total Qubits"] = get_qubit_count(program)
        properties["Compile Time"] = time.time() / 1000.0 - start_time
        results.append(properties)
        
    results = aos_to_soa(results)
    
    # Filters results if specific ones are requested
    if to_include != None:
        filtered = {}
        for val in to_include:
            if val in results:
                filtered[val] = results[val]
        results = filtered
    
    # Removes properties that are always zero, if requested
    if remove_zeros:
        for key in results.keys():
            if sum(results[key]) == 0:
                del results[key]
    
    return (input_list, results)

def plot_analysis(x_points, analysis, title, x_axis, y_axis, bounds=None):
    '''
    Takes results from the analyze function and plots them using plotly
    :param analysis: The analysis gathered by the analyze function
    :param title: The title to display on the plotted graph
    :param x_axis: A label for the x axis
    :param y_axis: A label for the y axis
    :param bounds: An optional dictionary of names to functions, which will
                   be plotted against the given x points. This is useful
                   for comparing your gate growth to the requested growth
    :param x_points: A list of x points to use for your data (i.e. input
                     for your programs, "n")
    '''
    runtime = {}
    data = []
    
    r = lambda: random.randint(0,255)
    
    for val in analysis.keys():   # For each property
        
        # Create the graph
        this_data = analysis[val]
        trace = go.Scatter(
            x = x_points,
            y = this_data,
            name = val,
            line = dict(
                color = ('rgb(%d, %d, %d)' % (r(), r(), r())),
                width = 4)
        )
        data.append(trace)
        
        # Find the polynomial runtime
        runtime[val] = get_runtime_approximate(x_points, this_data)
        #print str(val) + ": " + str(runtime[val][0]) + "n^" + str(runtime[val][1])
        
    if bounds is not None:
        for name in bounds.keys():
            # Create the graph
            function = bounds[name]
            this_data = [function(i) for i in x_points]
            trace = go.Scatter(
                x = x_points,
                y = this_data,
                name = name,
                line = dict(
                    color = ('rgb(%d, %d, %d)' % (r(), r(), r())),
                    width = 4,
                    dash = 'dash')
            )
            data.append(trace)
        

    # Edit the layout
    layout = dict(title = title,
                  xaxis = dict(title = x_axis),
                  yaxis = dict(title = y_axis),
                  )

    fig = dict(data=data, layout=layout)
    plotly.offline.plot(fig)

def aos_to_soa(aos):
        '''
        Converts an Array of Structures into a Structure of Arrays. Not robust yet, as
        the keys to be used in the structure are determined from the first structure in
        the input Array of Structure
        :param aos: An Array of Structure to convert into a Structure of Arrays
        '''
        
        result = {}
        for key in aos[0]:
            result[key] = []
            
        for test in aos:
            for item in test.keys():
                val = test[item]
                result[item].append(val)
                
        return result

def get_runtime_approximate(x, y):
    '''
    Given x and y points, determines an approximate running
    time for the given points using fitting
    '''
    
    # Check if this is polynomial
    current_max = 0
    poly_result = np.polyfit(x, y, 10)
    degree = 10
    polynomial_degree = (None, None)
    while degree > 0:
        degree -= 1
        coeff = poly_result[degree]
        if coeff > current_max:
            current_max = coeff
            polynomial_degree = (coeff, 10 - degree)
            
    return polynomial_degree
        
    # Check if this is exponential
    
    # Check if this is logarithmic
    
    
def get_all_gate_types():
    '''
    Simply returns a set of all possible Quil gate types
    '''
    return gates.STANDARD_GATES.keys() + \
            ['RESET', 'NOP', 'WAIT', 'HALT', 'TRUE', \
             'FALSE', 'NOT', 'AND', 'OR', 'WAIT', 'MOVE', \
             'EXCHANGE', 'JUMP-WHEN', 'JUMP-UNLESS', 'DEFGATE']
    
    
if __name__ == "__main__":
    
    tests = range(1, 150)
    results = analyze(qft, tests, should_range=True, remove_zeros=True)
    estimate = {"Total Gate Bound": lambda n: n*np.log2(n)**2} # From http://algassert.com/2016/06/14/qft-by-multiply.html
    plot_analysis(results[0], results[1], 
                  "QFT Implementation", 
                  "Number of Qubits Operated On", 
                  "Num Gates / Runtime (us)",
                  bounds=estimate)
    
    