import Bit_Functions as bf
import numpy as np

number_of_electrons = 4
number_of_states = 2**number_of_electrons

# S^z_i|state>
def Sz(state,position):
    return bf.ReadBit(state,position) - 0.5

# <state|S^z|state>
def diagonal(state):
    elem = 0
    for i in range(number_of_electrons):
        elem += Sz(state,i) * Sz(state,(i+1)%number_of_electrons)
    return elem

# Get proper row_state which obtain none-zero <row_state|S^xy|col_state>
# The proper row_state can never be |0>, but a none-zero state
def row_of_col(col_state,position):
    if bf.ReadBit(col_state,position) == bf.ReadBit(col_state,(position+1)%number_of_electrons):
        return 0
    row_state = bf.FlipBit(col_state,position)
    row_state = bf.FlipBit(row_state,(position+1)%number_of_electrons)
    return row_state

# Write a numpy array in file
def write_array(file_name,matrix):
    for row in matrix:
        for item in row:
            file_name.write('{:.1f}'.format(item)+' ')
        file_name.write('\n')