import Bit_Functions as bf
import numpy as np

from scipy.special import comb
from scipy.linalg  import block_diag

from Reuse_Functions import diagonal
from Reuse_Functions import row_of_col
from Reuse_Functions import write_array

number_of_electrons = 4
number_of_states = 2**number_of_electrons

# At first, we build the Lin table
Lin_a = np.array([[0,0,0,0],[1,1,2,0],[1,1,2,1],[0,1,2,1],[0,0,0,0]])
Lin_b_Block = np.array([[1,0,0,0],[0,2,3,0],[0,1,3,5],[0,0,1,2],[0,0,0,1]])

def Up_spin(state):
    return bf.PopCntBit(state)

def index_of_state_Block(state):
    up_spin = Up_spin(state)
    part_a = bf.PickBit(state,int(number_of_electrons/2),int(number_of_electrons/2))
    part_b = bf.PickBit(state,0,int(number_of_electrons/2))
    return Lin_a[up_spin][part_a] + Lin_b_Block[up_spin][part_b] - 1

# Below is the lookup method from index to state, which is unnecessary in this work
'''
state_from_index = np.zeros(number_of_states)
for state in range(number_of_states):
    state_from_index[index_of_state(state)] = state
def state_of_index(index):
    return state_from_index[index]
'''

# It is more important to calculate the eigenvalue and eigenvector of full H matrix
# Thus, we could diagonalize each diagonal block matrix separately

# Initialize each block  
block_matrix = []
for i in range(number_of_electrons+1):
    block_matrix.append(np.eye(int(comb(number_of_electrons,i))))

# Build each block
for col_state in range(number_of_states):
    for position in range(number_of_electrons):
        row_state = row_of_col(col_state,position)
        if row_state:
            block_matrix[Up_spin(col_state)][index_of_state_Block(row_state)][index_of_state_Block(col_state)] = 0.5
for state in range(number_of_states):
    block_matrix[Up_spin(state)][index_of_state_Block(state)][index_of_state_Block(state)] = diagonal(state)

# Eigenvalue of each block
evalues = []
for block in block_matrix:
    evalue,evector = np.linalg.eig(block)
    for e in evalue:
        evalues.append(e)

# We can also obtain full matrix of the Hamiltonian
H = block_diag(*block_matrix)

# Save our results in file
with open('Result_U1_Symmetry.txt','w') as f:
    f.write('Matrix of Hamiltonian:\n')
    write_array(f,H)
    f.write('\nBlock Matrix of Hamiltonian:\n')
    for i in range(number_of_electrons+1):
        f.write('Up_Spin = {}:\n'.format(i))
        write_array(f,block_matrix[i])
        f.write('\n')
    f.write('Eigenvalue:\n')
    for i in evalues:
        f.write('{:.1f}'.format(i)+' ')
