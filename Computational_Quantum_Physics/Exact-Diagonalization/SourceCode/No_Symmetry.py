import Bit_Functions as bf
import numpy as np

from Reuse_Functions import diagonal
from Reuse_Functions import row_of_col
from Reuse_Functions import write_array

number_of_electrons = 4
number_of_states = 2**number_of_electrons

# Now we can obtain the matrix of H
# However, below is a trivial method and we abandon it
'''
# <row|S^x,y_i,i+1|col>
def Sxy(row_state,col_state,position):
    if bf.ReadBit(col_state,position) == bf.ReadBit(col_state,(position+1)%number_of_electrons):
        return 0
    Sxy_col = bf.FlipBit(col_state,position)
    Sxy_col = bf.FlipBit(Sxy_col,(position+1)%number_of_electrons)
    if Sxy_col != row_state:
        return 0
    else:
        return 0.5

# <row|S^xy|col>
def off_diagonal(row_state,col_state):
    elem = 0
    for i in range(number_of_electrons):
        elem += Sxy(row_state,col_state,i)
    return elem

H = np.eye(number_of_states)
for i in range(number_of_states):
    for j in range(i):
        H[i][j] = H[j][i] = off_diagonal(i,j)
for i in range(number_of_states):
    H[i][i] = diagonal(i)
print(H)
'''

# A more efficient way to calculate the none-zero off-diagonal elements
# It can be shown that none-zero off-diagonal elements must equal to 0.5
H = np.eye(number_of_states)
for col_state in range(number_of_states):
    for position in range(number_of_electrons):
        row_state = row_of_col(col_state,position)
        if row_state:
            H[row_state][col_state] = 0.5
for state in range(number_of_states):
    H[state][state] = diagonal(state)

# Eigenvalue of Hamiltonian matrix
evalue,evector = np.linalg.eig(H)

# Save our results in file
with open('Result_No_Symmetry.txt','w') as f:
    f.write('Matrix of Hamiltonian:\n')
    write_array(f,H)
    f.write('Eigenvalue:\n')
    for i in evalue:
        f.write('{:.1f}'.format(i)+' ')