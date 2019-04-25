
# coding: utf-8

# In[ ]:


import time 
import numpy as np
import pandas as pd
#import sys
import json
from pandas.io.json import json_normalize #To flatten our json data
from flatten_json import flatten
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# In[ ]:


start = time.time()


# In[ ]:


def load_data_coor(path):
    
    with open(path) as f1:
        chem_data0 = json.load(f1)
    chem_data1 = json_normalize(chem_data0,'atoms',['En','id'],record_prefix='Atoms ')
    chem_data1['Atoms xyz'] = chem_data1['Atoms xyz'].astype(str)
    chem_data1['Atoms xyz'] = chem_data1['Atoms xyz'].map(lambda x: x.lstrip('[').rstrip(']'))
    chem_data1 = chem_data1.join(chem_data1['Atoms xyz'].str.split(',', expand=True).add_prefix('Atoms Coor '))
    chem_data1 = chem_data1.drop(['Atoms xyz'],1)
    chem_data1['Atoms Coor 0'] = chem_data1['Atoms Coor 0'].astype(float)
    chem_data1['Atoms Coor 1'] = chem_data1['Atoms Coor 1'].astype(float)
    chem_data1['Atoms Coor 2'] = chem_data1['Atoms Coor 2'].astype(float)
    chem_data2 = chem_data1.values
    
    return chem_data2


# In[ ]:


def charge(col_atoms_t):
    if col_atoms_t == 'H':
        no_atom = 1
        mass = 1.0079
    elif col_atoms_t == 'He':
        no_atom = 2
        mass = 4.0026
    elif col_atoms_t == 'Li':
        no_atom = 3
        mass = 6.941
    elif col_atoms_t == 'Be':
        no_atom = 4
        mass = 9.01218
    elif col_atoms_t == 'B':
        no_atom = 5
        mass = 10.811
    elif col_atoms_t == 'C':
        no_atom = 6
        mass = 12.0107
    elif col_atoms_t == 'N':
        no_atom = 7
        mass = 14.0067
    elif col_atoms_t == 'O':
        no_atom = 8
        mass = 15.9994
    elif col_atoms_t == 'F':
        no_atom = 9
        mass = 18.9984
    elif col_atoms_t == 'Ne':
        no_atom = 10
        mass = 20.1797
    elif col_atoms_t == 'Na':
        no_atom = 11
        mass = 22.9898
    elif col_atoms_t == 'Mg':
        no_atom = 12
        mass = 24.3050
    elif col_atoms_t == 'Al':
        no_atom = 13
        mass = 29.9815
    elif col_atoms_t == 'Si':
        no_atom = 14
        mass = 28.0855
    elif col_atoms_t == 'P':
        no_atom = 15
        mass = 30.9738
    elif col_atoms_t == 'S':
        no_atom = 16
        mass = 32.065
    elif col_atoms_t == 'Cl':
        no_atom = 17
        mass = 35.453
    elif col_atoms_t == 'Ar':
        no_atom = 18
        mass = 39.948
    elif col_atoms_t == 'K':
        no_atom = 19
        mass = 39.0983
    elif col_atoms_t == 'Ca':
        no_atom = 20
        mass = 40.078
    elif col_atoms_t == 'Sc':
        no_atom = 21
        mass = 44.9559
    elif col_atoms_t == 'Ti':
        no_atom = 22
        mass = 47.867
    elif col_atoms_t == 'V':
        no_atom = 23
        mass = 50.9415
    elif col_atoms_t == 'Cr':
        no_atom = 24
        mass = 51.9961
    elif col_atoms_t == 'Mn':
        no_atom = 25
        mass = 54.9381
    elif col_atoms_t == 'Fe':
        no_atom = 26
        mass = 55.845
    elif col_atoms_t == 'Co':
        no_atom = 27
        mass = 58.933195
    elif col_atoms_t == 'Ni':
        no_atom = 28
        mass = 58.6934
    elif col_atoms_t == 'Cu':
        no_atom = 29
        mass = 63.546
    elif col_atoms_t == 'Zn':
        no_atom = 30
        mass = 65.38
    elif col_atoms_t == 'Ga':
        no_atom = 31
        mass = 4.0026
    elif col_atoms_t == 'Ge':
        no_atom = 32
        mass = 4.0026
    elif col_atoms_t == 'As':
        no_atom = 33
        mass = 4.0026
    elif col_atoms_t == 'Se':
        no_atom = 34
        mass = 78.96
    elif col_atoms_t == 'Br':
        no_atom = 35
        mass = 79.904
    elif col_atoms_t == 'Kr':
        no_atom = 36
        mass = 82.798
    elif col_atoms_t == 'Rb':
        no_atom = 37
        mass = 85.4678
    elif col_atoms_t == 'Sr':
        no_atom = 38
        mass = 87.62
    elif col_atoms_t == 'Y':
        no_atom = 39
        mass = 88.9059
    elif col_atoms_t == 'Zr':
        no_atom = 40
        mass = 91.224
    elif col_atoms_t == 'Nb':
        no_atom = 41
        mass = 92.9064
    elif col_atoms_t == 'Mo':
        no_atom = 42
        mass = 95.96
    elif col_atoms_t == 'Tc':
        no_atom = 43
        mass = 97.9072
    elif col_atoms_t == 'Ru':
        no_atom = 44
        mass = 101.07
    elif col_atoms_t == 'Rh':
        no_atom = 45
        mass = 102.9055
    elif col_atoms_t == 'Pd':
        no_atom = 46
        mass = 106.42
    elif col_atoms_t == 'Ag':
        no_atom = 47
        mass = 107.8682
    elif col_atoms_t == 'Cd':
        no_atom = 48
        mass = 112.411
    elif col_atoms_t == 'In':
        no_atom = 49
        mass = 114.818
    elif col_atoms_t == 'Sn':
        no_atom = 50
        mass = 118.710
    elif col_atoms_t == 'Sb':
        no_atom = 51
        mass = 121.760
    elif col_atoms_t == 'Te':
        no_atom = 52
        mass = 127.60
    elif col_atoms_t == 'I':
        no_atom = 53
        mass = 126.9045
    elif col_atoms_t == 'Xe':
        no_atom = 54
        mass = 131.293
    elif col_atoms_t == 'Cs':
        no_atom = 55
        mass = 132.9055
    elif col_atoms_t == 'Ba':
        no_atom = 56
        mass = 137.327
    elif col_atoms_t == 'Hf':
        no_atom = 72
        mass = 178.49
    elif col_atoms_t == 'Ta':
        no_atom = 73
        mass = 180.9479
    elif col_atoms_t == 'W':
        no_atom = 74
        mass = 183.84
    elif col_atoms_t == 'Re':
        no_atom = 75
        mass = 186.207
    elif col_atoms_t == 'Os':
        no_atom = 76
        mass = 190.23
    elif col_atoms_t == 'Ir':
        no_atom = 77
        mass = 192.217
    elif col_atoms_t == 'Pt':
        no_atom = 78
        mass = 195.084
    elif col_atoms_t == 'Au':
        no_atom = 79
        mass = 196.9666
    elif col_atoms_t == 'Hg':
        no_atom = 80
        mass = 200.59
    elif col_atoms_t == 'Tl':
        no_atom = 81
        mass = 204.3833
    elif col_atoms_t == 'Pb':
        no_atom = 82
        mass = 207.2
    elif col_atoms_t == 'Bi':
        no_atom = 83
        mass = 208.9804
    elif col_atoms_t == 'Po':
        no_atom = 84
        mass = 208.9824
    elif col_atoms_t == 'At':
        no_atom = 85
        mass = 209.9871
    elif col_atoms_t == 'Rn':
        no_atom = 86
        mass = 222.0176
    elif col_atoms_t == 'Fr':
        no_atom = 87
        mass = 223
    elif col_atoms_t == 'Ra':
        no_atom = 88
        mass = 226
    elif col_atoms_t == 'Rf':
        no_atom = 104
        mass = 261
    elif col_atoms_t == 'Db':
        no_atom = 105
        mass = 262
    elif col_atoms_t == 'Sg':
        no_atom = 106
        mass = 266
    elif col_atoms_t == 'Bh':
        no_atom = 107
        mass = 264
    elif col_atoms_t == 'Hs':
        no_atom = 108
        mass = 277
    elif col_atoms_t == 'Mt':
        no_atom = 109
        mass = 286
    elif col_atoms_t == 'Ds':
        no_atom = 110
        mass = 271
    elif col_atoms_t == 'Rg':
        no_atom = 111
        mass = 272
    elif col_atoms_t == 'Cn':
        no_atom = 112
        mass = 285
    
    return no_atom


# In[ ]:


def coloumb_entry(a,b,row,column,a_coor,b_coor):
    output = 0
    if row==column:
        output = 0.5*np.power(charge(a),2.4)
    else:
        #print(np.size(a_coor))
        dist = 0
        for c in range(3):
            dist += np.power((a_coor[c]-b_coor[c]),2)
        dist   = np.sqrt(dist)
        output = charge(a)*charge(b)/dist 
    return output


# In[ ]:


def row_final_input(molecule_data,atoms_type_data,energy):
    coloumb_input = np.zeros((1,2501))
    for m in range(50):
        for n in range(50):
            if m==n:
                if atoms_type_data[m]=='-' or atoms_type_data[n]=='-':
                    coloumb_input[0,m*50+n]=0
                else:
                    coloumb_input[0,m*50+n]=coloumb_entry(atoms_type_data[m],atoms_type_data[n],m,n,molecule_data[m,:3],molecule_data[n,:3])
                if m==49 and n==49:
                    coloumb_input[0,2500]=energy
            else:
                if atoms_type_data[m]=='-' or atoms_type_data[n]=='-':
                    coloumb_input[0,m*50+n]=0
                else:
                    coloumb_input[0,m*50+n]=coloumb_entry(atoms_type_data[m],atoms_type_data[n],m,n,molecule_data[m,:3],molecule_data[n,:3])

    return coloumb_input


# In[ ]:


def extract_molecule(data):
    
    i           = 0
    atom_number = 0
    bound       = 0
    molecule_matrix = np.zeros((50,3))
    atoms_type      = np.array(range(50),dtype=str).reshape(50,1) #Type of atoms
    final_input_data = np.zeros((1,2501))
    
    while i<len(data):
        if i!=len(data)-1:
            
            if data[i,2]==data[i+1,2]: #PubChemId
                
                atom_number += 1
                if atom_number <= 50:
                    atoms_type[i-bound]=data[i,0]
                    for j in range(3):
                        molecule_matrix[i-bound,j] = data[i,j+3]
                        
            else:
                
                atom_number += 1
                if atom_number <= 50:
                    atoms_type[i-bound]=data[i,0]
                    for j in range(3):
                        molecule_matrix[i-bound,j] = data[i,j+3]
                    
                    for k in range(i+1-bound,50):
                        atoms_type[k]='-'
                        for l in range(3):
                            molecule_matrix[k,l] = 0
                
                    energy      = data[i,1]
                    new_row     = row_final_input(molecule_matrix,atoms_type,energy)
                    final_input_data = np.concatenate([final_input_data,new_row])
       
                atom_number = 0
                bound       = i+1
                
        else:
            
            atom_number += 1
            if atom_number <= 50:
                atoms_type[i-bound]=data[i,0]
                for j in range(3):
                    molecule_matrix[i-bound,j] = data[i,j+3]
                    
                for k in range(i-bound+1,50):
                    atoms_type[k]='-'
                    for l in range(3):
                        molecule_matrix[k,l] = 0
                            
                energy  = data[i,1]
                new_row = row_final_input(molecule_matrix,atoms_type,energy) 
                final_input_data = np.concatenate([final_input_data,new_row])
                final_input_data = np.delete(final_input_data,[0],axis=0)
        i += 1
            
    return final_input_data


# In[ ]:


load_data0 = load_data_coor("pubChem_p_00000001_00025000.json")
load_data1 = load_data_coor("pubChem_p_00025001_00050000.json")
load_data2 = load_data_coor("pubChem_p_00050001_00075000.json")
load_data3 = load_data_coor("pubChem_p_00075001_00100000.json")
load_data4 = load_data_coor("pubChem_p_00100001_00125000.json")
load_data5 = load_data_coor("pubChem_p_00125001_00150000.json")
load_data6 = load_data_coor("pubChem_p_00150001_00175000.json")
load_data7 = load_data_coor("pubChem_p_00175001_00200000.json")
load_data8 = load_data_coor("pubChem_p_00200001_00225000.json")
load_data9 = load_data_coor("pubChem_p_00225001_00250000.json")


# In[ ]:


coloumb_matrix0 = extract_molecule(load_data0)
coloumb_matrix1 = extract_molecule(load_data1)
coloumb_matrix2 = extract_molecule(load_data2)
coloumb_matrix3 = extract_molecule(load_data3)
coloumb_matrix4 = extract_molecule(load_data4)
coloumb_matrix5 = extract_molecule(load_data5)
coloumb_matrix6 = extract_molecule(load_data6)
coloumb_matrix7 = extract_molecule(load_data7)
coloumb_matrix8 = extract_molecule(load_data8)
coloumb_matrix9 = extract_molecule(load_data9)


# In[ ]:


coloumb_matrix_all = np.zeros(1,2501)
coloumb_matrix_all = np.concatenate([coloumb_matrix_all,coloumb_matrix0])
coloumb_matrix_all = np.concatenate([coloumb_matrix_all,coloumb_matrix1])
coloumb_matrix_all = np.concatenate([coloumb_matrix_all,coloumb_matrix2])
coloumb_matrix_all = np.concatenate([coloumb_matrix_all,coloumb_matrix3])
coloumb_matrix_all = np.concatenate([coloumb_matrix_all,coloumb_matrix4])
coloumb_matrix_all = np.concatenate([coloumb_matrix_all,coloumb_matrix5])
coloumb_matrix_all = np.concatenate([coloumb_matrix_all,coloumb_matrix6])
coloumb_matrix_all = np.concatenate([coloumb_matrix_all,coloumb_matrix7])
coloumb_matrix_all = np.concatenate([coloumb_matrix_all,coloumb_matrix8])
coloumb_matrix_all = np.concatenate([coloumb_matrix_all,coloumb_matrix9])
coloumb_matrix_all = np.delete(coloumb_matrix_all,[0],axis=0)


# In[ ]:


np.savetxt("Thesis_data_training",coloumb_matrix_all,delimiter=",")


# In[ ]:


end = time.time


# In[ ]:


with open('coloumb_comp_time.txt','w') as file5:
    print(end-start,file=file5)

