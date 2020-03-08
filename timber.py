#!/usr/bin/env python

#
# TIMBER
# Callum J Dickson
# 7 March 2020
#

import argparse
import os
import sys
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import SDWriter
from rdkit.Chem import ChemicalForceFields
from rdkit.Chem import BRICS
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdmolops

###############################################################################

def check_file(file_in):
    output=False
    if os.path.exists(file_in) and os.path.getsize(file_in)>0:
        output=True
    return output

def mapping_tuples(csv_file):
    map_list=[]

    # assumes first line are comments
    with open(csv_file,'r') as f:
        f.readline()
        for line in f:
            lig1=line.split(',')[0].strip()
            lig2=line.split(',')[1].strip()
            map_list.append((lig1,lig2))
            
    return map_list

def update_atom_position(mol1,mol2):
    mol_copy=Chem.Mol(mol2)

    # This is a work-around to get a seedSmarts for the FMCS algorithm
    # and prevent the occassional hanging of FMCS
    # Might be unnecessary with future versions of rdkit
    core_frags=BRICS.BRICSDecompose(Chem.RemoveHs(mol1))
    frag_smarts=[]
    for frag in enumerate(core_frags):
        smi_str=(re.sub('[[1-9][0-9]{0,2}\*]','[*]',frag[1]))
        frag_smarts.append(Chem.MolToSmarts(Chem.MolFromSmiles(smi_str)).replace(':','~').replace('-','~').replace('=','~').replace('#0','*'))

    seed=None
    for query in frag_smarts:
        if mol_copy.HasSubstructMatch(Chem.MolFromSmarts(query)):
            seed=query
            break

    # Now get MCSS
    res=rdFMCS.FindMCS([mol1,mol_copy],seedSmarts=seed)
    mcs_q=Chem.MolFromSmarts(res.smartsString)

    # Get atom IDs
    template=mol1.GetSubstructMatches(mcs_q)[0]
    hit_atom=mol_copy.GetSubstructMatches(mcs_q)[0]

    # Update XYZ coords of MCSS
    running_distance=0
    for i in range(0,len(template)):
        origin=mol1.GetConformer().GetAtomPosition(template[i])
        pos=mol_copy.GetConformer().GetAtomPosition(hit_atom[i])

        p1=np.array([origin.x,origin.y,origin.z])
        p2=np.array([pos.x,pos.y,pos.z])

        sq_dist=np.sum((p1-p2)**2,axis=0) 
        dist=np.sqrt(sq_dist)

        running_distance+=dist

        mol_copy.GetConformer().SetAtomPosition(hit_atom[i],(origin.x,origin.y,origin.z))

    if running_distance>0.1:
        # relax atoms outside MCSS
        res_atom=[]
        for atom in mol_copy.GetAtoms():
            if atom.GetIdx() not in hit_atom:
                res_atom.append(atom.GetIdx())

        # do minimization
        mp=ChemicalForceFields.MMFFGetMoleculeProperties(mol_copy)
        ff=ChemicalForceFields.MMFFGetMoleculeForceField(mol_copy,mp)

        for val in hit_atom:
            ff.AddFixedPoint(val)
        for val in res_atom:
            ff.MMFFAddPositionConstraint(val,1,5)

        ff.Minimize()

    return mol_copy

def run_antechamber(mol,sdf_file,ff):
    net_charge=int(rdmolops.GetFormalCharge(mol))
    
    os.system('antechamber -i %s -fi sdf -o UNL.mol2 -fo mol2 -rn UNL -nc %d -at %s -c bcc -s 0 -pf y' % (sdf_file,net_charge,ff))

    os.system('parmchk -i UNL.mol2 -f mol2 -o missing_gaff.frcmod -at %s' % (ff))

    with open('convert.leap','w') as f:
        f.write('source leaprc.%s\n' % (ff))
        f.write('UNL=loadmol2 UNL.mol2\n')
        f.write('saveoff UNL UNL.off\n')
        f.write('quit')

    os.system('tleap -f convert.leap>out')

###############################################################################

if __name__=='__main__':
    ## Command line arguments ##
    parser = argparse.ArgumentParser(description='TIMER code for AMBER TI setup\n')
    parser.add_argument('-i',help='CSV file with ligand mappings',required=True)
    parser.add_argument('-sdf',help='SDF file with ligands',required=False)
    parser.add_argument('-ff',help='Force field',choices=['gaff','gaff2'],required=False)
    parser.add_argument('-m',help='Mode',choices=['setup'],required=True)

    args=vars(parser.parse_args())

###############################################################################
## MODE: SETUP ##
###############################################################################

    ## Check files ##
    if args['m']=='setup':
        if args['i']!=None and check_file(args['i']):
            map_file=args['i']
        else:
            print('Error: CSV file required.\n')
            sys.exit()

        if args['sdf']!=None and check_file(args['sdf']):
            ligand_sdf=args['sdf']
        else:
            print('Error: setup mode requires input SDF ligand file.\n')
            sys.exit()

        if args['ff']!=None:
            ff=args['ff']
        else:
            print('Error: setup mode required force field specification.\n')
            sys.exit()

    ## Get a list of the mapping tuples ##
        map_list=mapping_tuples(map_file)

    ## Load the ligand file and save names ##
        ligands=Chem.SDMolSupplier(ligand_sdf,removeHs=False)
        ligands_name=[]
        for mol in ligands:
            ligands_name.append(mol.GetProp('_Name'))

    ## Make directories for each transformation ##
        for pair in map_list:
            print('%s -> %s \n' % (pair[0],pair[1]))

            pair_dir=pair[0]+'_'+pair[1]
            os.mkdir(pair_dir)
            os.chdir(pair_dir)

            os.mkdir('start')
            os.mkdir('endpoint')

    ## Write start ligand file, parameters ##
            os.chdir('start')
            writer=SDWriter('for_parm.sdf')
            writer.write(ligands[ligands_name.index(pair[0])])
            writer.flush()
            run_antechamber(ligands[ligands_name.index(pair[0])],'for_parm.sdf',ff)
            os.chdir('../')

    ## Check and fix XYZ coords of transform ligand ##
            fix_mol=update_atom_position(ligands[ligands_name.index(pair[0])],ligands[ligands_name.index(pair[1])])

    ## Write endpoint ligand file, parameters ##
            os.chdir('endpoint')
            writer=SDWriter('for_parm.sdf')
            writer.write(fix_mol)
            writer.flush()
            run_antechamber(ligands[ligands_name.index(pair[1])],'for_parm.sdf',ff)
            os.chdir('../')

    ## Now rename and re-order start and endpoint ligand atoms so that TI region is at the end

    ## Exit pair directory
            os.chdir('../')

###############################################################################

