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
import distutils.spawn

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

    # clean SDF file for rdkit
    os.system('antechamber -i UNL.mol2 -fi mol2 -o UNL.sdf -fo sdf')
    
    with open('convert.leap','w') as f:
        f.write('source leaprc.%s\n' % (ff))
        f.write('UNL=loadmol2 UNL.mol2\n')
        f.write('saveoff UNL UNL.off\n')
        f.write('quit')

    os.system('tleap -f convert.leap>out')

class Amber_atom(object):
    def __init__(self,name,element,atom_type,atom_charge,hybrid=None,bond_count=None,x=None,y=None,z=None,core=None):
        self.name=name
        self.element=element
        self.atom_type=atom_type
        self.atom_charge=atom_charge

        self.hybrid=hybrid
        self.bond_count=bond_count
        self.x=x
        self.y=y
        self.z=z
        self.core=core

def read_off(mol,off_file):
    off_obj=[]
    start=0
    with open(off_file,'r') as f:
        for line in f:
            start+=1
            if '!entry' in line:
                break

    counter=0
    with open(off_file,'r') as f:
        for line in f:
            if (start-1<counter<len(mol.GetAtoms())+start):
                name=line.split()[0].strip('"')
                element=mol.GetAtomWithIdx(counter-start).GetAtomicNum()
                atom_type=line.split()[1].strip('"')
                atom_charge=float(line.split()[7])
                hybridization=mol.GetAtomWithIdx(counter-start).GetHybridization()
                bond_count=int(len(mol.GetAtomWithIdx(counter-start).GetBonds()))
                x=float(mol.GetConformer().GetAtomPosition(counter-start).x)
                y=float(mol.GetConformer().GetAtomPosition(counter-start).y)
                z=float(mol.GetConformer().GetAtomPosition(counter-start).z)

                off_obj.append(Amber_atom(name,element,atom_type,atom_charge,hybridization,bond_count,x,y,z,False))
            counter+=1
    return off_obj

def compare_atom(atm1,atm2,tol=0.1):
    if (atm1.element==atm2.element) and (atm1.atom_type==atm2.atom_type) and (atm1.hybrid==atm2.hybrid) and (atm1.bond_count==atm2.bond_count) and abs(atm1.x-atm2.x)<tol and abs(atm1.y-atm2.y)<tol and abs(atm1.z-atm2.z)<tol:
        return True
    else:
        return False

def compare_mols(mol_off1,mol_off2):
    match=[]
    for i in range(0,len(mol_off2)):
        for j in range(0,len(mol_off1)):
            if compare_atom(mol_off1[j],mol_off2[i]):
                match.append(j)
    return match

def update_ti_atoms(mol_list,off_list):
    assert len(mol_list)==2
    assert len(off_list)==2

    periodic={'6':'C','1':'H','8':'O','7':'N','17':'Cl','9':'F','16':'S','35':'Br','15':'P','53':'I'}

    matches=compare_mols(off_list[0],off_list[1])

    MCS_atoms_amber=[]
    for i in matches:
        MCS_atoms_amber.append(off_list[0][i])

    out_mols=[]
    out_off=[]
    for mol,mol_amber in zip(mol_list,off_list):
        ele_count=dict([(6,1),(1,1),(8,1),(7,1),(17,1),(9,1),(16,1),(35,1),(15,1),(53,1)])

        write_core=[]
        write_last=[]

        mol_copy=Chem.Mol(mol)

        for i in range(0,len(MCS_atoms_amber)):
            for j in range(0,len(mol.GetAtoms())):
                if compare_atom(MCS_atoms_amber[i],mol_amber[j]) and j not in write_core:
                    write_core.append(j)

        for i in range(0,len(mol.GetAtoms())):
            if i not in write_core:
                write_last.append(i)

        for i in range(0,len(mol.GetAtoms())):
            if i in write_core:
                mol_amber[i].core=True
            elif i in write_last:
                mol_amber[i].core=False

        for i in write_core:
            new_atom_name=periodic[str(mol_amber[i].element)]+str(ele_count[int(mol_amber[i].element)])
            mol_amber[i].name=new_atom_name
            ele_count[int(mol_amber[i].element)]+=1

        for i in range(0,len(mol.GetAtoms())):
            if mol_amber[i].core==False:
                new_atom_name=periodic[str(mol_amber[i].element)]+str(ele_count[int(mol_amber[i].element)])
                mol_amber[i].name=new_atom_name
                ele_count[int(mol_amber[i].element)]+=1

        # return a re-ordered mol
        mol_copy=rdmolops.RenumberAtoms(mol_copy,write_core+write_last)
        out_mols.append(mol_copy)

        # return matchin re-ordered amber off
        mol_amber = [mol_amber[i] for i in write_core+write_last]
        out_off.append(mol_amber)

    return out_mols,out_off

def write_amber_off(mol,mol_amber,output_file,resi):
    with open('make_off.leap','w') as f:
        f.write('%s=loadpdb %s.pdb\n' % (resi,resi))

        for i in range(0,len(mol.GetAtoms())):
            f.write('set %s.1.%s type %s\n' % (resi,mol_amber[i].name,mol_amber[i].atom_type))

        for i in range(0,len(mol.GetAtoms())):
            f.write('set %s.1.%s charge %lf\n' % (resi,mol_amber[i].name,mol_amber[i].atom_charge))

        # bonds
        bond_list=[]
        for atom in mol.GetAtoms():
            for bond in atom.GetBonds():
                a1=bond.GetBeginAtomIdx()
                b1=bond.GetEndAtomIdx()

                if [a1,b1] not in bond_list and [b1,a1] not in bond_list:
                    bond_list.append([a1,b1])

        for val in bond_list:
            f.write('bond %s.1.%s %s.1.%s\n' % (resi,mol_amber[val[0]].name,resi,mol_amber[val[1]].name))

        f.write('saveoff %s %s\n' % (resi,output_file))
        f.write('quit')

    os.system('tleap -f make_off.leap>out')

def write_pdb_file(mol,mol_amber,output_file,resi):

    counter=0
    for atom in mol.GetAtoms():
        mi = Chem.AtomPDBResidueInfo()
        mi.SetName(mol_amber[counter].name)
        mi.SetResidueName(''.ljust(4-len(mol_amber[counter].name))+resi)
        mi.SetResidueNumber(1)
        mi.SetIsHeteroAtom(False)
        atom.SetMonomerInfo(mi)

        counter+=1

    Chem.MolToPDBFile(mol,output_file,flavor=2)

    # CONECT records break leap
    # a cleaner way would be to take the new pdb file 
    # and just write the first mol.GetAtoms() lines
    os.system('sed -i -e \'/CONECT/d\' %s' % (output_file))

def write_ti_strings(off_list,output_file):
    ti_region1=[]
    for atom in off_list[0]:
        if not atom.core:
            ti_region1.append(atom.name)

    ti_str1=''
    for at in ti_region1:
        ti_str1=ti_str1+str(at)+','

    ti_region2=[]
    for atom in off_list[1]:
        if not atom.core:
            ti_region2.append(atom.name)

    ti_str2=''
    for at in ti_region2:
        ti_str2=ti_str2+str(at)+','

    with open(output_file,'w') as f:
        f.write('%s\n' % (ti_str1))
        f.write('%s\n' % (ti_str2))

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

        if not distutils.spawn.find_executable('antechamber'):
            print('Error: cannot find antechamer.\n')
            sys.exit()

    ## Proceed with setup ##
        print('\nSetup: writing transformation directories.\n')
   
        dir_1_name='start'
        dir_2_name='endpoint'

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

            os.mkdir(dir_1_name)
            os.mkdir(dir_2_name)

    ## Write start ligand file, parameters ##
            os.chdir(dir_1_name)
            writer=SDWriter('for_parm.sdf')
            writer.write(ligands[ligands_name.index(pair[0])])
            writer.flush()
            run_antechamber(ligands[ligands_name.index(pair[0])],'for_parm.sdf',ff)
            os.chdir('../')

    ## Check and fix XYZ coords of transform ligand ##
            fix_mol=update_atom_position(ligands[ligands_name.index(pair[0])],ligands[ligands_name.index(pair[1])])

    ## Write endpoint ligand file, parameters ##
            os.chdir(dir_2_name)
            writer=SDWriter('for_parm.sdf')
            writer.write(fix_mol)
            writer.flush()
            run_antechamber(ligands[ligands_name.index(pair[1])],'for_parm.sdf',ff)
            os.chdir('../')

    ## Now rename and re-order start and endpoint ligand atoms so that TI region is at the end
            parm_mols=[]
            parm_off=[]
            for parm_dir in [dir_1_name,dir_2_name]:
                mol=Chem.SDMolSupplier(parm_dir+'/UNL.sdf',removeHs=False)[0]
                parm_mols.append(mol)
                parm_off.append(read_off(mol,parm_dir+'/UNL.off'))

            # pass a new copy of the off objects since they get modified
            # return re-ordered [mol1,mol2] and [off1,off2]
            refit_mols,refit_offs=update_ti_atoms(parm_mols,list(parm_off))

            os.chdir(dir_1_name)
            write_pdb_file(refit_mols[0],refit_offs[0],'LIG.pdb','LIG')
            write_amber_off(refit_mols[0],refit_offs[0],'LIG.off','LIG')
            os.chdir('../')

            os.chdir(dir_2_name)
            write_pdb_file(refit_mols[1],refit_offs[1],'MOD.pdb','MOD')
            write_amber_off(refit_mols[1],refit_offs[1],'MOD.off','MOD')
            os.chdir('../')

            write_ti_strings(refit_offs,'TI_MASKS.dat')

    ## Exit pair directory
            os.chdir('../')

    print('Setup complete.\n')

###############################################################################

