from rdkit import RDLogger
from rdkit import Chem
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.MolStandardize.fragment import is_organic
from rdkit.Chem.rdmolops import GetFormalCharge
import numpy as np

def neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            if atom.GetSymbol()=="B":
                continue
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol

def MolWithoutIsotopes(mol):
    atoms = [atom for atom in mol.GetAtoms() if atom.GetIsotope()]
    for atom in atoms:
    # restore original isotope values
        atom.SetIsotope(0)
    return mol

def RemoveStereoFromSmiles(s, chars=["@","/","\\"]):
    for c in chars:
        s=s.replace(c,"")
    return s

_MD = rdMolStandardize.MetalDisconnector()
_LFC = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_TE= rdMolStandardize.TautomerEnumerator()
_funcs=[_MD.Disconnect, _LFC.choose, neutralize_atoms, MolWithoutIsotopes, _TE.Canonicalize,
         Chem.MolToSmiles, RemoveStereoFromSmiles, Chem.CanonSmiles]

def ElabSmiles(sm, default=np.nan):
    mol=Chem.MolFromSmiles(sm)
    if not mol:
        return default

    if "." in sm:
        splitted=sm.split(".")
        if all(GetFormalCharge( Chem.MolFromSmiles(x) )==0 for x in splitted)\
        and all(Chem.MolFromSmiles(x).GetNumAtoms()>1 for x in splitted):
            return "mixture"
    
    for f in _funcs:
        mol=f(mol)
    
    if not is_organic(Chem.MolFromSmiles(mol)):
        return "inorganic"
    
    return mol