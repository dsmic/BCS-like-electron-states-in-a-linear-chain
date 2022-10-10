# pylint: disable=W,C enable=W0101,W0102,W0103,W0104,W0105,W0106,W0107,W0108,W0109,W0110,W0111,W0112,W0113,W0114,W0115,W0116,W0117,W0118,W0119,W0120,W0121,W0122,W0123,W0124,W0125,W0126,W0127,W0128,W0129,W0130,W0131,W0132,W0133,W0134,W0135,W0136,W0137,W0138,W0139,W0140,W0141,W0142,W0143,W0144,W0145,W0146,W0147,W0148,W0149,W0150,W0151,W0152,W0153,W0154,W0155,W0156,W0157,W0158,W0159,W0160,W0161,W0162,W0163,W0164,W0165,W0166,W0167,W0168,W0169,W0170,W0171,W0172,W0173,W0174,W0175,W0176,W0177,W0178,W0179,W0180,W0181,W0182,W0183,W0184,W0185,W0186,W0187,W0188,W0189,W0190,W0191,W0192,W0193,W0194,W0195,W0196,W0197,W0198,W0199,W0200,W0201,W0202,W0203,W0204,W0205,W0206,W0207,W0208,W0209,W0210,W0211,W0212,W0213,W0214,W0215,W0216,W0217,W0218,W0219,W0220,W0221,W0222,W0223,W0224,W0225,W0226,W0227,W0228,W0229,W0230,W0231
# catching very dangourus errors

import os, sys, signal
import multiprocessing
import multiprocess

def signal_handler(sig, frame):
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

#os.environ['NUMBA_OPT'] = "3"  # not sure if it is not to late to set!!!!

print(sys.getrecursionlimit())
sys.setrecursionlimit(100000)
print(sys.getrecursionlimit())

if 'NUMBA_OPT' in os.environ:
    print("numba opt environment ", os.environ['NUMBA_OPT'])
    sys.stdout.flush()

from sympy import lambdify, exp, I, pi, conjugate, Matrix, diff, sqrt, N, cos, symbols
import numba, vegas, time, numpy
import random, math, json

# Excitations not yet supported in 3D
# Hermite polynoms used for excitations
def H(x_var, n):
    x = symbols('x')
    rv =  (-1)**n*exp(x**2)*diff(exp(-x**2),x,n)
    rv = rv.subs(x, x_var)
    return rv

# the excitations are fixed with USE_HERMITES, so no they can not be changed during calls, but well during recalculation
USE_HERMITES = True
# *************************************

# (PREADAPT_NUM, FP_EVAL, INT_ALPHA, N_ITER, ENABLE_ADAPT)
CHECK_INTEGRTATION_PRECISION = [] # [(50000, 1000000, 0.5, 300, True),(50000, 1000000, 0.1, 300, True),(50000, 1000000, 0.01, 300, True),(50000, 5000000, 0.5, 300, True),(50000, 5000000, 0.1, 300, True),(50000, 5000000, 0.001, 300, True)]

DoCompile = True # setting to false can help debugging, but some @numba directives must be removed too at the moment

DoSimplifiedDiffQ = True
UseSymmetryForDiff_q = False # this means the differentiation coresponding to all atom positions is the same, only not the number!

output_name = "results_"+time.strftime("%Y%m%d_%H:%M:%S", time.localtime())+".json"

# Only True supported at the moment in 3D
CHAIN_WF_ONLY_REAL_SPACE_NO_EXCITATIONS = True

S_RELAX = 1 # relax ratio of the lattice constant
DO_RELAX = 0 # Relax steps
EL_PARAM_STEPS = 0.1 # other electron parameter steps

DO_NUCLEI_POTENTIAL = True
DO_MEAN = True
CHECK_MEAN_FIELD_NORMILIZATION = False # DO_MEAN

NOT_SYMMETRIZED = True
OPTIMIZE_ROUNDS = 3
ROUND_MULT_EVALS = 4
ROUND_MULT_N_ITER = 1

MP_THREADS = 24
nhcubes = 5000
MAX_MEM = 5e9

# Parameters supposed to be changed
# NParticles = 5
NUM_UNIT_CELLS = [5,1,1] # number of unit cells in each direction, first dimension is the supercell direction for the phonons
EL_WV_LIST_DEF = [
    [1, 0, 0, True],
    [2, 0, 0, True],
    [3, 0, 0, True],
    [4, 0, 0, True],
#    [2, 0, 0, True],
#    [1, 1, 0, True],
#    [2, 1, 0, True],
#    [3, 1, 0, True],
#    [4, 1, 0, True],
    [2, 1, 0, True]
]

UNIT_CELL_OVERLAP = 3


DEBUG_PLOTTING = False
DEBUG = False
if DEBUG: EL_WV_LIST_DEF = [ [ 0, 1, 0, True ], [ 1, 1, 0, True ], [ 2, 1, 0, True ], [ 3, 1, 0, True ], [ 4, 1, 0, True ], [ 5, 1, 0, True ], [ 0, 0, 1, True ], [ 1, 0, 0, True ], [ 2, 0, 0, True ], [ 3, 0, 0, True ], [ 4, 0, 0, True ], [ 5, 0, 0, True ] ]

EVALS=10000
FP_EVALS = EVALS * 10
DO_AFTER_OPTIMIZATION_EVAL = FP_EVALS
DO_AFTER_SIMULTAN = FP_EVALS

N_ELECTRON_STATES = [[1,0,0], [-1,0,0]]
excitations = None # [0,0,0,0,0]       # not yet supported and overwritten at loading from file

# Parameters to optimize integration
PREADAPT_NUM = 100000
ENABLE_ALWAYS_ADAPT = False

INT_ALPHA = 0.1
INT_BETA = 0 # disables stratification, makes not much sense with our integrand, probably harmfull
N_ITER = 100
WARMUP_ITER = 200
WARMUP_ITER_OPTIMIZE = 50

INT_NINC = 10000
ENABLE_ADAPT = True # if preadaption is good this can be set to false for very high dimensions

# GAUSS_SMALLER = 2 Now special optimizations for the actual wave function are done. The width of ThomasFermi shielding and Gaus ansatz for nuclei is used



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Configuration of the unit cell
# 
# as a multiple of the electron mass, we work in Rydberg atomic units (2m_e=1) as in quantum expresso, as we use their pseudo potentials


WhatToCalc = "CuO2"

#match WhatToCalc:
#    case "Al":
if WhatToCalc=="Al":
        # Al is in the following
        # Al
        # the first NUM_ATOM_TYPES are used, must be changed if different atom types are used
        Atom_Parameters = [
            [-2.7031537 ,  0.17364504, -0.44158174, -0.16704639,  0.05589287],   # Al
            [-2.7031537 ,  0.17364504, -0.44158174, -0.16704639,  0.05589287],   # Al
            ]

        Atom_Masses = [27]
        CHAIN_RS_PARM = [10] # Gaus width of real space amplitude for the atom types
        ATOM_Al = 0 # identifier for Al atoms

        DIST_UNITCELL = [7.63075, 7.63075, 7.63075]       # Aluminium parameters from quantum espresso (http://quantum-espresso.org/)
        UNIT_CELL_ATOMS = [[0,0,0, ATOM_Al],[0.5,0.5,0, ATOM_Al],[0.5,0,0.5, ATOM_Al],[0,0.5,0.5, ATOM_Al]]
        VALENCE_ELECTRONS = [3] # There are 3 valence electrons per atom in the pseudo potential 

        EXTEND_INTEGRAL_FACTOR = 1.5
        VP_FACT = 0.01 # this is a harmonic force, should become very small later, as the force between the atoms will result from electron phonon coupling, Only avoids simultan movement and the missing repulsive force of the nuclei (they are only considered for relaxiation at the moment)
#    case "Al2":
elif WhatToCalc=="Al2":
        # Test Al with two atoms, to check the two atom code
        # the first NUM_ATOM_TYPES are used, must be changed if different atom types are used
        Atom_Parameters = [
            [-2.7031537 ,  0.17364504, -0.44158174, -0.16704639,  0.05589287],   # Al
            [-2.7031537 ,  0.17364504, -0.44158174, -0.16704639,  0.05589287],   # Al
            ]

        Atom_Masses = [27,27] 
        CHAIN_RS_PARM = [10,10] # Gaus width of real space amplitude for the atom types # if not the same for all atoms, PREADAPT is incorrect!!
        ATOM_Al = 0 # identifier for Al atoms

        DIST_UNITCELL = [7.63075, 7.63075, 7.63075]       # Aluminium parameters from quantum espresso (http://quantum-espresso.org/)
        UNIT_CELL_ATOMS = [[0,0,0, 1],[0.5,0.5,0, ATOM_Al],[0.5,0,0.5, ATOM_Al],[0,0.5,0.5, ATOM_Al]]
        VALENCE_ELECTRONS = [3,3] # There are 3 valence electrons per atom in the pseudo potential 

        EXTEND_INTEGRAL_FACTOR = 1.5
        VP_FACT = 0.01 # this is a harmonic force, should become very small later, as the force between the atoms will result from electron phonon coupling, Only avoids simultan movement and the missing repulsive force of the nuclei (they are only considered for relaxiation at the moment)
#    case "CuO2":
elif WhatToCalc=="CuO2":
        # CuO2
        Atom_Parameters = [
            [-12.63733146,   0.51652043,  -3.15258552,  -0.27279421, 0.19859104],  # Cu 11 valence electrons
            [-11.87055761,   0.44851487,  -0.9361393 ,  -0.69372281, 0.36254616],  # O   6 valence electrons
            [-11.87055761,   0.44851487,  -0.9361393 ,  -0.69372281, 0.36254616]  # O   6 valence electrons
            ]
        Atom_Masses = [64,16,16]

        CHAIN_RS_PARM = [5,5,5] # Gaus width of real space amplitude for the atom types # if not the same for all atoms, PREADAPT is incorrect!!
        ATOM_Cu = 0 # identifier for Cu atoms
        ATOM_O = 1 # identifier for O atoms
        ATOM_O_SECOND = 2 # they are not identical to x direction excitations
        DIST_UNITCELL = [7.724, 7.724, 22]       # Gitterkonstanten: a = 3.8231 Å, b = 3.8864 Å, c = 11.6807Å (c is chosen large, only the CuO2 plane is used)
        UNIT_CELL_ATOMS = [[0,0,0, ATOM_Cu],[0,0.5,0, ATOM_O],[0.5,0,0, ATOM_O_SECOND]]
        VALENCE_ELECTRONS = [11,6,6] # valence electrons for Cu and O


        EXTEND_INTEGRAL_FACTOR = 5
        VP_FACT = 0.01 # this is a harmonic force, should become very small later, as the force between the atoms will result from electron phonon coupling, Only avoids simultan movement and the missing repulsive force of the nuclei (they are only considered for relaxiation at the moment)
        EL_WV_LIST_DEF = [
        [0, 0, 1, False],
        [0, 0, 2, False],
        [0, 0, 3, False],
        [0, 0, 4, False],
        [0, 1, 0, False],
        [0, 2, 0, False],
        [0, 3, 0, False],
        [5, 0, 0, False],
        [10, 0, 0, False],
        [15, 0, 0, False],
        [20, 0, 0, False],
        [1, 0, 0, True],
        [2, 0, 0, True],
        [3, 0, 0, True],
        [4, 0, 0, True],
        [1, 1, 0, True],
        [2, 1, 0, True],
        [3, 1, 0, True],
        [4, 1, 0, True],
        [1, 2, 0, True],
        [2, 2, 0, True],
        [3, 2, 0, True],
        [4, 2, 0, True],
        [1, 3, 0, True],
        [2, 3, 0, True],
        [3, 3, 0, True],
        [4, 3, 0, True],
        [1, 0, 1, True],
        [2, 0, 1, True],
        [3, 0, 1, True],
        [4, 0, 1, True],
        [1, 0, 2, True],
        [2, 0, 2, True],
        [3, 0, 2, True],
        [4, 0, 2, True],
        [1, 0, 3, True],
        [2, 0, 3, True],
        [3, 0, 3, True],
        [4, 0, 3, True],
        ]
#    case _:
else:
        raise Exception("Unknown WhatToCalc")

# end of configuration of the unit cell
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


DO_NORM_EL_WF = False

# in the pseudopotential subdirectory a fit is done
# Al_parameters = [-2.7031537 ,  0.17364504, -0.44158174, -0.16704639,  0.05589287]
@numba.jit(nopython=True, nogil=True)
def PseudoPotential_for_Al(x):
    return -2.7031537*numpy.exp(-1*0.17364504*x**2)*(1+-0.16704639*x**2+0.05589287*x**3) # +Al_parameters[2]


OPTIMIZE_PHASE = False # should be False ...

ZERO_KOEF_EL_WF = True

VE_FACT = 1 # must be 1, was from previous tests

old_res = None
last_steps = None
global_integrator = None # Global adaptive map, to use previous training

Comment = None
try:
    with open(sys.argv[1], 'r') as ff:
        calc_p = json.load(ff)
        print(calc_p)
    # NParticles = calc_p['NParticles']
    if 'NUM_UNIT_CELLS' in calc_p:
        NUM_UNIT_CELLS = calc_p['NUM_UNIT_CELLS']
    if 'EL_WV_LIST_DEF' in calc_p:
        EL_WV_LIST_DEF = calc_p['EL_WV_LIST_DEF']
    if 'INT_ALPHA' in calc_p:
        INT_ALPHA = calc_p['INT_ALPHA']
    if 'EL_PARAM_STEPS' in calc_p:
        EL_PARAM_STEPS = calc_p['EL_PARAM_STEPS']
    EVALS = calc_p['EVALS']
    FP_EVALS = EVALS
    if 'FP_EVALS' in calc_p:  # calculate only the results point, usually to be done with much more evals
        FP_EVALS = calc_p['FP_EVALS']
    if 'DIST_UNITCELL' in calc_p:
        DIST_UNITCELL = calc_p['DIST_UNITCELL']
        if not isinstance(DIST_UNITCELL, list):
            raise("DIST_UNITCELL must be a list")
    if 'VALENCE_ELECTRONS' in calc_p:
        VALENCE_ELECTRONS = calc_p['VALENCE_ELECTRONS']
        if not isinstance(VALENCE_ELECTRONS, list):
            raise("VALENCE_ELECTRONS must be a list")
    N_ELECTRON_STATES = calc_p['N_ELECTRON_STATES']
    if 'excitations' in calc_p:
        excitations = calc_p['excitations']
    else:
        excitations = [0] * NUM_UNIT_CELLS[0]
    if 'results' in calc_p:
        old_res = calc_p['results']['x']
    if 'last_steps' in calc_p:
        last_steps = calc_p['last_steps']
    if 'VP_FACT' in calc_p:
        VP_FACT = calc_p['VP_FACT']
    if 'EXTEND_INTEGRAL_FACTOR' in calc_p:
        EXTEND_INTEGRAL_FACTOR = calc_p['EXTEND_INTEGRAL_FACTOR']
    if 'VE_FACT' in calc_p:
        VE_FACT = calc_p['VE_FACT']
    if 'MP_THREADS' in calc_p:
        MP_THREADS = calc_p['MP_THREADS']
    if 'NOT_SYMMETRIZED' in calc_p:
        NOT_SYMMETRIZED = calc_p['NOT_SYMMETRIZED']
    if 'OPTIMIZE_ROUNDS' in calc_p:
        OPTIMIZE_ROUNDS = calc_p['OPTIMIZE_ROUNDS']
    if 'DO_AFTER_OPTIMIZATION_EVAL' in calc_p:
        DO_AFTER_OPTIMIZATION_EVAL = calc_p['DO_AFTER_OPTIMIZATION_EVAL']
    if 'CHAIN_WF_ONLY_REAL_SPACE_NO_EXCITATIONS' in calc_p:
        CHAIN_WF_ONLY_REAL_SPACE_NO_EXCITATIONS = calc_p['CHAIN_WF_ONLY_REAL_SPACE_NO_EXCITATIONS']
    if 'USE_HERMITES' in calc_p:
        USE_HERMITES = calc_p['USE_HERMITES']
    if 'UNIT_CELL_ATOMS' in calc_p:
        UNIT_CELL_ATOMS = calc_p['UNIT_CELL_ATOMS']
        if len(UNIT_CELL_ATOMS[0]) != 4:
            raise Exception('UNIT_CELL_ATOMS probably missing AtomIdentifier')
    if 'UNIT_CELL_OVERLAP' in calc_p:
        UNIT_CELL_OVERLAP = calc_p['UNIT_CELL_OVERLAP']
    if 'DO_RELAX' in calc_p:
        DO_RELAX = calc_p['DO_RELAX']
    if 'DO_AFTER_SIMULTAN' in calc_p:
        DO_AFTER_SIMULTAN = calc_p['DO_AFTER_SIMULTAN']
    if 'Comment' in calc_p:
        Comment = calc_p['Comment']
    if 'DO_NUCLEI_POTENTIAL' in calc_p:
        DO_NUCLEI_POTENTIAL = calc_p['DO_NUCLEI_POTENTIAL']
    if 'DO_MEAN' in calc_p:
        DO_MEAN = calc_p['DO_MEAN']
    if 'PREADAPT_NUM' in calc_p:
        PREADAPT_NUM = calc_p['PREADAPT_NUM']
    if 'N_ITER' in calc_p:
        N_ITER = calc_p['N_ITER']
    if 'ROUND_MULT_EVALS' in calc_p:
        ROUND_MULT_EVALS = calc_p['ROUND_MULT_EVALS']
    if 'ROUND_MULT_N_ITER' in calc_p:
        ROUND_MULT_N_ITER = calc_p['ROUND_MULT_N_ITER']
    if 'ENABLE_ADAPT' in calc_p:
        ENABLE_ADAPT = calc_p['ENABLE_ADAPT']
    if 'ENABLE_ALWAYS_ADAPT' in calc_p:
        ENABLE_ALWAYS_ADAPT = calc_p['ENABLE_ALWAYS_ADAPT']
    if 'WARMUP_ITER' in calc_p:
        WARMUP_ITER = calc_p['WARMUP_ITER']
    if 'WARMUP_ITER_OPTIMIZE' in calc_p:
        WARMUP_ITER_OPTIMIZE = calc_p['WARMUP_ITER_OPTIMIZE']
    if 'UseSymmetryForDiff_q' in calc_p:
        UseSymmetryForDiff_q = calc_p['UseSymmetryForDiff_q']
    if 'Atom_Parameters' in calc_p:
        Atom_Parameters = calc_p['Atom_Parameters']
    output_name = sys.argv[1]+"out.json"
    print("loaded from file", sys.argv[1], calc_p)
    print("Comment:", Comment)
    CHECK_INTEGRTATION_PRECISION = []
    print("No integration check!!!")
except NameError:
    print('no file name given')
except IndexError:
    print('no file name given')
except FileNotFoundError:
    print('not existing file name given')

def get_calc_p():
    calc_p = {}
    #calc_p['NParticles'] = NParticles
    calc_p['NUM_UNIT_CELLS'] = NUM_UNIT_CELLS
    calc_p['EL_WV_LIST_DEF'] = EL_WV_LIST_DEF
    calc_p['INT_ALPHA'] = INT_ALPHA
    calc_p['EL_PARAM_STEPS'] = EL_PARAM_STEPS
    calc_p['EVALS'] = EVALS
    calc_p['DIST_UNITCELL'] = DIST_UNITCELL
    calc_p['VALENCE_ELECTRONS'] = VALENCE_ELECTRONS
    calc_p['N_ELECTRON_STATES'] = N_ELECTRON_STATES
    calc_p['excitations'] = excitations
    calc_p['VP_FACT'] = VP_FACT
    calc_p['EXTEND_INTEGRAL_FACTOR'] = EXTEND_INTEGRAL_FACTOR
    calc_p['WIDTH_only_documentation'] = WIDTH # only used to keep the value for documentation
    calc_p['VE_FACT'] = VE_FACT
    calc_p['MP_THREADS'] = MP_THREADS
    calc_p['FP_EVALS'] = FP_EVALS
    calc_p['NOT_SYMMETRIZED'] = NOT_SYMMETRIZED
    calc_p['OPTIMIZE_ROUNDS'] = OPTIMIZE_ROUNDS
    calc_p['DO_AFTER_OPTIMIZATION_EVAL'] = DO_AFTER_OPTIMIZATION_EVAL
    calc_p['CHAIN_WF_ONLY_REAL_SPACE_NO_EXCITATIONS'] = CHAIN_WF_ONLY_REAL_SPACE_NO_EXCITATIONS
    calc_p['USE_HERMITES'] = USE_HERMITES
    calc_p['UNIT_CELL_ATOMS'] = UNIT_CELL_ATOMS
    calc_p['UNIT_CELL_OVERLAP'] = UNIT_CELL_OVERLAP
    calc_p['DO_RELAX'] = DO_RELAX
    calc_p['DO_AFTER_SIMULTAN'] = DO_AFTER_SIMULTAN
    if Comment is not None:
        calc_p['Comment'] = Comment
    calc_p['DO_NUCLEI_POTENTIAL'] = DO_NUCLEI_POTENTIAL
    calc_p['DO_MEAN'] = DO_MEAN
    calc_p['PREADAPT_NUM'] = PREADAPT_NUM
    calc_p['N_ITER'] = N_ITER
    calc_p['ROUND_MULT_EVALS'] = ROUND_MULT_EVALS
    calc_p['ROUND_MULT_N_ITER'] = ROUND_MULT_N_ITER
    calc_p['ENABLE_ADAPT'] = ENABLE_ADAPT
    calc_p['ENABLE_ALWAYS_ADAPT'] = ENABLE_ALWAYS_ADAPT
    calc_p['WARMUP_ITER'] = WARMUP_ITER
    calc_p['WARMUP_ITER_OPTIMIZE'] = WARMUP_ITER_OPTIMIZE
    calc_p['UseSymmetryForDiff_q'] = UseSymmetryForDiff_q
    calc_p['Atom_Parameters'] = Atom_Parameters
    return calc_p

# Parameters usualy not changed
Open_End = 0 # if 1, the end is open, if 0, the end is closed

# if NUM_UNIT_CELLS[0] != NParticles:
#     print("NUM_UNIT_CELLS[0] != NParticles")
#     exit()

def get_Atoms():
    ALL_ATOMS = []
    for i0 in range(-UNIT_CELL_OVERLAP, NUM_UNIT_CELLS[0]+UNIT_CELL_OVERLAP):
        i0_mod = i0 % NUM_UNIT_CELLS[0]
        for i1 in range(-UNIT_CELL_OVERLAP, NUM_UNIT_CELLS[1]+UNIT_CELL_OVERLAP):
            i1_mod = i1 % NUM_UNIT_CELLS[1]
            for i2 in range(-UNIT_CELL_OVERLAP, NUM_UNIT_CELLS[2]+UNIT_CELL_OVERLAP):
                i2_mod = i2 % NUM_UNIT_CELLS[2]
                c=0
                l=len(UNIT_CELL_ATOMS)
                for CELL in UNIT_CELL_ATOMS:
                    ALL_ATOMS.append([i0+CELL[0],i1+CELL[1],i2+CELL[2], i0_mod*NUM_UNIT_CELLS[1]*NUM_UNIT_CELLS[2]*l + i1_mod*NUM_UNIT_CELLS[2]*l + i2_mod * l + c, CELL[3]])                                                        # the last indicates the atom in the supercell
                    c+=1
    ALL_ATOMS = numpy.array(ALL_ATOMS)
    return ALL_ATOMS

UNIT_CELL_OVERLAP_OLD = UNIT_CELL_OVERLAP
UNIT_CELL_OVERLAP = 0
ONLY_SUPERCELL_ATOMS = numpy.array(DIST_UNITCELL + [ 1, 1]) * get_Atoms()

# for every WaveVector we have the number of plane wave coefficents
# EL_WaveVector           = [[0, 2*math.pi/( DIST_UNITCELL), 0, False], [0, 0, 2*math.pi/(DIST_UNITCELL), False], [2*math.pi/(NParticles * DIST_UNITCELL), 0, 0, True]  ]    # The forth is 1 if use the phonon Q's otherwise 0
# EL_NUM_for_WaveVector   = [1,                           1,                          NParticles                            ]


# just for reference, is created with the line below
# EL_WV_LIST1 = [
#     [0, 2*math.pi/(NUM_UNIT_CELLS[1] * DIST_UNITCELL), 0, False],
#     [0, 0, 2*math.pi/(NUM_UNIT_CELLS[2] * DIST_UNITCELL), False],
#     [2*math.pi/(NUM_UNIT_CELLS[0] * DIST_UNITCELL) * 1, 0, 0, True],
#     [2*math.pi/(NUM_UNIT_CELLS[0] * DIST_UNITCELL) * 2, 0, 0, True],
#     [2*math.pi/(NUM_UNIT_CELLS[0] * DIST_UNITCELL) * 3, 0, 0, True],
#     [2*math.pi/(NUM_UNIT_CELLS[0] * DIST_UNITCELL) * 4, 0, 0, True],
#     [2*math.pi/(NUM_UNIT_CELLS[0] * DIST_UNITCELL) * 5, 0, 0, True]
# ]

EL_WV_LIST = numpy.concatenate((numpy.array(EL_WV_LIST_DEF)[:,:3] * (2*math.pi/numpy.array(DIST_UNITCELL)/numpy.array(NUM_UNIT_CELLS)), numpy.array(EL_WV_LIST_DEF)[:,3:]), axis=1).tolist()

print('EL_WV_LIST', EL_WV_LIST, "def", EL_WV_LIST_DEF, "EL_PARAM_STEPS", EL_PARAM_STEPS)

INTEGRATE_ELECTRONS_VOLUME = [[0, NUM_UNIT_CELLS[0]*DIST_UNITCELL[0]], [0, NUM_UNIT_CELLS[1]*DIST_UNITCELL[1]], NUM_UNIT_CELLS[2]*[0,DIST_UNITCELL[2]]]

MEAN_FIELD_DIST = DIST_UNITCELL[0]   # this might be better the maximum of all directions or something more advanced
MEAN_FIELD_STD = MEAN_FIELD_DIST/4
ELECTRONS_IN_CALCULATION = VALENCE_ELECTRONS # It is approximatly the valence electrons, as the supercell contains about 20 atoms, and only 2 Electrons are calculated in total. Same reasoning for the MEAN Factor.  (Only of importance for the relax calculation)


# NUM_3D_ATOMS_IN_INTEGRATION_RANGE = numpy.array(NUM_UNIT_CELLS).prod()*len(UNIT_CELL_ATOMS) * 3
NUM_3D_ATOMS_IN_INTEGRATION_RANGE = len(ONLY_SUPERCELL_ATOMS) * 3
print("NUM_3D_ATOMS_IN_INTEGRATION_RANGE", NUM_3D_ATOMS_IN_INTEGRATION_RANGE)

def AtomtypeAtIndex(i): # index as in NUM_3D_ATOMS_IN_INTEGRATION_RANGE
    return int(ONLY_SUPERCELL_ATOMS[i//3,4])

# Valence electrons per INTEGRATE_ELECTONS_VOLUME / 2, as for Rydberg atomic units e^2 = 2
# MEAN_FACT = numpy.array(EL_NUM_for_WaveVector).prod() * len(UNIT_CELL_ATOMS) / 2 * VALENCE_ELECTRONS

VALENCE_ELECTONS_per_atom = 0
for atom in UNIT_CELL_ATOMS:
    VALENCE_ELECTONS_per_atom += VALENCE_ELECTRONS[atom[3]]
VALENCE_ELECTONS_per_atom /= len(UNIT_CELL_ATOMS)

MEAN_FACT = len(ONLY_SUPERCELL_ATOMS) / 2 * VALENCE_ELECTONS_per_atom    


# this is to make sure, that the parameters are set up correctly
NUM_ATOM_TYPES = len(CHAIN_RS_PARM)

UNIT_CELL_OVERLAP = UNIT_CELL_OVERLAP_OLD
ALL_ATOMS = get_Atoms()
ALL_ATOMS = numpy.array(DIST_UNITCELL + [1, 1]) * ALL_ATOMS            # the last is the variable number of the position variables of the nuclei in the integration range

MEAN_DIM = 0
if DO_MEAN:
    MEAN_DIM = 3

SYMMETRIC_WAVEFUNCTION = True

Atom_Parameters_numpy = numpy.array(Atom_Parameters)
Atom_Masses_numpy = numpy.array(Atom_Masses) * 1836

DIST_UNITCELL_numpy = numpy.array(DIST_UNITCELL)


MASS_ATOM = Atom_Masses_numpy.min()  # 27 * 1836 # aluminum times the proton-electron mass ratio (we only support one atom type at the moment)

#Parameters calculated from the parameters above, but from experience, calculation may be changed for higher precision e.g.
WIDTH=4 / MASS_ATOM **(1/4)  / VP_FACT ** (1/4)
WIDTH *= EXTEND_INTEGRAL_FACTOR
CALC_OVERLAP = 300

# Parameters calculated from the parameters above
RECIPROCAL_LENGTH = 2*pi/(NUM_UNIT_CELLS[0] * DIST_UNITCELL[0])   # the reciprocal length for electrons is only in the x direction
N_ELECTRONS = len(N_ELECTRON_STATES)

# NUM_KOEFS = sum(EL_NUM_for_WaveVector) + 1
NUM_KOEFS = len(EL_WV_LIST) + 1

@numba.jit(nopython=True, nogil=True)
def PseudoPotential(x, atom_type):
    ppp = Atom_Parameters_numpy[atom_type]
    return ppp[0]*numpy.exp(-1*ppp[1]*x**2)*(1+ppp[3]*x**2+ppp[4]*x**3) # +ppp[2]

if CHAIN_WF_ONLY_REAL_SPACE_NO_EXCITATIONS:
    NUM_CHAIN_PARAMS = len(Atom_Masses)
    if NUM_ATOM_TYPES != NUM_CHAIN_PARAMS:
        raise ValueError("NUM_ATOM_TYPES != NUM_CHAIN_PARAMS")
    wf_params = [1] + CHAIN_RS_PARM
else:
    wf_params = []
    NUM_CHAIN_PARAMS = int(NUM_UNIT_CELLS[0]/2)
    for i in range(NUM_CHAIN_PARAMS+1):
        wf_params.append(math.sqrt(MASS_ATOM) * math.sqrt(VP_FACT) * float(sqrt(2*(1-cos(2*pi/NUM_UNIT_CELLS[0] * i)))))



if excitations is None:
    excitations = NUM_UNIT_CELLS[0] * [0]       # set to the correct number, but not yet supported in 3D
if len(excitations) != NUM_UNIT_CELLS[0]:
    raise ValueError("excitations must be a list of length NUM_UNIT_CELLS[0]")

print("Old NParticles: ", NUM_UNIT_CELLS[0], "WIDTH: ", WIDTH, "EVALS: ", EVALS, "Open_End: ", Open_End)
print(time.strftime("Start time: %H:%M:%S", time.localtime()))
sys.stdout.flush()

rr = NUM_UNIT_CELLS[0]*[[-WIDTH,WIDTH]]
integrator = vegas.Integrator(rr)
print(integrator.mpi_rank)
sys.stdout.flush()

from types import SimpleNamespace
args = SimpleNamespace(**{})
args.verbose = 2

def f(n_xyz,k_xyz):
    return  exp(-I*(k_xyz[0]*n_xyz[0]+k_xyz[1]*n_xyz[1]+k_xyz[2]*n_xyz[2])) #Normalization probably not needed # 2pi was already in the k_xyz

# def qn(n,Q):
#     ret = 0
#     for k in range(0,NParticles):
#         ret += f(n,k) * Q[k]
#     return ret

def Qk(k_xyz,q):
    ret = 0
    for n in range(0,len(ONLY_SUPERCELL_ATOMS)): # This must iterate over all atoms in the supercell with the supercell order
        n_atom = ONLY_SUPERCELL_ATOMS[n]
        ret += f(n_atom,k_xyz) * q[n*3]    # the first direction is the one we take into account at the moment! Phonons only used for elongation along the x direction
    return ret

# def q(Q):
#     ret = []
#     for n in range(0,NParticles):
#         ret.append(qn(n,Q))
#     return Matrix(ret)

def Q(q):   # This must create Q for 3D later ... Or not used at all, as we could call directly the correct Qk function
    ret = [Qk([0,0,0],q)]
    for k in EL_WV_LIST:
        if k[3]:
            ret.append(Qk(k[0:3],q))
    return Matrix(ret)

if DEBUG_PLOTTING:
    # The first of EL_WV_LIST_DEF is used for the displacement generation
    inp = []
    shift = 0.4  *2*numpy.pi
    for atom in ONLY_SUPERCELL_ATOMS:
        p0 = numpy.cos(atom[0]*EL_WV_LIST[0][0]+atom[1]*EL_WV_LIST[0][1]+atom[2]*EL_WV_LIST[0][2] + shift) * 0.5
        # p0 = numpy.cos(atom[0]/DIST_UNITCELL*2*numpy.pi) * 0.1
        inp.append(p0)
        inp.append(0)
        inp.append(0)
        

    inp_nump=numpy.array(inp)
    print(N(Q(inp_nump)))

@numba.jit(nopython=True, nogil=True)
def v_lamb(q):
    ret =  (q[:NUM_3D_ATOMS_IN_INTEGRATION_RANGE]**2).sum()
    # this makes not much sense in 3D. just a q**2  for every direction to get some kind of harmonic oscillator, but actually the electrons should do the coupling later!!!!
    # for n in range(NParticles - Open_End):
    #     ret += (q[n]-q[(n+1)%NParticles])**2
    return VP_FACT * ret

@numba.jit(nopython=True, nogil=True)
def v_el_lamb(q, efact): # the last q's are for the positions of the electrons # the last of efact is the relaxiation of the lattice
    ret = 0
    for ne in range(N_ELECTRONS):
        pos_electron = q[NUM_3D_ATOMS_IN_INTEGRATION_RANGE + 3*ne:NUM_3D_ATOMS_IN_INTEGRATION_RANGE + 3*ne + 3]
        dd = - pos_electron + ALL_ATOMS[:,:3]*efact[-1] + numpy.concatenate((q[ALL_ATOMS[:,3].astype('int')*3].reshape(-1,1), q[ALL_ATOMS[:,3].astype('int')*3+1].reshape(-1,1), q[ALL_ATOMS[:,3].astype('int')*3+2].reshape(-1,1)), axis=1)              # the order of the coordinates is first all x, than all y, than all z
        # dd = numpy.linalg.norm(dd, axis=1) # not supported by numba at the moment
        dd = dd**2
        dd = dd.sum(axis=1)
        dd = dd**0.5
        
        # Check if all atom types are defined
        # CountAtoms = 0

        # This is needed for all atom types
        for now_atom in range(NUM_ATOM_TYPES):
            # CountAtoms += (ALL_ATOMS[:,4] == now_atom).sum()    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! This costs only to check, if no coding error is made, comment out for production
            dd1 = PseudoPotential(dd, now_atom) * (ALL_ATOMS[:,4] == now_atom) # PseudoPotential_for_Al(dd)   # dd * (ALL_ATOMS[:,4] == 1)   # Only atoms of type 1 
            dd1 = dd1.sum()
            ret += dd1

        # if CountAtoms != len(ALL_ATOMS[:,4]): # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! This costs only to check, if no coding error is made, comment out for production
        #     raise ValueError("Not all atom types are defined")


    return VE_FACT * ret

if DEBUG_PLOTTING:
    N_ELECTRONS = 1
    X = numpy.linspace(0,NUM_UNIT_CELLS[0]*DIST_UNITCELL[0],50)

    Y = numpy.linspace(0,NUM_UNIT_CELLS[2]*DIST_UNITCELL[2],50)

    X, Y = numpy.meshgrid(X,Y)
    
    def v_el_lamb_vec(X, Y, H):
        ret = []
        for i in range(X.shape[0]):
            ret1 = []
            for j in range(X.shape[1]):
                # inp_tmp = inp[:]
                # inp_tmp.append(X[i,j])
                # inp_tmp.append(Y[i,j])
                # inp_tmp.append(2)
                ret1.append(v_el_lamb(numpy.concatenate((inp_nump, numpy.array([X[i,j], H, Y[i,j]]))), numpy.array([1])))
            ret.append(ret1)
        return numpy.array(ret)
            

    V0 = v_el_lamb_vec(X, Y, DIST_UNITCELL[0]/4)
    V1 = v_el_lamb_vec(X, Y, DIST_UNITCELL[1]/8)
    V2 = v_el_lamb_vec(X, Y, 0)
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.suptitle('shift: '+str(numpy.around(shift/(2*numpy.pi),3))+' '+str(numpy.around(numpy.array(N(Q(inp_nump)[1]),dtype=complex),3)), fontsize=10)
    ax = fig.add_subplot(221, projection='3d')
    ax.plot_surface(X, Y, V0, cmap="plasma")
    ax = fig.add_subplot(222, projection='3d')
    ax.plot_surface(X, Y, V1, cmap="plasma")
    ax = fig.add_subplot(223, projection='3d')
    ax.plot_surface(X, Y, V2, cmap="plasma")
    
    # plt.show() # shown later

#@numba.jit(nopython=True, nogil=True)
def pot_nuclei_lamb(_, efact): # Calculates the potential of the 000 nuclei in all other nucleis
    # the potential for the 0,0,0 atom is used for the calculation
    dd = ALL_ATOMS[:,:3]*efact[-1] # the order of the coordinates is first all x, than all y, than all z
    # dd = numpy.linalg.norm(dd, axis=1) # not supported by numba at the moment

    if NUM_ATOM_TYPES>1:
        ret = 0
        for uc_atom in UNIT_CELL_ATOMS:
            pos = numpy.array(uc_atom[:3]) * DIST_UNITCELL_numpy
            dd1 = dd - pos
            dd1 = dd1**2
            dd1 = dd1.sum(axis=1)
            dd1 = dd1**0.5
            for now_atom in range(NUM_ATOM_TYPES):
                dd2 = PseudoPotential(dd1, now_atom) * (now_atom == ALL_ATOMS[:,4]) * (dd1 > 0.0001)
                ret += dd2.sum() * ELECTRONS_IN_CALCULATION[uc_atom[3]] # correct nucleis own potential
        return  - ret / len(UNIT_CELL_ATOMS)

    dd = dd**2
    dd = dd.sum(axis=1)
    dd = dd**0.5
    dd = PseudoPotential(dd, 0)
    ret = dd.sum() - PseudoPotential(0, 0) # correct nucleis own potential
    return -ELECTRONS_IN_CALCULATION[0] * ret

#deb=pot_nuclei_lamb(0,[-1])
#pot_nuclei_lamb(0,[-1])

# An electron mean field potential might be added by an addidtional 3D integration range for positions near the electrons (eg 3*[[-5,5]])
# Than we calculate the electron density phi_q_el_one at the postion near the electron and multiply it with a KelbgPotential with Thomas Fermi shielding
@numba.jit(nopython=True, nogil=True)
def KelgbPotentialTF(x,l,k0):
    x=x+0.000001
    return 1/x*numpy.exp(-x/k0)*(1-math.exp(-x**2/l**2)+math.sqrt(numpy.pi)*x/l*(1-math.erf(x/l)))

@numba.jit(nopython=True, nogil=True)
def v_el_mean_lamb(q, efact):
    if DO_MEAN:
        ret = 0
        for ne in range(N_ELECTRONS):
            pos_electron = q[NUM_3D_ATOMS_IN_INTEGRATION_RANGE + 3*ne:NUM_3D_ATOMS_IN_INTEGRATION_RANGE + 3*ne + 3]
            diff_mean = q[NUM_3D_ATOMS_IN_INTEGRATION_RANGE + 3*N_ELECTRONS:NUM_3D_ATOMS_IN_INTEGRATION_RANGE + 3*N_ELECTRONS + 3] # after last electron
            dd = numpy.linalg.norm(diff_mean)
            dd = KelgbPotentialTF(dd, 1, MEAN_FIELD_STD)   # some sample parameter used, should be improved             !!!!!!!!!!!!!!!!!!!!!!!!!!
            q_calc = numpy.concatenate((q[:-3], pos_electron + diff_mean))
            dd *= phi_q_one_dense_lamb(q_calc, efact) # it determines the density of the coordinate after the last electron !
            ret += dd
        return MEAN_FACT * ret
    else:
        return 0

def phi_q_el_one(xp, ne, me, koefs):  
    QP = Q(xp)          # This must use Q for 3D later ...
    if ZERO_KOEF_EL_WF:
        QP[0] = 1    # the "shift" amplitude is not used (constant value of wave function)
    if OPTIMIZE_PHASE:
        ret = exp(I * koefs[1])
    else:
        ret = 1
    offset = 2 # the first two koefs are not used at the moment
    norm = 1
    # for i_coef in range(len(EL_NUM_for_WaveVector)):
    #     for n in range(0, EL_NUM_for_WaveVector[i_coef]):
    n_parameter = 0
    for n in range(len(EL_WV_LIST)):
            phonon_amplitude = 1 # not used
            # if EL_WaveVector[i_coef][3]:
            if EL_WV_LIST[n][3]: # only this are used
                n_parameter += 1 # first is skipped, as it is k=0
                phonon_amplitude = QP[n_parameter] # % NParticles] # (n+1) is the first koefficient used in the base function
            ret += phonon_amplitude * (koefs[offset + 2*n] + I * koefs[offset + 2*n + 1]) * exp(I *    #(n+1) * I*  # (n+1) base function
            #        (EL_WaveVector[i_coef][0] / koefs[-1] * xp[NUM_3D_ATOMS_IN_INTEGRATION_RANGE+3*ne+0] + 
            #         EL_WaveVector[i_coef][1] / koefs[-1] * xp[NUM_3D_ATOMS_IN_INTEGRATION_RANGE+3*ne+1] + 
            #         EL_WaveVector[i_coef][2] / koefs[-1] * xp[NUM_3D_ATOMS_IN_INTEGRATION_RANGE+3*ne+2]))    # prepared for 3D, but number of koeffs might have to be increased
                    (EL_WV_LIST[n][0] / koefs[-1] * xp[NUM_3D_ATOMS_IN_INTEGRATION_RANGE+3*ne+0] + 
                     EL_WV_LIST[n][1] / koefs[-1] * xp[NUM_3D_ATOMS_IN_INTEGRATION_RANGE+3*ne+1] + 
                     EL_WV_LIST[n][2] / koefs[-1] * xp[NUM_3D_ATOMS_IN_INTEGRATION_RANGE+3*ne+2]))    # prepared for 3D, but number of koeffs might have to be increased
            if DO_NORM_EL_WF:
                norm *= phonon_amplitude * (koefs[offset + 2*n] + I * koefs[offset + 2*n + 1]) * conjugate(phonon_amplitude * (koefs[offset + 2*n] + I * koefs[offset + 2*n + 1])) # this is probably very small in most cases, tried to calculate with mathematica symbollically, but could not do it fully
    #    offset += 2 * EL_NUM_for_WaveVector[i_coef]
    for d in range(3):
        ret = ret * exp(I * RECIPROCAL_LENGTH / koefs[-1] * me[d] * xp[NUM_3D_ATOMS_IN_INTEGRATION_RANGE+3*ne+d])
    return ret/norm

if DEBUG_PLOTTING:
    koef_tmp = numpy.array([1, 0, 0.0, -0.02500000000000001, 1.025, 0.0, 0.9500000000000001, 0.0, 0.7500000000000001, 0.0, 0.5, 0.0, 0.0, -0.025000000000000015, 0.0, -0.025, 0.0, -0.02500000000000001, 0.0, 0.0, 1])
    # w2 = phi_q_el_one(numpy.concatenate((inp_nump, numpy.array([0, 0, 0]))), 0, [0,0,0], koef_tmp)
    def phi_q_el_one_vec(X, Y):
        ret = []
        for i in range(X.shape[0]):
            ret1 = []
            for j in range(X.shape[1]):
                ret1.append(N(phi_q_el_one(numpy.concatenate((inp_nump, numpy.array([X[i,j], 0, Y[i,j]]))), 0, [0,0,0], koef_tmp) * conjugate(phi_q_el_one(numpy.concatenate((inp_nump, numpy.array([X[i,j], 0, Y[i,j]]))), 0, [0,0,0], koef_tmp))))
            ret.append(ret1)
        return numpy.array(ret)


    V0 = phi_q_el_one_vec(X, Y)

    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    ax = fig.add_subplot(224, projection='3d')
    ax.plot_surface(X, Y, V0, cmap="plasma")
    plt.show()
    sys.exit() # exit, as major globals might be changed, so normal calculation could go wrong (at least N_ELECTRON is changed)

def phi_q_el_slater(xp, koefs): 
    symm = -1
    if SYMMETRIC_WAVEFUNCTION: # it are bosons or the symmetric wavefunction is a result of asymmetric spin functions
        symm = 1
    def getMatrixMinor(m,i,j):
        return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

    def getMatrixDeternminant(m):
        if len(m) == 2: #base case for 2x2 matrix
            return m[0][0]*m[1][1]+symm*m[0][1]*m[1][0]
        determinant = 0
        for c in range(len(m)):
            determinant += ((symm)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
        return determinant
    
    if NOT_SYMMETRIZED:
        return phi_q_el_one(xp, 0, N_ELECTRON_STATES[0], koefs)*phi_q_el_one(xp, 1, N_ELECTRON_STATES[1], koefs)
    
    slater_matrix = []
    for i in range(N_ELECTRONS):
        one_line = []
        for j in N_ELECTRON_STATES:
            one_line.append(phi_q_el_one(xp, i, j, koefs))
        slater_matrix.append(one_line)

    if len(slater_matrix) != N_ELECTRONS:
        raise RuntimeError('slater matrix with wrong number of electrons')
    return getMatrixDeternminant(slater_matrix)  


DoDiffMultiprocessing = True

def diff_q_el(xp, ee, efact):
    if DoDiffMultiprocessing:
        diffpool = multiprocess.Pool(processes=MP_THREADS)
        t = phi_full(xp, ee, efact)
        def d2(x):
            t1 = diff(t, x)   # this seems to be much faster if very large expression at generation, not tested at execution
            t2 = diff(t1, x)
            return t2
        dl = [xp[i] for i in range(NUM_3D_ATOMS_IN_INTEGRATION_RANGE, NUM_3D_ATOMS_IN_INTEGRATION_RANGE+3*N_ELECTRONS)]
        result = diffpool.map(d2, dl)
        res = 0
        for r in result:
            res += r
        diffpool.close()
        diffpool.join()
        return res

    ret = 0
    t = phi_full(xp, ee, efact)
    for n in range(NUM_3D_ATOMS_IN_INTEGRATION_RANGE, NUM_3D_ATOMS_IN_INTEGRATION_RANGE+3*N_ELECTRONS):
        # ret += diff(phi_full(xp, ee, efact), xp[n],2)
        t1 = diff(t, xp[n])   # this seems to be much faster if very large expression at generation, not tested at execution
        t2 = diff(t1, xp[n])
        ret += t2
    return ret

def diff_q(xp, ee, efact, AtomIdentifier=0):
    if UseSymmetryForDiff_q:
        if NUM_ATOM_TYPES > 1:
            raise RuntimeError('symmetry for diff_q not implemented for NUM_ATOM_TYPES > 1') 
        tmp = symbols('tmp')
        if DoSimplifiedDiffQ: # might be possible to speed up further, as phi_q has only one term in the product, which has to be taken into account for differentiation
            t = phi_q(xp, ee, efact)
        else:
            t = phi_full(xp, ee, efact)
        def d2(x):
            t1 = diff(t, x)   # this seems to be much faster if very large expression at generation, not tested at execution
            t2 = diff(t1, x)
            return t2
        print('start diff_q')
        dl = d2(xp[0])
        print('end diff_q', time.strftime("one_diff_sym %H:%M:%S", time.localtime()))
        if AtomtypeAtIndex(0) == AtomIdentifier:
            ret = dl
        else:
            ret=0
        for i in range(1, NUM_3D_ATOMS_IN_INTEGRATION_RANGE):
            if AtomtypeAtIndex(i) == AtomIdentifier:
                ret += dl.subs({xp[i]: tmp}).subs({xp[0]:xp[i]}).subs({tmp:xp[0]})
        if DoSimplifiedDiffQ:
            return ret * phi_q_el_slater(xp, efact[NUM_CHAIN_PARAMS+1:])
        else:
            return ret
            
    if DoDiffMultiprocessing:
        diffpool = multiprocess.Pool(processes=MP_THREADS)
        if DoSimplifiedDiffQ:
            t = phi_q(xp, ee, efact)
        else:
            t = phi_full(xp, ee, efact)
        def d2(x):
            t1 = diff(t, x)   # this seems to be much faster if very large expression at generation, not tested at execution
            t2 = diff(t1, x)
            return t2
        #dl = [xp[i] for i in range(NUM_3D_ATOMS_IN_INTEGRATION_RANGE)]
        dl = []
        for i in range(NUM_3D_ATOMS_IN_INTEGRATION_RANGE):
            if AtomtypeAtIndex(i) == AtomIdentifier:
                dl.append(xp[i])
        result = diffpool.map(d2, dl)
        res = 0
        for r in result:
            res += r
        diffpool.close()
        diffpool.join()
        if DoSimplifiedDiffQ:
            return res * phi_q_el_slater(xp, efact[NUM_CHAIN_PARAMS+1:])
        else:
            return res 

    if DoSimplifiedDiffQ: # this is much faster, but approximates the differentiations of phi_q_el_slater to be zero
        ret = 0
        t = phi_q(xp, ee, efact)
        for n in range(NUM_3D_ATOMS_IN_INTEGRATION_RANGE):
            # ret += diff(phi_full(xp, ee, efact), xp[n],2)
            if AtomtypeAtIndex(n) == AtomIdentifier:
                t1 = diff(t, xp[n])   # this seems to be much faster if very large expression at generation, not tested at execution
                t2 = diff(t1, xp[n])
                ret += t2
        return ret * phi_q_el_slater(xp, efact[NUM_CHAIN_PARAMS+1:])
    else:
        ret = 0
        t = phi_full(xp, ee, efact)
        for n in range(NUM_3D_ATOMS_IN_INTEGRATION_RANGE):
            # ret += diff(phi_full(xp, ee, efact), xp[n],2)
            if AtomtypeAtIndex(n) == AtomIdentifier:
                t1 = diff(t, xp[n])   # this seems to be much faster if very large expression at generation, not tested at execution
                t2 = diff(t1, xp[n])
                ret += t2
        return ret

# This must be 3D now, but is not yet
# def phi_Q(Q,ee,efact): # This must handle Q for 3D later ...
#     prod = 1
#     for k in range(1,NParticles):
#         ki = k
#         if k > int(NParticles/2):
#             ki = NParticles - k
#         if USE_HERMITES:
#             prod *= exp(-efact[ki] / 2 * Q[k]*conjugate(Q[k])) * H(sqrt(efact[ki])*Q[k], excitations[k])
#         else:
#             prod *= exp(-efact[ki] / 2 * Q[k]*conjugate(Q[k])) * (ee[k] * Q[k]  + (1 - ee[k])) 
        
#     return prod

def phi_q(q, ee, efact):
    if not CHAIN_WF_ONLY_REAL_SPACE_NO_EXCITATIONS:
        return phi_Q(Q(q), ee,efact)
    else:
        prod = 1
        for k in range(0,NUM_3D_ATOMS_IN_INTEGRATION_RANGE):                 # all dimensions are used
            atom_type = AtomtypeAtIndex(k)
            if atom_type >= NUM_ATOM_TYPES:
                raise Exception('atom_type >= NUM_ATOM_TYPES') # this test does not cost during calculation
            prod *= exp(-efact[1 + atom_type] * q[k]**2)              # efact[1], because in the Q space no efact[0] is used, therefore it is not included in the parameters
        return prod

def phi_full(q, ee, efact):
    return phi_q(q, ee, efact) * phi_q_el_slater(q, efact[NUM_CHAIN_PARAMS+1:])

xp_symbols = list(symbols(''.join(' xp'+str(i) for i in range(NUM_3D_ATOMS_IN_INTEGRATION_RANGE+3*N_ELECTRONS + MEAN_DIM)), real=True))
ee1_symbols = list(symbols(''.join(' ee1'+str(i) for i in range(NUM_UNIT_CELLS[0])), real=True))
ee2_symbols = list(symbols(''.join(' ee2'+str(i) for i in range(NUM_UNIT_CELLS[0])), real=True))
efact_symbols = list(symbols(''.join(' efact'+str(i) for i in range(NUM_CHAIN_PARAMS+1 + NUM_KOEFS*2 + 1)), real=True))

xp = Matrix(xp_symbols) # MatrixSymbol('xp',NParticles,1)
ee1 = Matrix(ee1_symbols) # ee1 = MatrixSymbol('eea',NParticles,1) 
ee2 = Matrix(ee2_symbols) # ee2 = MatrixSymbol('eeb',NParticles,1) 

efact = Matrix(efact_symbols) # efact = MatrixSymbol('efact',int(NParticles/2)+1 + NUM_KOEFS*2,1)

# ************************************************************************************************************************************************************
# Attention, this part is copied to below, as for FullRun global parameters have to be changed, which makes new creation and compilation of the expressions necessary


# The next functions are removed from sympy generation, the predefined functions are used instead
# 
# v_el_lamb = lambdify([xp], v_el(xp), cse=True, modules=['numpy'])
# if DoCompile:
#     v_el_lamb = numba.jit(v_el_lamb, nogil=True)

# v_lamb = lambdify([xp], v(xp), cse=True, modules=['numpy'])
# if DoCompile:
#     v_lamb = numba.jit(v_lamb, nogil=True)


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# phi_q_el_one is not normalized, therefore the calculation is wrong. This part has to be divided by the phi_q_one_dense_lamb integral ....
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

if DO_MEAN:
    phi_q_one_dense_lamb = lambdify([xp, efact], conjugate(phi_q_el_one(xp, N_ELECTRONS, [0,0,0], efact[NUM_CHAIN_PARAMS+1:]) * phi_q_el_one(xp, N_ELECTRONS, [0,0,0], efact[NUM_CHAIN_PARAMS+1:])) , cse=True, modules=['numpy'])    # the last is taken
    if DoCompile:
        phi_q_one_dense_lamb = numba.jit(phi_q_one_dense_lamb, nogil=True)


phi_full_lamb = lambdify([xp, ee1, efact], phi_full(xp,ee1,efact), cse=True, modules=['numpy'])
if DoCompile:
    phi_full_lamb = numba.jit(phi_full_lamb, nogil=True)

tttt = 0
for i in range(len(Atom_Masses)):
    tttt += diff_q(xp,ee1,efact,i) / Atom_Masses_numpy[i]
print(time.strftime("diff: %H:%M:%S", time.localtime()))
energy_q_lamb  = lambdify([xp, ee1, efact], tttt, cse=True, modules=['numpy'])
print(time.strftime("lamb: %H:%M:%S", time.localtime()))
if DoCompile:
    energy_q_lamb = numba.jit(energy_q_lamb, nogil=True)
    print(time.strftime("diff_q_lamb: %H:%M:%S", time.localtime()))

tttt = diff_q_el(xp,ee1,efact)
print(time.strftime("diff_q_el: %H:%M:%S", time.localtime()))
diff_q_el_lamb   = lambdify([xp, ee1, efact], tttt, cse=True, modules=['numpy'])
print(time.strftime("diff_q_el_lamb: %H:%M:%S", time.localtime()))
if DoCompile:
    diff_q_el_lamb = numba.jit(diff_q_el_lamb, nogil=True)
    print(time.strftime("compile diff_q_el_lamb: %H:%M:%S", time.localtime()))
#from numpy import conjugate as numpy_conjugate

#@numba.jit(nopython=True, nogil=True)
def integrand_H_numba(xp, ee1, ee2, efact, norm_density):
    return numpy.conjugate(phi_full_lamb(xp,ee1,efact)) * ((v_el_lamb(xp,efact)+v_lamb(xp)+v_el_mean_lamb(xp,efact)/norm_density) * phi_full_lamb(xp,ee2,efact) -  energy_q_lamb(xp, ee2, efact) - diff_q_el_lamb(xp, ee2, efact))
#@numba.jit(nopython=True, nogil=True)
def integrand_wf_numba(xp, ee1, ee2, efact):
    return phi_full_lamb(xp,ee1,efact)*numpy.conjugate(phi_full_lamb(xp,ee2,efact))
if DoCompile:
    integrand_H_numba = numba.jit(integrand_H_numba, nogil=True)
    print(time.strftime("integrand_H_numba: %H:%M:%S", time.localtime()))
    integrand_wf_numba = numba.jit(integrand_wf_numba, nogil=True)
    print(time.strftime("integrand_wf_numba: %H:%M:%S", time.localtime()))

#************************************************************************************************************************************************************




# Parallel version of adapt_to_samples
import numpy as np
import multiprocessing

def adapt_to_samples(map, x, fx, nitn=5, alpha=1.0, nproc=1):
    x = np.asarray(x)
    fx = np.asarray(fx)
    map_grid = np.array(map.grid)
    dim = map_grid.shape[0] 
    if dim < nproc:
        nproc = dim 
    args = []
    end = 0
    for i in range(nproc):
        nd = (dim - end) // (nproc - i)
        start = end
        end = start + nd
        args += [(
            map_grid[start:end].flat[:],
            x[:, start:end].flat[:],
            fx,
            nitn, alpha
            )]
    pool = multiprocessing.Pool(processes=nproc)
    results = pool.map(_apply, args)    
    return vegas.AdaptiveMap(np.concatenate(results, axis=0))

def _apply(args):
    grid, x, fx, nitn, alpha = args
    x.shape = (len(fx), -1)
    grid.shape = (x.shape[1], -1)
    map = vegas.AdaptiveMap(grid)
    map.adapt_to_samples(x, fx, nitn=nitn, alpha=alpha)
    return np.asarray(map.grid).tolist() 

class parallelintegrand(vegas.BatchIntegrand):
    """ Convert integrand into multiprocessor integrand.
    Integrand can only handle a single input !!!!
    Integrand should return a numpy array.
    """
    def __init__(self, fcn, nproc=MP_THREADS):
        " Save integrand; create pool of nproc processes. "
        self.fcn = fcn
        self.nproc = nproc
        self.pool = multiprocess.Pool(processes=nproc)

    def __del__(self):
        " Standard cleanup. "
        self.pool.close()
        self.pool.join()

    def __call__(self, x):
        " Divide x into chunks feeding one to each process. "
        chunks = self.nproc * 20
        # nx = x.shape[0] // chunks + 1
        num_chanks = min(chunks, x.shape[0])
        # launch evaluation of self.fcn for each chunk, in parallel
        results = self.pool.map(
            self.fcn,
            numpy.array_split(x, num_chanks), 
            )
        return numpy.concatenate(results)

def integrand_wf_debug(xp, ee1, ee2, efact):
    r =integrand_wf_numba(xp, ee1, ee2,efact)
    return r


while len(excitations) < NUM_CHAIN_PARAMS+1:
    excitations.append(0)

# here to compile the integrands
test_list = [(random.random() - 0.5) for i in range(NUM_3D_ATOMS_IN_INTEGRATION_RANGE+3*N_ELECTRONS + MEAN_DIM)]
norm_d = 1
res_test=integrand_H_numba(numpy.array(test_list).T, numpy.array(excitations), numpy.array(excitations), numpy.array(wf_params+[1]+[0.1]*(NUM_KOEFS*2-1)+[1]), norm_d)
print(res_test)
print(integrand_wf_numba(numpy.array(test_list).T, numpy.array(excitations), numpy.array(excitations), numpy.array(wf_params+[1]+[0.1]*(NUM_KOEFS*2-1) + [1])))
print(time.strftime("Time: %H:%M:%S", time.localtime()))
sys.stdout.flush()

if DEBUG:
    deb = phi_q_el_one(numpy.array(test_list).T, 0, [0,0,0], numpy.array([1]+[0.1]*(NUM_KOEFS*2-1 + 1)))

if integrator.mpi_rank == 0:
    calc_p = get_calc_p()
    with open('dump_tmp.json', 'w') as fp:
        json.dump(calc_p, fp,  indent=4)

lf = None
def batch_from_array(x):
    rr = []
    for l in x:
        rr.append(lf(l))
    return numpy.array(rr)

# vectorized version worked but not faster
# def batch_from_array_vec(x):
#     x1 = x.T
#     rr = lf(x1)
#     rt = rr.T
#     return rt.copy(order='C')

def instead_batch(x):
    return lf(numpy.array(x))

lf_preadapt = None
last_log = None
def do_integrate(wf_params):
    global lf, last_log, lf_preadapt, global_integrator
    PREADAPT_STD = math.sqrt(1/2/wf_params[1]) #  This is not yet for more than one atome type, but should only be a performance issue, as PREADAPT is only used once in the beginning. In case we start with the same PREADAPT parameter for all atoms, this is correct.
    start0 = time.time()
    excit = numpy.array(excitations)
    wf_params_mtx = numpy.array(wf_params)
    i_e_v = (numpy.array(INTEGRATE_ELECTRONS_VOLUME)*wf_params_mtx[-1]).tolist()
    if CHECK_MEAN_FIELD_NORMILIZATION:
        # integrate the normal density
        
        # not sure, but every testposition should give the correct density integral
        testpos = numpy.array([(random.random() - 0.5)*0.1 for i in range(NUM_3D_ATOMS_IN_INTEGRATION_RANGE+3*N_ELECTRONS + MEAN_DIM)])
        #@numba.jit(nogil=True) # This is done allways, as this runs usually take very long
        def lf_norm(x):
            x = numpy.concatenate((testpos[:-3],x)) # use the last electron range to integrate the normal density
            return numpy.array(phi_q_one_dense_lamb(x, wf_params_mtx).real)
        lf = lf_norm
        test_value = lf(numpy.array([0.1]*3)) # done, to compile before integration (multithreading might compile in every thread?)
        # test =N(phi_q_el_one([(random.random() - 0.5)*0.1 for i in range(NUM_3D_ATOMS_IN_INTEGRATION_RANGE+3*N_ELECTRONS + 3)], 0, [0,0,0], wf_params_mtx[int(NParticles/2)+1:]))
        if MP_THREADS > 1:
            parallel_integrand = parallelintegrand(batch_from_array)
        else:
            parallel_integrand = instead_batch

        rr = numpy.array(i_e_v)
        global_integrator = vegas.Integrator(rr, nhcube_batch = nhcubes, max_mem = MAX_MEM, alpha=INT_ALPHA, beta=INT_BETA)
        start = time.time()
        rr =  global_integrator(parallel_integrand, nitn=N_ITER, neval=FP_EVALS / 10)

        if global_integrator.mpi_rank == 0:
            if args.verbose >1:
                print('test value', phi_q_one_dense_lamb(numpy.concatenate((testpos[:-3],numpy.array([0.1]*3))), wf_params_mtx))
                print('norm integral\n',rr.summary())
        
        norm_d = rr.mean
    norm_d = numpy.array(i_e_v)[:,1].prod()

    @numba.jit(nogil=True) # This is done allways, as this runs usually take very long
    def lf_local(x):
        # norm_d=1
        return numpy.array([integrand_H_numba(x, excit, excit, wf_params_mtx, norm_d).real,integrand_wf_numba(x, excit, excit, wf_params_mtx).real])
    lf = lf_local
    lf(numpy.array([0.1]*(NUM_3D_ATOMS_IN_INTEGRATION_RANGE+3*N_ELECTRONS + MEAN_DIM))) # done, to compile before integration (multithreading might compile in every thread?)
    
    if MP_THREADS > 1:
        parallel_integrand = parallelintegrand(batch_from_array)
    else:
        parallel_integrand = instead_batch

    if ENABLE_ALWAYS_ADAPT or global_integrator is None:
        rr = NUM_3D_ATOMS_IN_INTEGRATION_RANGE*[[-WIDTH,WIDTH]] + i_e_v * N_ELECTRONS + MEAN_DIM * [[-MEAN_FIELD_DIST, MEAN_FIELD_DIST]]
        global_adaptive_map = vegas.AdaptiveMap(rr, ninc=INT_NINC) 
    
        if PREADAPT_NUM > 0:
            if not all(x == CHAIN_RS_PARM[0] for x in CHAIN_RS_PARM):
                raise Exception('CHAIN_RS_PARM must be the same for all atoms, for PREADAPT_NUM > 0 at the moment')
            print(time.strftime("Start preadapt1: %H:%M:%S", time.localtime()))
            sys.stdout.flush()
            x_samp = numpy.random.normal(loc=0, scale=PREADAPT_STD, size=(PREADAPT_NUM,NUM_3D_ATOMS_IN_INTEGRATION_RANGE))
            print(time.strftime("Start preadapt2: %H:%M:%S", time.localtime()))
            sys.stdout.flush()
            for _ in range(N_ELECTRONS):
                for ll in i_e_v:
                    x_samp = numpy.concatenate([x_samp, numpy.random.uniform(low=ll[0],high=ll[1], size=(PREADAPT_NUM,1))], axis=1)
            print(time.strftime("Start preadapt3: %H:%M:%S", time.localtime()))
            sys.stdout.flush()
            if DO_MEAN:
                x_samp = numpy.concatenate([x_samp, numpy.random.normal(loc=0, scale=MEAN_FIELD_STD, size=(PREADAPT_NUM,3))], axis=1)

            @numba.jit()
            def lf_local_vec(xx):
                ret = []
                norm_d = 1
                for x in xx:
                    ret.append(integrand_H_numba(x, excit, excit, wf_params_mtx, norm_d).real)
                return numpy.array(ret)
            lf_preadapt = lf_local_vec
            pint = parallelintegrand(lf_preadapt)
            print(time.strftime("Start preadapt4: %H:%M:%S", time.localtime()))
            sys.stdout.flush()
            parray = pint(x_samp)
            print(time.strftime("Start preadapt5: %H:%M:%S", time.localtime()))
            sys.stdout.flush()
            # global_adaptive_map.adapt_to_samples(x_samp, parray, nitn=N_ITER)    # seems to create map with ninc=1000 if not specified before????
            global_adaptive_map = adapt_to_samples(global_adaptive_map,x_samp, parray, nitn=N_ITER, nproc=MP_THREADS)    # seems to create map with ninc=1000 if not specified before????
            print(time.strftime("Start preadapt6: %H:%M:%S", time.localtime()),'alpha',INT_ALPHA,'beta',INT_BETA)
            sys.stdout.flush()

            #nstrat =  NUM_3D_ATOMS_IN_INTEGRATION_RANGE *[1] + 3 * N_ELECTRONS * [1] + MEAN_DIM * [5]   
            global_integrator = vegas.Integrator(global_adaptive_map, nhcube_batch = nhcubes, max_mem = MAX_MEM, alpha=INT_ALPHA, beta=INT_BETA) #, nstrat=nstrat)
    start = time.time()
    if WARMUP_ITER > 0: # this is slower than needed, as one only needs one integrand ... In case it will be used, this might be optimized
        warmup = global_integrator(parallel_integrand, nitn=WARMUP_ITER, neval=FP_EVALS, adapt = ENABLE_ADAPT)
        print('warmup\n',warmup.summary())
    rr =  global_integrator(parallel_integrand, nitn=N_ITER, neval=FP_EVALS, adapt = ENABLE_ADAPT)
    if DO_NUCLEI_POTENTIAL:
        nuclei_pot = pot_nuclei_lamb(None, wf_params_mtx)
    else:
        nuclei_pot = 0
    if global_integrator.mpi_rank == 0:
        last_log = rr.summary() + '\n' + time.strftime("Time: %H:%M:%S", time.localtime()) + " dt" + str(time.time() - start) + " compile and preadapt time " + str(time.time() - start0) + " result " + str(rr) + " " + str(rr[0]/ rr[1])
        if args.verbose >1:
            print(rr.summary())
        if args.verbose >0:
            print(time.strftime("Time: %H:%M:%S", time.localtime()), "dt", time.time()-start, "compile and preadapt time", start-start0, "result", rr, rr[0]/rr[1])
            sys.stdout.flush()
            if DO_NUCLEI_POTENTIAL:
                print("nuclei potential", nuclei_pot)
                sys.stdout.flush()
    return (rr[0]/rr[1]).mean + nuclei_pot

def do_integrate2(wf_params1, wf_params2):
    """
    the integration for two points is done simultaniously using vegas feature to take the same sampleing points for multiple integrands
    used for H and wf integration at two different wf_params
    """
    global lf, lf_preadapt, global_integrator
    PREADAPT_STD = math.sqrt(1/2/wf_params1[1]) # WIDTH/GAUSS_SMALLER
    start0 = time.time()
    excit = numpy.array(excitations)
    wf_params1_mtx = numpy.array(wf_params1)
    wf_params2_mtx = numpy.array(wf_params2)
    i_e_v = (numpy.array(INTEGRATE_ELECTRONS_VOLUME)*wf_params1_mtx[-1]).tolist()
    if CHECK_MEAN_FIELD_NORMILIZATION:
        # integrate the normal density
        
        # not sure, but every testposition should give the correct density integral
        testpos = numpy.array([(random.random() - 0.5)*0.1 for i in range(NUM_3D_ATOMS_IN_INTEGRATION_RANGE+3*N_ELECTRONS + MEAN_DIM)])
        #@numba.jit(nogil=True) # This is done allways, as this runs usually take very long
        def lf_norm(x):
            x = numpy.concatenate((testpos[:-3],x)) # use the last electron range to integrate the normal density
            return [numpy.array(phi_q_one_dense_lamb(x, wf_params1_mtx).real), numpy.array(phi_q_one_dense_lamb(x, wf_params2_mtx).real)]
        lf = lf_norm
        lf(numpy.array([0.1]*3)) # done, to compile before integration (multithreading might compile in every thread?)
        # test =N(phi_q_el_one([(random.random() - 0.5)*0.1 for i in range(NUM_3D_ATOMS_IN_INTEGRATION_RANGE+3*N_ELECTRONS + 3)], 0, [0,0,0], wf_params_mtx[int(NParticles/2)+1:]))
        if MP_THREADS > 1:
            parallel_integrand = parallelintegrand(batch_from_array)
        else:
            parallel_integrand = instead_batch

        rr = i_e_v
        global_integrator = vegas.Integrator(rr, nhcube_batch = nhcubes, max_mem = MAX_MEM, alpha=INT_ALPHA, beta=INT_BETA)
        start = time.time()
        rr =  global_integrator(parallel_integrand, nitn=N_ITER, neval=FP_EVALS / 10)
        if global_integrator.mpi_rank == 0:
            if args.verbose >1:
                print('test value', phi_q_one_dense_lamb(numpy.concatenate((testpos[:-3],numpy.array([0.1]*3))), wf_params1_mtx))
                print('norm integral\n',rr.summary())
        
        norm_d1 = rr[0].mean
        norm_d2 = rr[1].mean
    norm_d1 = numpy.array(i_e_v)[:,1].prod()
    norm_d2 = numpy.array(i_e_v)[:,1].prod()
    

    # @numba.jit(nogil=True) # This might leak memory, but at the moment we have enough
    def lf_local(x):
        # norm_d1 = 1
        # norm_d2 = 1
        return numpy.array([integrand_H_numba(x, excit, excit, wf_params1_mtx, norm_d1).real,integrand_wf_numba(x, excit, excit, wf_params1_mtx).real,
                                integrand_H_numba(x, excit, excit, wf_params2_mtx, norm_d2).real,integrand_wf_numba(x, excit, excit, wf_params2_mtx).real])

    lf = lf_local
    if MP_THREADS > 1:
        parallel_integrand = parallelintegrand(batch_from_array)
    else:
        parallel_integrand = instead_batch

    if ENABLE_ALWAYS_ADAPT or global_integrator is None:
        rr = NUM_3D_ATOMS_IN_INTEGRATION_RANGE*[[-WIDTH,WIDTH]] + i_e_v * N_ELECTRONS + MEAN_DIM * [[-MEAN_FIELD_DIST, MEAN_FIELD_DIST]]
        global_adaptive_map = vegas.AdaptiveMap(rr, ninc=INT_NINC)
        
        if PREADAPT_NUM > 0:
            x_samp = numpy.random.normal(loc=0, scale=PREADAPT_STD, size=(PREADAPT_NUM,NUM_3D_ATOMS_IN_INTEGRATION_RANGE))
            for _ in range(N_ELECTRONS):
                for ll in i_e_v:
                    x_samp = numpy.concatenate([x_samp, numpy.random.uniform(low=ll[0],high=ll[1], size=(PREADAPT_NUM,1))], axis=1)
            if DO_MEAN:
                x_samp = numpy.concatenate([x_samp, numpy.random.normal(loc=0, scale=MEAN_FIELD_STD, size=(PREADAPT_NUM,3))], axis=1)

            @numba.jit()
            def lf_local_vec(xx):
                ret = []
                norm_d = 1
                for x in xx:
                    ret.append(integrand_H_numba(x, excit, excit, wf_params1_mtx, norm_d).real)
                return numpy.array(ret)
            lf_preadapt = lf_local_vec
            pint = parallelintegrand(lf_preadapt)
            global_adaptive_map.adapt_to_samples(x_samp, pint(x_samp), nitn=N_ITER)    
            global_integrator = vegas.Integrator(global_adaptive_map, nhcube_batch = nhcubes, max_mem = MAX_MEM, alpha=INT_ALPHA, beta=INT_BETA)

    start = time.time()
    if WARMUP_ITER_OPTIMIZE > 0: # this is slower than needed, as one only needs one integrand ... In case it will be used, this might be optimized
        warmup = global_integrator(parallel_integrand, nitn=WARMUP_ITER_OPTIMIZE, neval=EVALS, adapt = ENABLE_ADAPT)
        print('warmup\n',warmup.summary())
    rr =  global_integrator(parallel_integrand, nitn=N_ITER, neval=EVALS)
    if DO_NUCLEI_POTENTIAL:
        nucl_pot1 = pot_nuclei_lamb(None, wf_params1_mtx)
        nucl_pot2 = pot_nuclei_lamb(None, wf_params2_mtx)
    else:
        nucl_pot1 = 0
        nucl_pot2 = 0
    if global_integrator.mpi_rank == 0:
        if args.verbose >1:
            print(rr.summary())
        if args.verbose >0:
            print(time.strftime("Time: %H:%M:%S", time.localtime()), "dt", time.time()-start, "compile and preadapt time", start-start0, "result", rr, rr[0]/rr[1], rr[2]/rr[3])
            sys.stdout.flush()
            if DO_NUCLEI_POTENTIAL:
                print("Nuclear potentials:", nucl_pot1, nucl_pot2)
    return (rr[0]/rr[1]).mean + nucl_pot1, (rr[2]/rr[3]).mean + nucl_pot2

from scipy.optimize import OptimizeResult

def optimize(f, point_in, steps_in, iter=5):
    point = point_in[:]
    f(point) # pretrain the integrator
    bestvalue = f(point)
    bestpoint = point[:]
    steps = steps_in[:]
    for r in range(iter):
        for d in range(len(point)):
            if steps[d] != 0:
                if integrator.mpi_rank == 0:
                    if args.verbose >0:
                        print("iteration: ", r, " direction ", d, "best: ", bestvalue, "point", bestpoint)
                point[d] += steps[d]
                tmpvalue = f(point)
                if integrator.mpi_rank == 0:
                    if args.verbose >0:
                        print(tmpvalue, point)
                if tmpvalue < bestvalue:
                    bestvalue = tmpvalue
                    bestpoint = point[:]
                    if integrator.mpi_rank == 0:
                        if args.verbose >0:
                            print("new best: ", bestvalue, "point", bestpoint,r,d)
                            sys.stdout.flush()
                else:
                    steps[d] = -steps[d]
                    point[d] += steps[d]
                ok = True
                while ok:
                    point[d] += steps[d]
                    tmpvalue = f(point)
                    if integrator.mpi_rank == 0:
                        if args.verbose >0:
                            print(tmpvalue, point)
                    if tmpvalue < bestvalue:
                        bestvalue = tmpvalue
                        bestpoint = point[:]
                        if integrator.mpi_rank == 0:
                            if args.verbose >0:
                                print("new best: ", bestvalue, "point", bestpoint,r,d)
                                sys.stdout.flush()
                    else:
                        ok = False
                        point[d] -= steps[d]
        for d in range(len(steps)):
            steps[d] *= 0.5
    return OptimizeResult({'x':bestpoint, 'fun':bestvalue})

def optimize2(f, point_in, steps_in, point_min = [], iter=5):
    # pylint: disable=W0102
    """
    optimizing Monte Carlo results is quite difficult, as they have statistical noise.
    Therefore, we use a simple optimization algorithm to find the best parameters.
    To check if a point is better than an earlier we use integrate2, which does integration at both points using the same sample points
    """
    global EVALS, N_ITER
    point = point_in[:]
    #f(point,point) # pretrain the integrator
    bestvalue, _ = f(point,point) # does the pretraining, not used later
    bestpoint = point[:]
    steps = steps_in[:]
    for r in range(iter):
        for d in range(len(point)):
            if steps[d] != 0:
                if integrator.mpi_rank == 0:
                    if args.verbose >0:
                        print("iteration: ", r, " direction ", d, "best: ", bestvalue, "point", bestpoint)
                point[d] += steps[d]
                bestvalue, tmpvalue = f(bestpoint, point)
                if integrator.mpi_rank == 0:
                    if args.verbose >0:
                        print(tmpvalue, point)
                if tmpvalue < bestvalue:
                    bestvalue = tmpvalue
                    bestpoint = point[:]
                    if integrator.mpi_rank == 0:
                        if args.verbose >0:
                            print("new best: ", bestvalue, "point", bestpoint,r,d)
                            sys.stdout.flush()
                else:
                    steps[d] = -steps[d]
                    point[d] += steps[d]
                if d<len(point_min):
                    if point[d] < point_min[d]:
                        point[d] = point_min[d]
                ok = True
                while ok:
                    point[d] += steps[d]
                    bestvalue, tmpvalue = f(bestpoint, point)
                    if integrator.mpi_rank == 0:
                        if args.verbose >0:
                            print(tmpvalue, point)
                    if tmpvalue < bestvalue:
                        bestvalue = tmpvalue
                        bestpoint = point[:]
                        if integrator.mpi_rank == 0:
                            if args.verbose >0:
                                print("new best: ", bestvalue, "point", bestpoint,r,d)
                                sys.stdout.flush()
                    else:
                        ok = False
                        point[d] -= steps[d]
                    if d<len(point_min):
                        if point[d] < point_min[d]:
                            point[d] = point_min[d]
                            ok = False
                    
        for d in range(len(steps)):
            steps[d] *= 0.5
        EVALS *= ROUND_MULT_EVALS
        N_ITER *= ROUND_MULT_N_ITER
        if integrator.mpi_rank == 0:
            calc_p = get_calc_p()
            calc_p['results'] = {'fun':bestvalue, 'x':bestpoint}
            calc_p['last_steps'] = steps
            print('tmp_saved', calc_p)
            with open(output_name + 'tmp.json', 'w') as fp:
                json.dump(calc_p, fp,  indent=4)                
        
    return OptimizeResult({'x':bestpoint, 'fun':bestvalue})

def opt_f(x):
    return do_integrate([0]+x)

def opt_f2(x1,x2):
    return do_integrate2([0]+x1, [0]+x2)

if CHAIN_WF_ONLY_REAL_SPACE_NO_EXCITATIONS:
    wf_params = [0] + CHAIN_RS_PARM + [0]*int(NUM_UNIT_CELLS[0]/2-len(CHAIN_RS_PARM))

start_search =  wf_params[1:] +  ([1] + [0]*(NUM_KOEFS*2-1)) + [S_RELAX]
if old_res is not None:
    start_search = old_res
    if integrator.mpi_rank == 0:
        print('old_res', old_res)

rrr = None
if FP_EVALS > 0:
    fp_res = opt_f(start_search)
    if integrator.mpi_rank == 0:
        rrr = {'fun':fp_res, 'x':start_search}
        print(fp_res, start_search, excitations, 'VP_FACT', VP_FACT, 'DIST_ATOMS', DIST_UNITCELL)
    sys.stdout.flush()

# the second is to check if the results are the same as the previous run
if FP_EVALS > 0:
    fp_res = opt_f(start_search)
    if integrator.mpi_rank == 0:
        rrr = {'fun':fp_res, 'x':start_search}
        print(fp_res, start_search, excitations, 'VP_FACT', VP_FACT, 'DIST_ATOMS', DIST_UNITCELL)
    sys.stdout.flush()

FP_EVALS_OLD = FP_EVALS
PREADAPT_NUM_OLD = PREADAPT_NUM
INT_ALPHA_OLD = INT_ALPHA
N_ITER_OLD = N_ITER
ENABLE_ADAPT_OLD = ENABLE_ADAPT

for ll in CHECK_INTEGRTATION_PRECISION:
    (PREADAPT_NUM, FP_EVALS, INT_ALPHA, N_ITER, ENABLE_ADAPT) = ll
    print("checking integration precision PREADAPT_NUM", PREADAPT_NUM, "FP_EVALS", FP_EVALS, ll)
    fp_res = opt_f(start_search)
    if integrator.mpi_rank == 0:
        rrr = {'fun':fp_res, 'x':start_search}
        print(fp_res, start_search, excitations, 'VP_FACT', VP_FACT, 'DIST_ATOMS', DIST_UNITCELL)
    sys.stdout.flush()

if len(CHECK_INTEGRTATION_PRECISION) > 0:
    print('quiting, because there were precision tests running')
    sys.exit()
    
FP_EVALS = FP_EVALS_OLD
PREADAPT_NUM = PREADAPT_NUM_OLD
INT_ALPHA = INT_ALPHA_OLD
N_INT = N_ITER_OLD
ENABLE_ADAPT = ENABLE_ADAPT_OLD

if EVALS > 0:
    last_log = None # the log is only relevant, if FP_EVALS calculation is done not only additionaly in the beginning
    phase_steps = 0
    if OPTIMIZE_PHASE:
        phase_steps = 0.1

    start_steps = [2]*(NUM_CHAIN_PARAMS) + ([0] + [phase_steps] + [EL_PARAM_STEPS]*(NUM_KOEFS*2-2)) + [DO_RELAX]
    if CHAIN_WF_ONLY_REAL_SPACE_NO_EXCITATIONS:
        start_steps = (numpy.array(CHAIN_RS_PARM) / 10).tolist() + [0]*int(NUM_UNIT_CELLS[0]/2-len(CHAIN_RS_PARM)) + ([0] + [phase_steps] + [EL_PARAM_STEPS]*(NUM_KOEFS*2-2)) + [DO_RELAX]
    if last_steps is not None:
        start_steps = last_steps
        if integrator.mpi_rank == 0:
            print('last_steps', last_steps)
    print("start_steps", start_steps)
    rrr = optimize2(opt_f2, start_search, start_steps, [], iter=OPTIMIZE_ROUNDS)

if integrator.mpi_rank == 0:
    print(rrr)
    calc_p = get_calc_p()
    calc_p['results'] = rrr
    calc_p['fp_res'] = fp_res
    calc_p['last_log'] = last_log
    with open(output_name, 'w') as fp:
        json.dump(calc_p, fp,  indent=4)

# *********************************************************************************************************************************************
# This block is only added to provide a simple run, which shows all effects. For real calculations you might not use DO_AFTER_OPTIMIZATION_EVAL
# *********************************************************************************************************************************************
if DO_AFTER_OPTIMIZATION_EVAL > 0:
    # print koefficients nice
    if not CHAIN_WF_ONLY_REAL_SPACE_NO_EXCITATIONS:
        raise Exception("not implemented")
    ppp =  rrr['x']   
    for i in range(NUM_CHAIN_PARAMS):
        print("{0:3d} {1:5.3f}".format(Atom_Masses[i], ppp[i]))
    ppp = ppp[NUM_CHAIN_PARAMS+2:-1]
    for i in range(len(ppp)):
        if i%2 == 0:
            print("{0:20s} {1:9.3f}".format(str(EL_WV_LIST_DEF[int(i / 2)]), ppp[i]),  end='')
        else:
            print("{0:20s} {1:9.3f}".format("", ppp[i]))

    FP_EVALS = DO_AFTER_OPTIMIZATION_EVAL
    fp_res = opt_f(rrr['x'])
    result1 = fp_res
    if integrator.mpi_rank == 0:
        rrr = {'fun':fp_res, 'x':rrr['x']}
        print(fp_res, rrr['x'], excitations, 'VP_FACT', VP_FACT, 'DIST_ATOMS', DIST_UNITCELL)
        calc_p = get_calc_p()
        calc_p['results'] = rrr
        calc_p['fp_res'] = fp_res
        calc_p['last_log'] = last_log
        with open(output_name + '_lc1_result.json', 'w') as fp:
            json.dump(calc_p, fp,  indent=4)
        sys.stdout.flush()
    
    NOT_SYMMETRIZED = not NOT_SYMMETRIZED

    integrand_H_numba_before = integrand_H_numba
    integrand_wf_numba_before = integrand_wf_numba
    
    # pylint: disable=E0102

    # ************************************************************************************************************************************************************
    # Attention, this part is copied to below, as for FullRun global parameters have to be changed, which makes new creation and compilation of the expressions necessary


    # The next functions are removed from sympy generation, the predefined functions are used instead
    # 
    # v_el_lamb = lambdify([xp], v_el(xp), cse=True, modules=['numpy'])
    # if DoCompile:
    #     v_el_lamb = numba.jit(v_el_lamb, nogil=True)

    # v_lamb = lambdify([xp], v(xp), cse=True, modules=['numpy'])
    # if DoCompile:
    #     v_lamb = numba.jit(v_lamb, nogil=True)


    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # phi_q_el_one is not normalized, therefore the calculation is wrong. This part has to be divided by the phi_q_one_dense_lamb integral ....
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if DO_MEAN:
        phi_q_one_dense_lamb = lambdify([xp, efact], conjugate(phi_q_el_one(xp, N_ELECTRONS, [0,0,0], efact[NUM_CHAIN_PARAMS+1:]) * phi_q_el_one(xp, N_ELECTRONS, [0,0,0], efact[NUM_CHAIN_PARAMS+1:])) , cse=True, modules=['numpy'])    # the last is taken
        if DoCompile:
            phi_q_one_dense_lamb = numba.jit(phi_q_one_dense_lamb, nogil=True)


    phi_full_lamb = lambdify([xp, ee1, efact], phi_full(xp,ee1,efact), cse=True, modules=['numpy'])
    if DoCompile:
        phi_full_lamb = numba.jit(phi_full_lamb, nogil=True)

    tttt = 0
    for i in range(len(Atom_Masses)):
        tttt += diff_q(xp,ee1,efact,i) / Atom_Masses_numpy[i]
    print(time.strftime("diff: %H:%M:%S", time.localtime()))
    energy_q_lamb  = lambdify([xp, ee1, efact], tttt, cse=True, modules=['numpy'])
    print(time.strftime("lamb: %H:%M:%S", time.localtime()))
    if DoCompile:
        energy_q_lamb = numba.jit(energy_q_lamb, nogil=True)
        print(time.strftime("diff_q_lamb: %H:%M:%S", time.localtime()))

    tttt = diff_q_el(xp,ee1,efact)
    print(time.strftime("diff_q_el: %H:%M:%S", time.localtime()))
    diff_q_el_lamb   = lambdify([xp, ee1, efact], tttt, cse=True, modules=['numpy'])
    print(time.strftime("diff_q_el_lamb: %H:%M:%S", time.localtime()))
    if DoCompile:
        diff_q_el_lamb = numba.jit(diff_q_el_lamb, nogil=True)
        print(time.strftime("compile diff_q_el_lamb: %H:%M:%S", time.localtime()))
    #from numpy import conjugate as numpy_conjugate

    #@numba.jit(nopython=True, nogil=True)
    def integrand_H_numba(xp, ee1, ee2, efact, norm_density):
        return numpy.conjugate(phi_full_lamb(xp,ee1,efact)) * ((v_el_lamb(xp,efact)+v_lamb(xp)+v_el_mean_lamb(xp,efact)/norm_density) * phi_full_lamb(xp,ee2,efact) -  energy_q_lamb(xp, ee2, efact) - diff_q_el_lamb(xp, ee2, efact))
    #@numba.jit(nopython=True, nogil=True)
    def integrand_wf_numba(xp, ee1, ee2, efact):
        return phi_full_lamb(xp,ee1,efact)*numpy.conjugate(phi_full_lamb(xp,ee2,efact))
    if DoCompile:
        integrand_H_numba = numba.jit(integrand_H_numba, nogil=True)
        print(time.strftime("integrand_H_numba: %H:%M:%S", time.localtime()))
        integrand_wf_numba = numba.jit(integrand_wf_numba, nogil=True)
        print(time.strftime("integrand_wf_numba: %H:%M:%S", time.localtime()))

    #************************************************************************************************************************************************************

    # pylint: enable=E0102

    FP_EVALS = DO_AFTER_OPTIMIZATION_EVAL
    fp_res = opt_f(rrr['x'])
    result2 = fp_res
    if integrator.mpi_rank == 0:
        rrr = {'fun':fp_res, 'x':rrr['x']}
        print(fp_res, rrr['x'], excitations, 'VP_FACT', VP_FACT, 'DIST_ATOMS', DIST_UNITCELL)
        calc_p = get_calc_p()
        calc_p['results'] = rrr
        calc_p['fp_res'] = fp_res
        calc_p['last_log'] = last_log
        with open(output_name + '_lc2_result.json', 'w') as fp:
            json.dump(calc_p, fp,  indent=4)
        sys.stdout.flush()

        if NOT_SYMMETRIZED:
            t1 = "symmetrized"
            t2 = "not symmetrized"
        else:
            t1 = "not symmetrized"
            t2 = "symmetrized"
        
        print("\n\n\n**************************************************************************************")
        print("RESULTS:")
        print("Energy "+t1+":", result1, "Energy "+t2+":", result2, "Energy difference:", result2-result1)

    
    def do_integrate_simultan(wf_params, int_H1, int_wf1, int_H2, int_wf2):
        global lf, last_log, lf_preadapt, global_integrator
        PREADAPT_STD = math.sqrt(1/2/wf_params[1]) # WIDTH/GAUSS_SMALLER
        start0 = time.time()
        excit = numpy.array(excitations)
        wf_params_mtx = numpy.array(wf_params)
        i_e_v = (numpy.array(INTEGRATE_ELECTRONS_VOLUME)*wf_params_mtx[-1]).tolist()
        if CHECK_MEAN_FIELD_NORMILIZATION:
            # integrate the normal density
            
            # not sure, but every testposition should give the correct density integral
            testpos = numpy.array([(random.random() - 0.5)*0.1 for i in range(NUM_3D_ATOMS_IN_INTEGRATION_RANGE+3*N_ELECTRONS + MEAN_DIM)])
            #@numba.jit(nogil=True) # This is done allways, as this runs usually take very long
            def lf_norm(x):
                x = numpy.concatenate((testpos[:-3],x)) # use the last electron range to integrate the normal density
                return numpy.array(phi_q_one_dense_lamb(x, wf_params_mtx).real)
            lf = lf_norm
            test_value = lf(numpy.array([0.1]*3)) # done, to compile before integration (multithreading might compile in every thread?)
            # test =N(phi_q_el_one([(random.random() - 0.5)*0.1 for i in range(NUM_3D_ATOMS_IN_INTEGRATION_RANGE+3*N_ELECTRONS + 3)], 0, [0,0,0], wf_params_mtx[int(NParticles/2)+1:]))
            if MP_THREADS > 1:
                parallel_integrand = parallelintegrand(batch_from_array)
            else:
                parallel_integrand = instead_batch

            rr = numpy.array(i_e_v)
            global_integrator = vegas.Integrator(rr, nhcube_batch = nhcubes, max_mem = MAX_MEM, alpha=INT_ALPHA, beta=INT_BETA)
            start = time.time()
            rr =  global_integrator(parallel_integrand, nitn=N_ITER, neval=FP_EVALS / 10)

            if global_integrator.mpi_rank == 0:
                if args.verbose >1:
                    print('test value', phi_q_one_dense_lamb(numpy.concatenate((testpos[:-3],numpy.array([0.1]*3))), wf_params_mtx))
                    print('norm integral\n',rr.summary())
            
            norm_d = rr.mean
        norm_d = numpy.array(i_e_v)[:,1].prod()

        @numba.jit(nogil=True) # This is done allways, as this runs usually take very long
        def lf_local(x):
            # norm_d=1
            return numpy.array([int_H1(x, excit, excit, wf_params_mtx, norm_d).real,int_wf1(x, excit, excit, wf_params_mtx).real, int_H2(x, excit, excit, wf_params_mtx, norm_d).real,int_wf2(x, excit, excit, wf_params_mtx).real])
        lf = lf_local
        lf(numpy.array([0.1]*(NUM_3D_ATOMS_IN_INTEGRATION_RANGE+3*N_ELECTRONS + MEAN_DIM))) # done, to compile before integration (multithreading might compile in every thread?)
        
        if MP_THREADS > 1:
            parallel_integrand = parallelintegrand(batch_from_array)
        else:
            parallel_integrand = instead_batch

        rr = NUM_3D_ATOMS_IN_INTEGRATION_RANGE*[[-WIDTH,WIDTH]] + i_e_v * N_ELECTRONS + MEAN_DIM * [[-MEAN_FIELD_DIST, MEAN_FIELD_DIST]]
        map = vegas.AdaptiveMap(rr, ninc=INT_NINC)
        
        if PREADAPT_NUM > 0:
            x_samp = numpy.random.normal(loc=0, scale=PREADAPT_STD, size=(PREADAPT_NUM,NUM_3D_ATOMS_IN_INTEGRATION_RANGE))
            for _ in range(N_ELECTRONS):
                for ll in i_e_v:
                    x_samp = numpy.concatenate([x_samp, numpy.random.uniform(low=ll[0],high=ll[1], size=(PREADAPT_NUM,1))], axis=1)
            if DO_MEAN:
                x_samp = numpy.concatenate([x_samp, numpy.random.normal(loc=0, scale=MEAN_FIELD_STD, size=(PREADAPT_NUM,3))], axis=1)

            @numba.jit()
            def lf_local_vec(xx):
                ret = []
                norm_d = 1
                for x in xx:
                    ret.append(int_H1(x, excit, excit, wf_params_mtx, norm_d).real)
                return numpy.array(ret)
            lf_preadapt = lf_local_vec
            pint = parallelintegrand(lf_preadapt)
            map.adapt_to_samples(x_samp, pint(x_samp), nitn=N_ITER)    
    
        global_integrator = vegas.Integrator(map, nhcube_batch = nhcubes, max_mem = MAX_MEM, alpha=INT_ALPHA, beta=INT_BETA)
        start = time.time()
        if WARMUP_ITER > 0: # this is slower than needed, as one only needs one integrand ... In case it will be used, this might be optimized
            warmup = global_integrator(parallel_integrand, nitn=WARMUP_ITER, neval=FP_EVALS, adapt = ENABLE_ADAPT)
            print('warmup\n',warmup.summary())
        rr =  global_integrator(parallel_integrand, nitn=N_ITER, neval=FP_EVALS, adapt = ENABLE_ADAPT)
        if DO_NUCLEI_POTENTIAL:
            nuclei_pot = pot_nuclei_lamb(None, wf_params_mtx)
        else:
            nuclei_pot = 0
        if global_integrator.mpi_rank == 0:
            last_log = rr.summary() + '\n' + time.strftime("Time: %H:%M:%S", time.localtime()) + " dt" + str(time.time() - start) + " compile and preadapt time " + str(time.time() - start0) + " result " + str(rr) + " " + str(rr[0]/ rr[1])
            if args.verbose >1:
                print(rr.summary())
            if args.verbose >0:
                print(time.strftime("Time: %H:%M:%S", time.localtime()), "dt", time.time()-start, "compile and preadapt time", start-start0, "result", rr, rr[0]/rr[1], rr[2]/rr[3])
                sys.stdout.flush()
                if DO_NUCLEI_POTENTIAL:
                    print("nuclei potential", nuclei_pot)
                    sys.stdout.flush()
        return (rr[0]/rr[1]).mean + nuclei_pot, (rr[2]/rr[3]).mean + nuclei_pot

    if DO_AFTER_SIMULTAN > 0:
        FP_EVALS = DO_AFTER_SIMULTAN
        result_simulatan = do_integrate_simultan([0] + rrr['x'], integrand_H_numba_before, integrand_wf_numba_before, integrand_H_numba, integrand_wf_numba)
        last_log1 = last_log
        rrr1 = result_simulatan
        simultan1 = "result_simultan %s with the difference %f %s" % (str(result_simulatan), result_simulatan[0] - result_simulatan[1], "(t2-t1)")
        print(simultan1)
        print('change sym and nosym')
        result_simulatan = do_integrate_simultan([0] + rrr['x'], integrand_H_numba, integrand_wf_numba, integrand_H_numba_before, integrand_wf_numba_before)
        last_log2 = last_log
        rrr2 = result_simulatan
        simultan2 = "result_simultan %s with the difference %f %s" % (str(result_simulatan), result_simulatan[0] - result_simulatan[1], "(t1-t2)")
        print(simultan2)
        if integrator.mpi_rank == 0:
            calc_p = get_calc_p()
            calc_p['results1'] = rrr1
            calc_p['results2'] = rrr2
            calc_p['last_log1'] = last_log1
            calc_p['last_log2'] = last_log2
            calc_p['simultan1'] = simultan1
            calc_p['simultan2'] = simultan2
            with open(output_name + '_simultan_result.json', 'w') as fp:
                json.dump(calc_p, fp,  indent=4)
