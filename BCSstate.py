import os, sys

#os.environ['NUMBA_OPT'] = "1"  # not sure if it is not to late to set!!!!

if 'NUMBA_OPT' in os.environ:
    print("numba opt environment ", os.environ['NUMBA_OPT'])
    sys.stdout.flush()

from sympy import lambdify, exp, I, pi, conjugate, diff, re, MatrixSymbol, sqrt, N, Matrix, cos, sin, Abs, symbols
import numba, vegas, time, argparse, numpy
import random
import math
import json

DoCompile = True
output_name = "dump_tmp.json"

NOT_SYMMETRIZED = False
OPTIMIZE_ROUNDS = 3

MP_THREADS = 24
nhcubes = 20000

COMPILE_BEFORE_INTEGRATION_IF_EVAL_IS_MORE = -1 # is leaking memory if turned on

# Parameters supposed to be changed
NParticles = 5
EVALS=4000
FP_EVALS = EVALS *2
DIST_ATOMS = 3
N_ELECTRON_STATES = [1, -1, 0]
excitations = [0,1,0,0,0]
NUM_KOEFS_PER_ELECTRON = NParticles + 1
SHARE_KOEFS = True
OPTIMIZE_PHASE = True # is used wrong at the moment, so False is not different from True ??
EXTEND_INTEGRAL_FACTOR = 1

CHECK_ZERO_KOEF = True

VP_FACT = 1
VE_FACT = 10.0
POTENZ_Vel = 4

DO_AFTER_OPTIMIZATION_EVAL = None
old_res = None
last_steps = None
try:
    with open(sys.argv[1], 'r') as f:
        calc_p = json.load(f)
        print(calc_p)
    NParticles = calc_p['NParticles']
    EVALS = calc_p['EVALS']
    FP_EVALS = EVALS
    if 'FP_EVALS' in calc_p:  # calculate only the results point, usually to be done with much more evals
        FP_EVALS = calc_p['FP_EVALS']
    DIST_ATOMS = calc_p['DIST_ATOMS']
    N_ELECTRON_STATES = calc_p['N_ELECTRON_STATES']
    excitations = calc_p['excitations']
    NUM_KOEFS_PER_ELECTRON = calc_p['NUM_KOEFS_PER_ELECTRON']
    OPTIMIZE_PHASE = calc_p['OPTIMIZE_PHASE']
    if 'results' in calc_p:
        old_res = calc_p['results']['x']
    if 'last_steps' in calc_p:
        last_steps = calc_p['last_steps']
    if 'VP_FACT' in calc_p:
        VP_FACT = calc_p['VP_FACT']
    if 'SHARE_KOEFS' in calc_p:
        SHARE_KOEFS = calc_p['SHARE_KOEFS']
    if 'CHECK_ZERO_KOEF' in calc_p:
        CHECK_ZERO_KOEF = calc_p['CHECK_ZERO_KOEF']
    if 'EXTEND_INTEGRAL_FACTOR' in calc_p:
        EXTEND_INTEGRAL_FACTOR = calc_p['EXTEND_INTEGRAL_FACTOR']
    if 'POTENZ_Vel' in calc_p:
        POTENZ_Vel = calc_p['POTENZ_Vel']
    if 'VE_FACT' in calc_p:
        VE_FACT = calc_p['VE_FACT']
    if 'MP_THREADS' in calc_p:
        MP_THREADS = calc_p['MP_THREADS']
    if 'COMPILE_BEFORE_INTEGRATION_IF_EVAL_IS_MORE' in calc_p:
        COMPILE_BEFORE_INTEGRATION_IF_EVAL_IS_MORE = calc_p['COMPILE_BEFORE_INTEGRATION_IF_EVAL_IS_MORE']
    if 'UNSYMMETIZED' in calc_p:  # old wrong spelling
        NOT_SYMMETRIZED = calc_p['UNSYMMETIZED']
    if 'NOT_SYMMETRIZED' in calc_p:
        NOT_SYMMETRIZED = calc_p['NOT_SYMMETRIZED']
    if 'OPTIMIZE_ROUNDS' in calc_p:
        OPTIMIZE_ROUNDS = calc_p['OPTIMIZE_ROUNDS']
    if 'DO_AFTER_OPTIMIZATION_EVAL' in calc_p:
        DO_AFTER_OPTIMIZATION_EVAL = calc_p['DO_AFTER_OPTIMIZATION_EVAL']
    output_name = sys.argv[1]+"out.json"
    print("loaded from file", sys.argv[1], calc_p)
except NameError:
    print('no file name given')
except IndexError:
    print('no file name given')

def get_calc_p():
    calc_p = {}
    calc_p['NParticles'] = NParticles
    calc_p['EVALS'] = EVALS
    calc_p['DIST_ATOMS'] = DIST_ATOMS
    calc_p['N_ELECTRON_STATES'] = N_ELECTRON_STATES
    calc_p['excitations'] = excitations
    calc_p['NUM_KOEFS_PER_ELECTRON'] = NUM_KOEFS_PER_ELECTRON
    calc_p['OPTIMIZE_PHASE'] = OPTIMIZE_PHASE
    calc_p['VP_FACT'] = VP_FACT
    calc_p['SHARE_KOEFS'] = SHARE_KOEFS
    calc_p['CHECK_ZERO_KOEF'] = CHECK_ZERO_KOEF
    calc_p['EXTEND_INTEGRAL_FACTOR'] = EXTEND_INTEGRAL_FACTOR
    calc_p['WIDTH'] = WIDTH # only used to keep the value for documentation
    calc_p['POTENZ_Vel'] = POTENZ_Vel
    calc_p['VE_FACT'] = VE_FACT
    calc_p['MP_THREADS'] = MP_THREADS
    calc_p['FP_EVALS'] = FP_EVALS
    calc_p['COMPILE_BEFORE_INTEGRATION_IF_EVAL_IS_MORE'] = COMPILE_BEFORE_INTEGRATION_IF_EVAL_IS_MORE
    calc_p['NOT_SYMMETRIZED'] = NOT_SYMMETRIZED
    calc_p['OPTIMIZE_ROUNDS'] = OPTIMIZE_ROUNDS
    calc_p['DO_AFTER_OPTIMIZATION_EVAL'] = DO_AFTER_OPTIMIZATION_EVAL
    return calc_p

# Parameters usualy not changed
Open_End = 0 # if 1, the end is open, if 0, the end is closed
MASS_ATOM = 1000 # in atomic units
SYMMETRIC_WAVEFUNCTION = True

#Parameters calculated from the parameters above, but from experience, calculation may be changed for higher precision e.g.
WIDTH=4 / MASS_ATOM **(1/4)  / VP_FACT ** (1/4)
WIDTH *= EXTEND_INTEGRAL_FACTOR
CALC_OVERLAP = 3

# Parameters calculated from the parameters above
RECIPROCAL_LENGTH = 2*pi/(NParticles * DIST_ATOMS)
N_ELECTRONS = len(N_ELECTRON_STATES)


if SHARE_KOEFS:
    KOEFS_SETS = 1
else:
    KOEFS_SETS = N_ELECTRONS

NUM_KOEFS = NUM_KOEFS_PER_ELECTRON * KOEFS_SETS

print("NParticles: ", NParticles, "WIDTH: ", WIDTH, "EVALS: ", EVALS, "Open_End: ", Open_End, 'OPTIMIZE_PHASE', OPTIMIZE_PHASE)
print(time.strftime("Start time: %H:%M:%S", time.localtime()))
sys.stdout.flush()

rr = NParticles*[[-WIDTH,WIDTH]]
integrator = vegas.Integrator(rr)
print(integrator.mpi_rank)
sys.stdout.flush()

from types import SimpleNamespace
args = SimpleNamespace(**{})
args.verbose = 2

def f(n,k):
    return 1/sqrt(NParticles) * exp(2*pi*I/NParticles*k*n)

def qn(n,Q):
    ret = 0
    for k in range(0,NParticles):
        ret += f(n,k) * Q[k]
    return ret

def Qk(k,q):
    ret = 0
    for n in range(0,NParticles):
        ret += f(n,-k) * q[n]
    return ret

def q(Q):
    ret = []
    for n in range(0,NParticles):
        ret.append(qn(n,Q))
    return Matrix(ret)

def Q(q):
    ret = []
    for k in range(0,NParticles):
        ret.append(Qk(k,q))
    return Matrix(ret)

def v(q):
    ret = 0
    for n in range(NParticles - Open_End):
        ret += (q[n]-q[(n+1)%NParticles])**2
    return VP_FACT * ret

def v_el(q): # the last q's are for the positions of the electrons
    ret = 0
    for ne in range(N_ELECTRONS):
        for n in range(-CALC_OVERLAP, NParticles + CALC_OVERLAP):
            ret += exp(-(q[n%NParticles] + n * DIST_ATOMS - q[NParticles + ne])**POTENZ_Vel)
    return VE_FACT * ret

def phi_q_el_one(xp, ne, me, koefs):
    QP = Q(xp)
    if CHECK_ZERO_KOEF:
        QP[0] = 1
    if SHARE_KOEFS:
        ne_tmp = 0
    else:
        ne_tmp = ne
    if OPTIMIZE_PHASE:
        ret = exp(I * koefs[2*NUM_KOEFS_PER_ELECTRON*ne_tmp + 2*0 + 1])
    else:
        ret = 1
    for n in range(1, NUM_KOEFS_PER_ELECTRON):
        ret += QP[n % NParticles] * (koefs[2*NUM_KOEFS_PER_ELECTRON*ne_tmp + 2*n] + I * koefs[2*NUM_KOEFS_PER_ELECTRON*ne_tmp + 2*n + 1] * exp(I * RECIPROCAL_LENGTH * n * xp[NParticles+ne])) 
    return ret * exp(I * RECIPROCAL_LENGTH * me * xp[NParticles+ne])

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
        raise('slater matrix with wrong number of electrons')
    return getMatrixDeternminant(slater_matrix)  

def diff_q_el(xp, ee, efact):
    ret = 0
    for n in range(NParticles, NParticles+N_ELECTRONS):
        ret += diff(phi_full(xp, ee, efact), xp[n],2)
    return ret

def diff_q(xp, ee, efact):
    ret = 0
    for n in range(NParticles):
        ret += diff(phi_full(xp, ee, efact), xp[n],2)
    return ret

def phi_Q(Q,ee,efact):
    prod = 1
    for k in range(1,NParticles):
        ki = k
        if k > int(NParticles/2):
            ki = NParticles - k
        prod *= exp(-efact[ki] / 2 * Q[k]*conjugate(Q[k])) * (ee[k] * Q[k]  + (1 - ee[k])) 
    return prod

def phi_q(q, ee, efact):
    return phi_Q(Q(q), ee,efact)

def phi_full(q, ee, efact):
    return phi_q(q, ee, efact) * phi_q_el_slater(q, efact[int(NParticles/2)+1:])

xp_symbols = list(symbols(''.join(' xp'+str(i) for i in range(NParticles+N_ELECTRONS)), real=True))
ee1_symbols = list(symbols(''.join(' ee1'+str(i) for i in range(NParticles)), real=True))
ee2_symbols = list(symbols(''.join(' ee2'+str(i) for i in range(NParticles)), real=True))
efact_symbols = list(symbols(''.join(' efact'+str(i) for i in range(int(NParticles/2)+1 + NUM_KOEFS*2)), real=True))

xp = Matrix(xp_symbols) # MatrixSymbol('xp',NParticles,1)
ee1 = Matrix(ee1_symbols) # ee1 = MatrixSymbol('eea',NParticles,1) 
ee2 = Matrix(ee2_symbols) # ee2 = MatrixSymbol('eeb',NParticles,1) 

efact = Matrix(efact_symbols) # efact = MatrixSymbol('efact',int(NParticles/2)+1 + NUM_KOEFS*2,1)

# ************************************************************************************************************************************************************
# Attention, this part is copied to below, as for FullRun global parameters have to be changed, which makes new creation and compilation of the expressions necessary
integrand_wf_lambdify = lambdify([xp, ee1, ee2, efact], phi_full(xp,ee1,efact)*conjugate(phi_full(xp,ee2,efact)), cse=True, modules=['numpy'])
integrand_H_lambdify = lambdify([xp, ee1, ee2, efact], conjugate(phi_full(xp,ee1,efact)) * ((v_el(xp)+v(xp)) * phi_full(xp,ee2,efact) - 1.0 / MASS_ATOM * diff_q(xp, ee2, efact) - diff_q_el(xp, ee2, efact)), cse=True, modules=['numpy']) # at both terms the factor 1/2 is left out

def integrand_wf_numba(xp, ee1, ee2, efact):
    return [integrand_wf_lambdify(xp, ee1, ee2, efact)[0,0]]
def integrand_H_numba(xp, ee1, ee2, efact):
    return [integrand_H_lambdify(xp, ee1, ee2, efact)[0,0]]

if DoCompile:
    integrand_wf_numba = numba.jit(integrand_wf_lambdify, nogil=True) #, fastmath = True, nogil=True)
    integrand_H_numba = numba.jit(integrand_H_lambdify, nogil=True) #, fastmath = True, nogil=True)

#************************************************************************************************************************************************************

import multiprocessing
class parallelintegrand(vegas.BatchIntegrand):
    """ Convert integrand into multiprocessor integrand.
    Integrand can only handle a single input !!!!
    Integrand should return a numpy array.
    """
    def __init__(self, fcn, nproc=MP_THREADS):
        " Save integrand; create pool of nproc processes. "
        self.fcn = fcn
        self.nproc = nproc
        self.pool = multiprocessing.Pool(processes=nproc)

    def __del__(self):
        " Standard cleanup. "
        self.pool.close()
        self.pool.join()

    def __call__(self, x):
        " Divide x into chunks feeding one to each process. "
        chunks = self.nproc * 10
        nx = x.shape[0] // chunks + 1
        num_chanks = min(chunks, x.shape[0])
        # launch evaluation of self.fcn for each chunk, in parallel
        results = self.pool.map(
            self.fcn,
            numpy.array_split(x, num_chanks), 
            )
        return numpy.concatenate(results)

wf_params = []
for i in range(int(NParticles/2)+1):
    wf_params.append(math.sqrt(MASS_ATOM) * math.sqrt(VP_FACT) * float(sqrt(2*(1-cos(2*pi/NParticles * i)))))

def integrand_wf_debug(xp, ee1, ee2, efact):
    r =integrand_wf_numba(xp, ee1, ee2,efact)
    return r


while len(excitations) < int(NParticles/2)+1:
    excitations.append(0)

# here to compile the integrands
test_list = [(random.random() - 0.5) for i in range(NParticles+N_ELECTRONS)]
res_test=integrand_H_numba(numpy.array(test_list).T, numpy.array(excitations), numpy.array(excitations), numpy.array(wf_params+[1]+[0.1]*(NUM_KOEFS*2-1)))
print(res_test)
print(integrand_wf_numba(numpy.array(test_list).T, numpy.array(excitations), numpy.array(excitations), numpy.array(wf_params+[1]+[0.1]*(NUM_KOEFS*2-1))))
print(time.strftime("Time: %H:%M:%S", time.localtime()))
sys.stdout.flush()

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

last_log = None
def do_integrate(wf_params):
    global lf, last_log
    start0 = time.time()
    excit = numpy.array(excitations)
    wf_params_mtx = numpy.array(wf_params)
    @numba.jit(nogil=True) # This is done allways, as this runs usually take very long
    def lf_local(x):
        return numpy.array([integrand_H_numba(x, excit, excit, wf_params_mtx).real,integrand_wf_numba(x, excit, excit, wf_params_mtx).real])
    lf = lf_local
    lf(numpy.array([0.1]*(NParticles+N_ELECTRONS))) # done, to compile before integration (multithreading might compile in every thread?)
    
    if MP_THREADS > 1:
        parallel_integrand = parallelintegrand(batch_from_array)
    else:
        parallel_integrand = instead_batch

    rr = NParticles*[[-WIDTH,WIDTH]] + [[0, NParticles*DIST_ATOMS]] * N_ELECTRONS
    integrator = vegas.Integrator(rr, nhcube_batch = nhcubes)
    start = time.time()
    rr =  integrator(parallel_integrand, nitn=10, neval=FP_EVALS)
    if integrator.mpi_rank == 0:
        last_log = rr.summary() + '\n' + time.strftime("Time: %H:%M:%S", time.localtime()) + " dt" + str(time.time() - start) + " compile time " + str(time.time() - start0) + " result " + str(rr) + " " + str(rr[0]/ rr[1])
        if args.verbose >1:
            print(rr.summary())
        if args.verbose >0:
            print(time.strftime("Time: %H:%M:%S", time.localtime()), "dt", time.time()-start, "compile time", start-start0, "result", rr, rr[0]/rr[1])
            sys.stdout.flush()
    return (rr[0]/rr[1]).mean

def do_integrate2(wf_params1, wf_params2):
    """
    the integration for two points is done simultaniously using vegas feature to take the same sampleing points for multiple integrands
    used for H and wf integration at two different wf_params
    """
    global lf
    start0 = time.time()
    excit = numpy.array(excitations)
    wf_params1_mtx = numpy.array(wf_params1)
    wf_params2_mtx = numpy.array(wf_params2)
    if COMPILE_BEFORE_INTEGRATION_IF_EVAL_IS_MORE > 0 and EVALS >= COMPILE_BEFORE_INTEGRATION_IF_EVAL_IS_MORE:
        @numba.jit(nogil=True)
        def lf_local(x):
            return numpy.array([integrand_H_numba(x, excit, excit, wf_params1_mtx).real,integrand_wf_numba(x, excit, excit, wf_params1_mtx).real,
                                    integrand_H_numba(x, excit, excit, wf_params2_mtx).real,integrand_wf_numba(x, excit, excit, wf_params2_mtx).real])
        lf_local(numpy.array([0.1]*(NParticles+N_ELECTRONS))) # done, to compile before integration (multithreading might compile in every thread?)
    else:
        def lf_local(x):
            return numpy.array([integrand_H_numba(x, excit, excit, wf_params1_mtx).real,integrand_wf_numba(x, excit, excit, wf_params1_mtx).real,
                                    integrand_H_numba(x, excit, excit, wf_params2_mtx).real,integrand_wf_numba(x, excit, excit, wf_params2_mtx).real])

    lf = lf_local
    if MP_THREADS > 1:
        parallel_integrand = parallelintegrand(batch_from_array)
    else:
        parallel_integrand = instead_batch

    rr = NParticles*[[-WIDTH,WIDTH]] + [[0, NParticles*DIST_ATOMS]] * N_ELECTRONS
    integrator = vegas.Integrator(rr, nhcube_batch = nhcubes)
    start = time.time()
    rr =  integrator(parallel_integrand, nitn=10, neval=EVALS)
    if integrator.mpi_rank == 0:
        if args.verbose >1:
            print(rr.summary())
        if args.verbose >0:
            print(time.strftime("Time: %H:%M:%S", time.localtime()), "dt", time.time()-start, "compile time", start-start0, "result", rr, rr[0]/rr[1], rr[2]/rr[3])
            sys.stdout.flush()
    return (rr[0]/rr[1]).mean, (rr[2]/rr[3]).mean

from scipy.optimize import OptimizeResult, minimize

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
    """
    optimizing Monte Carlo results is quite difficult, as they have statistical noise.
    Therefore, we use a simple optimization algorithm to find the best parameters.
    To check if a point is better than an earlier we use integrate2, which does integration at both points using the same sample points
    """
    global EVALS
    point = point_in[:]
    #f(point,point) # pretrain the integrator
    bestvalue, bestvalue = f(point,point) # does the pretraining, not used later
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
        EVALS *= 4
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

start_search =  wf_params[1:] +  ([1] + [0]*(NUM_KOEFS_PER_ELECTRON*2-1))*KOEFS_SETS
if old_res is not None:
    start_search = old_res
    if integrator.mpi_rank == 0:
        print('old_res', old_res)

rrr = None
if FP_EVALS > 0:
    fp_res = opt_f(start_search)
    if integrator.mpi_rank == 0:
        rrr = {'fun':fp_res, 'x':start_search}
        print(fp_res, start_search, excitations, 'VP_FACT', VP_FACT, 'DIST_ATOMS', DIST_ATOMS)
    sys.stdout.flush()

if FP_EVALS <= EVALS:
    last_log = None # the log is only relevant, if FP_EVALS calculation is done not only additionaly in the beginning
    phase_steps = 0
    if OPTIMIZE_PHASE:
        phase_steps = 0.1

    start_steps = [2]*(int(NParticles/2)) + ([0] + [phase_steps] + [0.1]*(NUM_KOEFS_PER_ELECTRON*2-2)) * KOEFS_SETS
    if last_steps is not None:
        start_steps = last_steps
        if integrator.mpi_rank == 0:
            print('last_steps', last_steps)

    rrr = optimize2(opt_f2, start_search, start_steps, [0.01,0.01,0.01,0.01], iter=OPTIMIZE_ROUNDS)

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
if DO_AFTER_OPTIMIZATION_EVAL is not None:
    FP_EVALS = DO_AFTER_OPTIMIZATION_EVAL
    fp_res = opt_f(rrr['x'])
    result1 = fp_res
    if integrator.mpi_rank == 0:
        rrr = {'fun':fp_res, 'x':rrr['x']}
        print(fp_res, rrr['x'], excitations, 'VP_FACT', VP_FACT, 'DIST_ATOMS', DIST_ATOMS)
        calc_p = get_calc_p()
        calc_p['results'] = rrr
        calc_p['fp_res'] = fp_res
        calc_p['last_log'] = last_log
        with open(output_name + '_lc1_result.json', 'w') as fp:
            json.dump(calc_p, fp,  indent=4)
        sys.stdout.flush()
    

    NOT_SYMMETRIZED = not NOT_SYMMETRIZED
    
    # ************************************************************************************************************************************************************
    # this has to be done, as a global variable has changed !!!!
    integrand_wf_lambdify = lambdify([xp, ee1, ee2, efact], phi_full(xp,ee1,efact)*conjugate(phi_full(xp,ee2,efact)), cse=True, modules=['numpy'])
    integrand_H_lambdify = lambdify([xp, ee1, ee2, efact], conjugate(phi_full(xp,ee1,efact)) * ((v_el(xp)+v(xp)) * phi_full(xp,ee2,efact) - 1.0 / MASS_ATOM * diff_q(xp, ee2, efact) - diff_q_el(xp, ee2, efact)), cse=True, modules=['numpy']) # at both terms the factor 1/2 is left out

    def integrand_wf_numba(xp, ee1, ee2, efact):
        return [integrand_wf_lambdify(xp, ee1, ee2, efact)[0,0]]
    def integrand_H_numba(xp, ee1, ee2, efact):
        return [integrand_H_lambdify(xp, ee1, ee2, efact)[0,0]]

    if DoCompile:
        integrand_wf_numba = numba.jit(integrand_wf_lambdify, nogil=True) #, fastmath = True, nogil=True)
        integrand_H_numba = numba.jit(integrand_H_lambdify, nogil=True) #, fastmath = True, nogil=True)
    # ************************************************************************************************************************************************************
    
    FP_EVALS = DO_AFTER_OPTIMIZATION_EVAL
    fp_res = opt_f(rrr['x'])
    result2 = fp_res
    if integrator.mpi_rank == 0:
        rrr = {'fun':fp_res, 'x':rrr['x']}
        print(fp_res, rrr['x'], excitations, 'VP_FACT', VP_FACT, 'DIST_ATOMS', DIST_ATOMS)
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
    