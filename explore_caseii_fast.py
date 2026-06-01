import numpy as np, time, sys
from include import cts, analitical, dynamics
from include.optimizer import _case_II_return_screen, _case_II_validate_after_flyby
from main import ATOL, RTOL
F=dynamics.F
res=analitical.resonance_case_II_estimate(n=1,n_earth=12)
t0=0.0; tf=float(res['t_sim'])
max_step=45*cts.DAY_TO_S
screen_atol=np.array([1e0,1e0,1e-4,1e-4])
screen_rtol=1e-7
# local scan
ths=np.linspace(-0.47,-0.27,31)
dvs=np.linspace(7.262,7.2768,21)
print('N',len(ths)*len(dvs), flush=True)
start=time.time(); screens=[]
for i,th in enumerate(ths):
    for dv in dvs:
        scr=_case_II_return_screen(F, th, dv, res, t0, tf, screen_atol, screen_rtol, max_step,
                                   min_flyby_altitude_km=300, max_flyby_altitude_km=900000,
                                   require_positive_energy=True, return_window_years=1.5)
        if scr:
            screens.append(scr)
    if i%5==0: print('i',i,'screens',len(screens),'time',time.time()-start, flush=True)
print('screens',len(screens),'time',time.time()-start, flush=True)
if not screens: sys.exit()
for name,key,rev in [('fuelguess',lambda s:s['dv_ign'],False),('energy',lambda s:s['delta_energy'],True),('rapo',lambda s:s['r_apo_after'],True),('lowalt',lambda s:s['minimum_altitude'],False)]:
    print('\nTOP',name, flush=True)
    for s in sorted(screens,key=key, reverse=rev)[:8]:
        print(f"th={s['theta']:.6f} dv={s['dv_ign']:.6f} alt={s['minimum_altitude']:.0f} dE={s['delta_energy']:.3f} rapo={s['r_apo_after']/1e6:.1f}", flush=True)
# Validate selected diverse candidates
cand=[]
for key,rev in [(lambda s:s['r_apo_after'],True),(lambda s:s['delta_energy'],True),(lambda s:s['minimum_altitude'],False),(lambda s:s['dv_ign'],False)]:
    cand += sorted(screens,key=key, reverse=rev)[:15]
seen=set(); uniq=[]
for s in cand:
    k=(s['theta'],s['dv_ign'])
    if k not in seen:
        seen.add(k); uniq.append(s)
print('validating',len(uniq), flush=True)
vals=[]
max_step_val=20*cts.DAY_TO_S
for j,s in enumerate(uniq):
    out=_case_II_validate_after_flyby(F,s,t0,tf,ATOL,RTOL,max_step_val,store_full_solution=False)
    if out:
        vals.append(out)
        print(f"VALID th={out['theta']:.6f} dv={out['dv_ign']:.6f} dvfin={out['dv_fin']:.6f} dvtot={out['dv_tot']:.6f} alt={out['minimum_altitude']:.0f} dE={out['delta_energy']:.3f} rapo={out['r_apo_after']/1e6:.1f} tfin={out['t_fin']/cts.YEAR_TO_S:.3f}", flush=True)
print('valid count',len(vals), flush=True)
for out in sorted(vals,key=lambda o:o['dv_tot'])[:20]:
    print(f"BEST th={out['theta']:.6f} dv={out['dv_ign']:.6f} dvfin={out['dv_fin']:.6f} dvtot={out['dv_tot']:.6f} alt={out['minimum_altitude']:.0f} dE={out['delta_energy']:.3f} rapo={out['r_apo_after']/1e6:.1f}", flush=True)
