import numpy as np, time
from include import cts, analitical, dynamics
from include.optimizer import _case_II_return_screen, _case_II_validate_after_flyby
from main import ATOL, RTOL

F=dynamics.F
res=analitical.resonance_case_II_estimate(n=1,n_earth=12)
t0=0.0; tf=float(res['t_sim'])
nstep=1200
max_step=min((tf-t0)/nstep,20*cts.DAY_TO_S)
screen_atol=np.maximum(np.asarray(ATOL), np.array([1e-1,1e-1,1e-5,1e-5]))
screen_rtol=max(RTOL,1e-8)

# broader scan around useful theta and dv
ths=np.linspace(-0.55,-0.22,67)
dvs=np.linspace(float(res['dv_ign'])-0.008, min(float(analitical.deltaV_ignI)-5e-4, float(res['dv_ign'])+0.006), 29)
print('theta range',ths[0],ths[-1],'dv',dvs[0],dvs[-1],'N',len(ths)*len(dvs))
start=time.time(); screens=[]
for i,th in enumerate(ths):
    for dv in dvs:
        scr=_case_II_return_screen(F, th, dv, res, t0, tf, screen_atol, screen_rtol, max_step,
                                   min_flyby_altitude_km=300, max_flyby_altitude_km=900000,
                                   require_positive_energy=True, return_window_years=1.0)
        if scr:
            screens.append(scr)
print('screens',len(screens),'time',time.time()-start)
if screens:
    # print summaries by altitude bins
    screens_sorted=sorted(screens,key=lambda s:s['dv_ign'])
    print('min/max alt',min(s['minimum_altitude'] for s in screens), max(s['minimum_altitude'] for s in screens))
    # top r_apo / energy
    for name,key,rev in [('energy',lambda s:s['delta_energy'],True),('rapo',lambda s:s['r_apo_after'],True),('lowalt',lambda s:s['minimum_altitude'],False)]:
        print('\nTOP',name)
        for s in sorted(screens,key=key, reverse=rev)[:10]:
            print(f"th={s['theta']:.6f} dv={s['dv_ign']:.6f} alt={s['minimum_altitude']:.0f} dE={s['delta_energy']:.3f} rapo={s['r_apo_after']/1e6:.1f}Gm")
    # validate top candidates from mixed criteria
    cand=[]
    for sortkey in [lambda s:s['r_apo_after'], lambda s:s['delta_energy'], lambda s:-s['minimum_altitude'], lambda s:-s['dv_ign']]:
        cand += sorted(screens,key=sortkey, reverse=True)[:30]
    # unique by rounded th/dv
    uniq=[]; seen=set()
    for s in cand:
        k=(round(s['theta'],9), round(s['dv_ign'],9))
        if k not in seen:
            seen.add(k); uniq.append(s)
    print('validating',len(uniq))
    vals=[]; start=time.time()
    for s in uniq:
        out=_case_II_validate_after_flyby(F, s, t0, tf, ATOL, RTOL, max_step, store_full_solution=False)
        if out:
            vals.append(out)
            print(f"VALID th={out['theta']:.6f} dv={out['dv_ign']:.6f} dvfin={out['dv_fin']:.6f} dvtot={out['dv_tot']:.6f} alt={out['minimum_altitude']:.0f} dE={out['delta_energy']:.3f} rapo={out['r_apo_after']/1e6:.1f} tfin={out['t_fin']/cts.YEAR_TO_S:.3f}")
    print('valid count',len(vals),'time',time.time()-start)
    if vals:
        print('\nBEST by dv_tot')
        for out in sorted(vals,key=lambda o:o['dv_tot'])[:20]:
            print(f"th={out['theta']:.6f} dv={out['dv_ign']:.6f} dvfin={out['dv_fin']:.6f} dvtot={out['dv_tot']:.6f} alt={out['minimum_altitude']:.0f} dE={out['delta_energy']:.3f} rapo={out['r_apo_after']/1e6:.1f} tfin={out['t_fin']/cts.YEAR_TO_S:.3f}")
