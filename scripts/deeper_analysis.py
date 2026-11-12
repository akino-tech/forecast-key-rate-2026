"""Reproducible sensitivity, spectrum and AR(1) Monte Carlo. No network."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]; D=R/'data/processed'
a=pd.read_json(R/'data/raw/boe-headline-extract.json').set_index('year')
logs={'gdp_pc':np.log(a.real_gdp/a.population_thousands*1000), 'prices':np.log(a.wholesale_prices)}
# Descriptive growth, geometric annual rate and volatility of annual log growth.
rows=[]
for name,y in logs.items():
 for lo,hi in [(1780,1913),(1913,1945),(1945,1973),(1973,2016),(1780,2016)]:
  g=100*y.loc[lo:hi].diff().dropna()
  rows.append(dict(series=name,start=lo,end=hi,years=hi-lo,cagr_pct=100*np.expm1((y.loc[hi]-y.loc[lo])/(hi-lo)),log_growth_sd=g.std(ddof=1)))
pd.DataFrame(rows).to_csv(D/'growth-regimes.csv',index=False,float_format='%.6f')
# Do the five target turns survive degree/window changes? A descriptive selection check.
rows=[]
for name,y in logs.items():
 t=np.linspace(-1,1,len(y))
 for degree in [1,2,3]:
  resid=pd.Series(100*(y.to_numpy()-np.polyval(np.polyfit(t,y,degree),t)),index=y.index)
  for w in [5,9,13]:
   z=resid.rolling(w,center=True,min_periods=w).mean()
   for kind,yr in [('max',1814),('min',1849),('max',1873),('min',1896),('max',1920)]:
    win=z.loc[yr-7:yr+7]; e=int(win.idxmax() if kind=='max' else win.idxmin())
    rows.append(dict(series=name,degree=degree,window=w,reference=yr,extremum=e,shift=e-yr,boundary=e in (yr-7,yr+7)))
pd.DataFrame(rows).to_csv(D/'robustness-turns.csv',index=False)
# Spectrum: demean, Hann taper, squared FFT, exclude zero. Frequencies in cycles/year.
def spec(x):
 x=np.asarray(x); x=x-x.mean(axis=-1,keepdims=True)
 power=abs(np.fft.rfft(x*np.hanning(x.shape[-1]),axis=-1))**2
 f=np.fft.rfftfreq(x.shape[-1]); mask=(f>0)&(f>=1/70)&(f<=1/40)
 share=power[...,mask].sum(axis=-1)/power[...,1:].sum(axis=-1)
 peak=1/f[1:][np.argmax(power[...,1:],axis=-1)]
 return share,peak,f,power,mask
spectra=[]; powers=[]; mc=[]; rng=np.random.default_rng(20261112); B=5000
for name,y in logs.items():
 t=np.linspace(-1,1,len(y))
 for degree in [1,2,3]:
  x=y.to_numpy()-np.polyval(np.polyfit(t,y,degree),t)
  sh,pk,f,p,m=spec(x)
  spectra.append(dict(series=name,transform=f'log_detrended_{degree}',n=len(x),band_share_pct=float(100*sh),dominant_period=float(pk),band_bins=int(m.sum())))
  for freq,powr in zip(f[1:],p[1:]):powers.append(dict(series=name,transform=f'log_detrended_{degree}',period=1/freq,power_share_pct=100*powr/p[1:].sum()))
 x=np.diff(y.to_numpy()); sh,pk,f,p,m=spec(x)
 spectra.append(dict(series=name,transform='log_growth',n=len(x),band_share_pct=float(100*sh),dominant_period=float(pk),band_bins=int(m.sum())))
 for freq,powr in zip(f[1:],p[1:]):powers.append(dict(series=name,transform='log_growth',period=1/freq,power_share_pct=100*powr/p[1:].sum()))
 # Null: stationary Gaussian AR(1), phi via demeaned lag regression; scale cancels in share.
 xc=x-x.mean(); phi=float(np.dot(xc[:-1],xc[1:])/np.dot(xc[:-1],xc[:-1])); assert abs(phi)<1
 sims=np.empty((B,len(x))); sims[:,0]=rng.normal(size=B)/np.sqrt(1-phi**2)
 for i in range(1,len(x)):sims[:,i]=phi*sims[:,i-1]+rng.normal(size=B)
 simulated=spec(sims)[0]; pval=(1+int((simulated>=sh).sum()))/(B+1)
 mc.append(dict(series=name,phi=phi,observed_share_pct=float(100*sh),null_median_pct=float(100*np.median(simulated)),null_q95_pct=float(100*np.quantile(simulated,.95)),p_upper=pval,simulations=B,seed=20261112))
pd.DataFrame(spectra).to_csv(D/'spectral-summary.csv',index=False,float_format='%.6f')
pd.DataFrame(powers).to_csv(D/'spectral-density.csv',index=False,float_format='%.8f')
pd.DataFrame(mc).to_csv(D/'spectral-null.csv',index=False,float_format='%.6f')
print('GROWTH\n',pd.read_csv(D/'growth-regimes.csv').round(3).to_string(index=False))
print('SPECTRA\n',pd.DataFrame(spectra).round(3).to_string(index=False))
print('NULL\n',pd.DataFrame(mc).round(4).to_string(index=False))
print('ROBUSTNESS\n',pd.DataFrame(rows).groupby(['series','reference']).agg(min_year=('extremum','min'),max_year=('extremum','max'),boundary_count=('boundary','sum')).to_string())
