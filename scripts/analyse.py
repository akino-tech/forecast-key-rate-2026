"""Reproduce descriptive long-wave comparison; Python + numpy + pandas."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_json(ROOT/'data/raw/boe-headline-extract.json').set_index('year')
assert df.index.tolist()==list(range(1780,2017))
assert not df.isna().any().any()
df['gdp_per_capita']=df.real_gdp/df.population_thousands*1000
df['wholesale_rebased']=df.wholesale_prices/df.loc[1913,'wholesale_prices']*100
# The full-window centered average has no artificial endpoint padding.
df['price_ma9']=df.wholesale_rebased.rolling(9,center=True,min_periods=9).mean()
t=np.arange(len(df),dtype=float); t=(t-t.mean())/t.std()
for degree in [1,2]:
 y=np.log(df.gdp_per_capita.to_numpy()); trend=np.polyval(np.polyfit(t,y,degree),t)
 df[f'gdp_residual_{degree}']=100*(y-trend)
 df[f'gdp_ma9_{degree}']=df[f'gdp_residual_{degree}'].rolling(9,center=True,min_periods=9).mean()
# Benchmark dates are fixed from the text, not selected after seeing the data.
checks=[]
for kind,year in [('max',1814),('min',1849),('max',1873),('min',1896),('max',1920)]:
 for col in ['wholesale_rebased','price_ma9','gdp_ma9_1','gdp_ma9_2']:
  s=df.loc[year-7:year+7,col].dropna(); extremum=int(s.idxmax() if kind=='max' else s.idxmin())
  checks.append(dict(reference_year=year,kind=kind,series=col,window_start=year-7,window_end=year+7,extremum_year=extremum,shift=extremum-year,boundary=extremum in [year-7,year+7]))
# Local extrema in +/-10-year windows, after 1920, no target 50-year spacing enforced.
peaks={}
for col in ['price_ma9','gdp_ma9_1','gdp_ma9_2']:
 s=df[col].dropna(); maxima=[]; minima=[]
 for year in s.index:
  if year<1930 or year>2002: continue
  z=s.loc[year-10:year+10]
  if len(z)!=21: continue
  if s.loc[year]==z.max(): maxima.append(int(year))
  if s.loc[year]==z.min(): minima.append(int(year))
 peaks[col]={'maxima':maxima,'minima':minima,'peak_intervals':np.diff(maxima).tolist()}
summary={'n':len(df),'period':[1780,2016],'historical_checks':checks,'post1920_extrema':peaks}
(ROOT/'data/processed/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
df.to_csv(ROOT/'data/processed/series.csv',float_format='%.8f')
pd.DataFrame(checks).to_csv(ROOT/'data/processed/turning-points.csv',index=False)
print(json.dumps(summary,ensure_ascii=False,indent=2))
