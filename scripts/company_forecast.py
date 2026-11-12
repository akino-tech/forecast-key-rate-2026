"""Financial ratios and explicitly hypothetical scenarios; no market-price estimates."""
from pathlib import Path
import json
import pandas as pd
R=Path(__file__).resolve().parents[1];D=R/'data/processed'
s=json.loads((R/'data/raw/nvidia-financials.json').read_text()); p=pd.DataFrame(s['periods']).set_index('period')
assert (p.revenue-p.cost_revenue==p.gross_profit).all()
assert (p.gross_profit-p.opex==p.operating_income).all()
for numerator in ['gross_profit','operating_income','net_income','rnd','opex']:p[numerator+'_pct']=100*p[numerator]/p.revenue
p.to_csv(D/'nvidia-ratios.csv',float_format='%.6f')
a,b=p.iloc[0],p.iloc[1]
ratios={'revenue_growth_pct':100*(b.revenue/a.revenue-1),'op_income_growth_pct':100*(b.operating_income/a.operating_income-1),'rnd_growth_pct':100*(b.rnd/a.rnd-1),'incremental_op_margin_pct':100*(b.operating_income-a.operating_income)/(b.revenue-a.revenue),'dc_share_pct':100*89000/b.revenue}
# Stress: all cost of revenue proportional, operating expense fixed. Not a prediction.
gm=b.gross_profit/b.revenue
stress=[]
for loss in [0,.1,.2,.3]:
 rev=b.revenue*(1-loss);oi=rev*gm-b.opex
 stress.append({'revenue_change_pct':-loss*100,'revenue_USDm':rev,'op_income_USDm':oi,'op_income_change_pct':100*(oi/b.operating_income-1)})
pd.DataFrame(stress).to_csv(D/'nvidia-stress.csv',index=False,float_format='%.6f')
# Hypothetical bundle of capacity, not per GPU: capex 200k, fixed annual opex 30k,
# 3-year life, zero residual, 15% discount, 1 USD variable cost per capacity-hour.
r=.15;n=3;annual_capital=200000*r/(1-(1+r)**(-n));fixed=annual_capital+30000
rows=[]
for u in [.3,.6,.9]:rows.append({'utilization':u,'annual_hours':8760*u,'annual_fixed_USD':fixed,'unit_cost_USD_per_capacity_hour':fixed/(8760*u)+1})
pd.DataFrame(rows).to_csv(D/'tco-scenarios.csv',index=False,float_format='%.6f')
# Probabilistic scenario model defined after forecasts, not historical frequencies.
scenarios=[('Disinflation',.30,.75),('Sticky',.50,.20),('Shock',.20,.125)]
q=sum(w*c for _,w,c in scenarios); assert abs(q-.35)<1e-12
pd.DataFrame(scenarios,columns=['scenario','weight_assumed','cut_probability_assumed']).to_csv(D/'rate-scenarios.csv',index=False)
# Hypothetical likelihood ratios applied once to the original prior; avoid double counting.
updates=[]
for prior in [.7,.35]:
 for lr in [.25,.5,1,2,4]:updates.append({'prior':prior,'likelihood_ratio_assumed':lr,'posterior':prior*lr/(1-prior+prior*lr)})
pd.DataFrame(updates).to_csv(D/'forecast-sensitivity.csv',index=False,float_format='%.6f')
(R/'data/processed/company-model-summary.json').write_text(json.dumps({'ratios':ratios,'tco_annual_capital_USD':annual_capital,'tco_annual_fixed_USD':fixed,'scenario_cut_probability':q},indent=2))
print(p.round(3).to_string());print(json.dumps(ratios,indent=2));print(pd.DataFrame(stress).round(3).to_string(index=False));print(pd.DataFrame(rows).round(3).to_string(index=False))
