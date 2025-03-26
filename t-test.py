import scipy.stats as stats
import numpy as np
import pingouin as pg

ppl_baseline =  37.00037384033203
ppl_tg = 7.546139881406414
ppl_tg1 = 7.6360916124500875
ppl_tg2 = 7.91510995785016

#print(np.var(dist3_baseline)/np.var(dist3_tg2))
t_stat = pg.ttest(ppl_baseline, ppl_tg, correction=True) # The ratio of dist1_baseline to dist1_tg is less that 4:1
t_crit = stats.t.ppf(q=0.025, df=5.616978)
print(t_crit)