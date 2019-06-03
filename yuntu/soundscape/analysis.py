from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

def GLMM(df, out, fixedEff, randIntercept, randSlope=[],family='Gamma(link="inverse")'):
    base = importr('base')
    stats = importr('stats')
    lme4 = importr('lme4')

    variabelen = out+ ' ~ ' + '+'.join(fixedEff)
    intercept = ['(1|'+i+')' for i in randIntercept]
    intercept = '+'.join(intercept)
    slope = ['(1+'+fixedEff[0]+'|'+i+')' for i in randSlope]
    slope = ' '.join(slope)

    pandas2ri.activate()
    robjects.globalenv["dataframe"] = df

    #model
    #random intercepts + random slopes
    if len(randSlope) > 0:
        formula = variabelen + '+' + intercept + '+' + slope

    else:
        formula = variabelen + '+' + intercept
    
    print(formula)

    model = lme4.glmer(formula,data=base.as_symbol('dataframe'),family=family)
    coeffs = base.summary(model).rx2('coefficients')
    summary = base.summary(model)

    return summary,coeffs