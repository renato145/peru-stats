import math
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns, altair as alt

FLDS_POPULATION = ['poblacion_total_urbana', 'poblacion_total_rural']

def get_data_series(df, variables, ratio_by=[], ratio_desc='ratio', ratio_mult=1):
    '''
    Format inei data for given variables.

    Parameters
    ----------
    df : pandas.DataFrame
        Inei data.
    variables : str list
        Vars to include in the result.
    ratio_by : str list
        Vars that will divide 'variables' (eg: number of cases / population).
        If more than one is given, they would be summed.
    ratio_desc : str
        Unit to replace field if 'ratio_by' is given.
    ratio_mult : int, float
        Multiply results if 'ratio_by' is given (eg: 'ratio_mult'=1000, if 
        you want cases for each 1000 cases).

    Returns
    -------
    df_out : pandas.DataFrame

    Examples
    --------
    Simple var retrieve:
    >>> inei_series.get_data(df, ['tasa_global_de_fecundidad'])
    
    Var retrieve and get ratios:
    >>> inei_series.get_data(df, ['denuncias_de_violencia_familiar_con_maltrato_psicologico'],
                             ratio_by=['poblacion_total_urbana', 'poblacion_total_rural'])

    Since population may be used regularly, use the shorthand FLDS_POPULATION
    >>> inei_series.get_data(df, ['denuncias_de_violencia_familiar_con_maltrato_psicologico'],
                             ratio_by=inei_series.FLDS_POPULATION)
    '''
    df_out = (df.loc[variables].
                 dropna(axis=1, how='all').
                 set_index(['ambito', 'unidad'], append=True).
                 stack().reset_index())
    df_out.columns = ['indicador', 'departamento', 'unidad', 'y', 'value']
    df_out.y = df_out.y.astype(int)
    
    if len(ratio_by) > 0:
        totals = (get_data_series(df, ratio_by).
                  groupby(['departamento', 'y'])['value'].
                  sum().reset_index().
                  rename({'value': 'total'}, axis=1))
        # extrapolate missing years
        total_years = totals.y.unique()
        missing_years = set(df_out.y.unique()) - set(total_years)
        if len(missing_years) > 0:
            missing_map = {e:total_years[np.argmin(np.abs(total_years - e))] for e in missing_years}
            missing_df = [(e, y, totals.query(f'departamento == {e!r} & y == {missing_map[y]}')['total'].iat[0])
                        for e in totals.departamento.unique() for y in missing_years]
            totals = pd.concat([totals, pd.DataFrame(missing_df, columns=totals.columns)], ignore_index=True)

        df_out = df_out.merge(totals, 'left', ['departamento', 'y']).rename({'value': 'quantity'}, axis=1)
        df_out['value'] = df_out['quantity'] * ratio_mult / df_out['total']
        df_out['unidad'] = ratio_desc

    return df_out

def plot_var_from(df, v, cols=6, fs=(13,6), plot_avg=True, sharey=True):
    avgs = df.loc[v].groupby('y')['value'].mean().reset_index()
    df_t = df.loc[v]
    u = df_t.unidad.iloc[0]
    ys = avgs.y.unique()
    years = ys
    ys = [*ys[::3], ys[-1]]
    ysl = [f'{e}'[2:] for e in ys]
    n = df_t.departamento.nunique()
    rows =  math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, sharey=sharey, sharex=True, figsize=(fs))
    for ax,dep in zip(axes.flatten(), df_t.departamento.unique()):
        df_dep = df_t[df_t.departamento == dep]
        if plot_avg: ax.plot(avgs.y, avgs.value, ':', color='gray')
        ax.plot(df_dep.y, df_dep.value, '-o', markersize=3, alpha=0.75)
        ax.set_title(dep)
        ax.set_xticks(ys)
        ax.set_xticklabels(ysl)
        
    for ax in axes.flatten()[len(df_t.departamento.unique()):]: ax.set_axis_off()

    print(v.title().replace('_', ' '), f'({u}):')
    sns.despine()
    plt.tight_layout()
    plt.show()
    print(f'Years: {", ".join(years.astype(str))}.')
    
def plot_var(df, v, **kwargs):
    ratio_by = kwargs.pop('ratio_by', [])
    ratio_desc = kwargs.pop('ratio_desc', 'ratio')
    ratio_mult = kwargs.pop('ratio_mult', 1)
    plot_var_from(get_data_series(df, v, ratio_by, ratio_desc, ratio_mult).set_index('indicador'), v, **kwargs)
    
def plot_vars(df, vs, **kwargs):
    for i,v in enumerate(vs):
        plot_var(df, v, **kwargs)
        if i < len(vs)-1: print('\n')
