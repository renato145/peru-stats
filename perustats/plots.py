import math
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns, altair as alt
from . import utils

def _build_slope(df, values, time, bars, col, text, filter_in,
                 y_pos=10, width=350, height=400, y_title=None):
    base = alt.Chart(df)

    if time is None:
        slope_x_title = ''
        slope_filter = {'or': [f'datum.slope_x == "averages"', filter_in.ref()]}
    else:
        max_time = df[time].max()
        slope_x_title = f'{max_time}'
        slope_filter = {'and':
            [{'or': [f'datum.slope_x == "averages"', filter_in.ref()]},
             f'datum.{time} == {max_time}']}
    
    slope_base = base.encode(
        x=alt.X('slope_x:N', title=slope_x_title,
                axis=alt.Axis(labels=False),
                scale=alt.Scale(domain=['averages', 'measures', ''])),
        y=alt.Y(values, title=y_title,
                scale=alt.Scale(domain=[df[values].min(), df[values].max()])),
        color=alt.Color(col, legend=None)
    ).transform_filter(
        slope_filter
    ).properties(
        width=width, height=height
    )
    
    slope_points = slope_base.mark_point(filled=True, size=150)
    slope_lines = slope_base.mark_line()

    slope_text_1 = slope_base.mark_text(dx=-25, dy=5).encode(
        text=values
    ).transform_filter(
        'datum.slope_x == "averages"'
    )

    slope_text_2 = slope_base.mark_text(align='left', dx=8, dy=5).encode(
        text='slope_text:N'
    ).transform_filter(
        'datum.slope_x == "measures"'
    )

    slope_title = slope_base.mark_text(size=14, dy=-15).encode(
        y=alt.value(y_pos),
        text=bars,
        color=alt.ColorValue('black')
    )

    return slope_points + slope_lines + slope_text_1 + slope_text_2 + slope_title

def tl_summary(df, values, time, bars, col, text,
               bars_w=810, bars_h=200,
               timeline_w=450, timeline_h=200,
               slope_avg='Average', slope_w=300, slope_h=200, slope_y_pos=10,
               palette='tableau10'):
    '''
    Plots 3 charts: bars, timeline and slopegraph

    Parameters
    ----------
    df : pandas.DataFrame
    values : str
        Name of the column used for values.
    time : str
        Name of the column used for time values.
    bars : str
        Name of the column used to plot as X-axis on the bars.
    col : str
        Name of the column used for colors.
    text : str
        Name of the column used to show text on slopegraph.
    bars_w : int
        Bars plot width.
    bars_h : int
        Bars plot height.
    timeline_w : int
        Timeline plot width.
    timeline_h : int
        Timeline plot height.
    slope_avg : str
        Title for the avg measures on slopegraph.
    slope_w : int
        Slopegraph plot width.
    slope_h : int
        Slopegraph plot height.
    slope_y_pos : int
        Slopegraph titles position.
    palette : str
        Check https://vega.github.io/vega/docs/schemes/#reference

    Returns
    -------
        altair.Chart
    '''
    df = df.copy()
    df['slope_x'] = 'measures'
    df_avg = df.groupby([col, time]).mean().reset_index()
    df_avg[bars] = slope_avg
    df_avg['slope_x'] = 'averages'
    df = pd.concat([df, df_avg], ignore_index=True, sort=True)
    df[values] = df[values].round(2)
    df['slope_text'] = df[values].astype(str) + ' ' + df[col]

    max_time = df[time].max()
    filter_in = alt.selection_single(fields=[bars], on='mouseover', empty='none')

    base = alt.Chart(df)
    barsplot = base.mark_bar().encode(
        alt.X(f'{bars}:N', title=None),
        alt.Y(f'{values}:Q', title=text),
        alt.Color(col,
                legend=alt.Legend(orient='bottom-left', title=None),
                scale=alt.Scale(scheme=palette)
                ),
        opacity=alt.condition(filter_in, alt.value('1'), alt.value('0.6'))
    ).transform_filter(
        {'and': [f'datum.{time} == {max_time}', 'datum.slope_x == "measures"']}
    ).properties(
        selection=filter_in,
        width=bars_w, height=bars_h
    )

    timeline_base = base.mark_line().encode(
        alt.X(f'{time}:O'),
        alt.Y(f'{values}:Q', title=text, scale=alt.Scale(domain=[df[values].min(), df[values].max()])),
        alt.Color(col, legend=None)
    ).properties(
        width=timeline_w, height=timeline_h
    )

    timeline = timeline_base.transform_filter(
        filter_in
    )
    timeline += timeline.mark_circle(size=25)

    timeline_avg = timeline_base.mark_line(strokeDash=[4,2], opacity=0.45).transform_filter(
        f'datum.{bars} == {slope_avg!r}'
    )

    slope = _build_slope(df, values, time, bars, col, text, filter_in, slope_y_pos, slope_w, slope_h)
    chart = barsplot & ((timeline_avg + timeline) | slope)

    return chart

def slope_comparison(df, values, bars, col, text, bars_w=200, bars_h=515,
                     slope_avg='Average', slope_w=350, slope_h=240,
                     slope_y_pos=10, slope_y_title=None):
    '''
    Plots 3 charts: v-bars and 2 slopegraph for comparison.

    Parameters
    ----------
    df : pandas.DataFrame
    values : str
        Name of the column used for values.
    bars : str
        Name of the column used to plot as X-axis on the bars.
    col : str
        Name of the column used for colors.
    text : str
        Name of the column used to show text on slopegraph.
    bars_w : int
        Bars plot width.
    bars_h : int
        Bars plot height.
    slope_avg : str
        Title for the avg measures on slopegraph.
    slope_w : int
        Slopegraph plot width.
    slope_h : int
        Slopegraph plot height.
    slope_y_pos : int
        Slopegraph titles position.
    slope_y_title : str
        Title to use on slope y axis.

    Returns
    -------
        altair.Chart

    Parameters
    ----------
    df : pandas.DataFrame
    vs : str list
        List of variables to include in the plot.
    year : int
        Year to extract from data.
    custom_fn : function
        Function to apply to df after formatting.
        Use it to format names on the df.
    kwargs : arguments passed to get_data_series
    '''
    df = df.copy()
    df['slope_x'] = 'measures'
    df_avg = df.groupby(col).mean().reset_index()
    df_avg[bars] = slope_avg
    df_avg['slope_x'] = 'averages'
    df = pd.concat([df, df_avg], ignore_index=True, sort=True)
    df[values] = df[values].round(2)
    df['slope_text'] = df[values].astype(str) + ' ' + df[col]

    mouse = alt.selection_single(on='mouseover', fields=[bars], empty='none', nearest=True)
    click = alt.selection_single(fields=[bars], empty='none')

    base = alt.Chart(df)

    barsplot = base.mark_point(filled=True).encode(
        alt.X(f'mean({values})', scale=alt.Scale(zero=False), axis=alt.Axis(title=None)),
        alt.Y(f'{bars}:N', axis=alt.Axis(title=None)),
        size=alt.condition(mouse, alt.value(400), alt.value(200))
    ).transform_filter(
        'datum.slope_x == "measures"'
    ).properties(
        selection=mouse,
        width=bars_w, height=bars_h
    )

    barsplot += barsplot.encode(
        size=alt.condition(click, alt.value(350), alt.value(200)),
        color=alt.condition(click, alt.ColorValue('#800000'), alt.value('#879cab'))
    ).properties(selection=click)

    bars_ci = base.mark_rule().encode(
        x=f'ci0({values})',
        x2=f'ci1({values})',
        y=f'{bars}:N'
    ).transform_filter(
        'datum.slope_x == "measures"'
    ).properties(
        width=bars_w, height=bars_h
    )
    
    slope_mouse = _build_slope(df, values, None, bars, col, text, mouse,
                               slope_y_pos, slope_w, slope_h, slope_y_title)
    slope_click = _build_slope(df, values, None, bars, col, text, click,
                               slope_y_pos, slope_w, slope_h)
    chart = (bars_ci + barsplot) | (slope_mouse & slope_click)

    return chart

def pdp_plot(df, rows, columns, values, variables=None, vars_filter=None,
             clusters=False, cluster_centers=5,
             columns_type='N', x_title=None, y_title=None,
             width=700, height=300):
    '''
    Plots a pdp plot for one variable.

    Parameters
    ----------
    df : pandas.DataFrame
        Expects the l

    Returns
    -------
    altair.Chart
    '''
    df = df.copy()
    if vars_filter and variables:
        df = df[df[variables] == vars_filter].drop(variables, axis=1)

    base = alt.Chart(df).properties(
        width=width, height=height
    )

    if clusters:
        df_clusters = utils.pdp_clusters(cluster_centers, df, rows, columns, values)
        background = alt.Chart(df_clusters).mark_line(strokeWidth=2).encode(
            alt.X(f'{columns}:{columns_type}', title=x_title),
            alt.Y(values, title=y_title),
            alt.Opacity(rows, legend=None),
            alt.ColorValue('#468499')
        ).properties(
            width=width, height=height
        )
    else:
        background = base.mark_line(strokeWidth=1).encode(
            alt.X(f'{columns}:{columns_type}', title=x_title),
            alt.Y(values, title=y_title),
            alt.Opacity(rows, legend=None),
            alt.ColorValue('#bbbbbb')
        )

    df_avg = df.groupby(columns)[values].mean().reset_index()
    avg_base = alt.Chart(df_avg).encode(
        alt.X(f'{columns}:{columns_type}', title=x_title),
        alt.Y(values, title=y_title),
    )
    avg = avg_base.mark_line(strokeWidth=5, color='gold')
    avg += avg_base.mark_line(strokeWidth=2)
    avg += avg_base.mark_point(filled=True, size=55)

    return background + avg

def pdp_plot_filter(filter_in, df, rows, columns, values, variables,
                    clusters=True, cluster_centers=3, cluster_lines=True,
                    columns_type='N', x_title=None, y_title=None,
                    width=700, height=400):
    df = df.copy()

    def get_lines(data, stroke_w, color, **kwargs):
        lines = alt.Chart(data).mark_line(strokeWidth=stroke_w, **kwargs).encode(
            alt.X(f'{columns}:{columns_type}',
                  title=x_title, axis=alt.Axis(minExtent=30)),
            alt.Y(values, title=y_title),
            alt.Opacity(rows, legend=None),
            alt.ColorValue(color)
        ).transform_filter(
            filter_in
        ).properties(
            width=width, height=height
        )
        return lines

    if clusters:
        df_clusters = utils.pdp_clusters(cluster_centers, df, rows, columns, values, variables)
        background = get_lines(df_clusters, 2, '#468499')
    else:
        background = get_lines(df, 1, '#bbbbbb')

    if cluster_lines: background = get_lines(df, 1, '#bbbbbb', strokeDash=[2,2]) + background

    df_avg = df.groupby([columns, variables])[values].mean().reset_index()
    avg_base = alt.Chart(df_avg).encode(
        alt.X(f'{columns}:{columns_type}', title=x_title),
        alt.Y(values, title=y_title),
    ).transform_filter(filter_in)
    
    avg = avg_base.mark_line(strokeWidth=5, color='gold')
    avg += avg_base.mark_line(strokeWidth=2)
    avg += avg_base.mark_point(filled=True, size=55)

    return background + avg

def pdp_explore(df, rows, columns, values, variables,
                bars_w=570, bars_h=100,
                title='', **kwargs):
    select = alt.selection_single(on='mouseover', fields=[variables], empty='none')

    base = alt.Chart(df)

    barsplot = base.mark_bar().encode(
        alt.X(f'mean({values})', title=None),
        alt.Y(variables, axis=alt.Axis(orient='right'), title=None),
        opacity=alt.condition(select, alt.value(1), alt.value(0.65))
    ).properties(
        selection=select,
        width=bars_w, height=bars_h
    )

    pdp = pdp_plot_filter(select, df, rows, columns, values, variables, **kwargs)

    chart = alt.vconcat(
        barsplot, pdp,
        title=title
    )

    return chart

def timeline_grid(df, values, time, grid, variables=None, vars_filter=[], vars_lbl=[],
                  title=None, unit=None, cols=6, fs=(14,6), plot_avg=True, avg_lbl='avg',
                  sharey=True, percentage=False, legend=False,
                  legend_bbox=(0,-0.5), legend_loc='upper center'):
    df = df.copy()
    if percentage: df[values] = df[values] * 100
    if variables is None: vars_filter = [None]
    if type(vars_filter) == str: vars_filter = [vars_filter]
    cmap = plt.get_cmap('tab10')

    for idx,var in enumerate(vars_filter):
        col = cmap(idx)
        if variables: df_t = df[df[variables] == var].drop(variables, axis=1)
        else: df_t = df
        
        label = var if len(vars_lbl) == 0 else vars_lbl[idx]
        
        if idx == 0:
            avgs = df_t.groupby(time)[values].mean().reset_index()
            ys = avgs[time].unique()
            years = ys
            ys = [*ys[::3], ys[-1]]
            ysl = [f'{e}'[2:4] for e in ys]
            n = df_t[grid].nunique()
            rows =  math.ceil(n / cols)
            fig, axes = plt.subplots(rows, cols, sharey=sharey, sharex=True, figsize=(fs))
        
        for i,(ax,g) in enumerate(zip(axes.flatten(), df_t[grid].unique())):
            df_g = df_t[df_t[grid] == g]
            ax.plot(avgs[time], avgs[values], ':', color=col,
                    alpha=0.4, label=f'{avg_lbl}-{label}')
            ax.plot(df_g[time], df_g[values], '-o', color=col,
                    markersize=3, alpha=0.75, label=label)
            ax.set_title(g)
            ax.set_xticks(ys)
            ax.set_xticklabels(ysl)
            if percentage: ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        
    if legend: ax.legend(bbox_to_anchor=legend_bbox, loc=legend_loc, borderaxespad=0.)
    for ax in axes.flatten()[len(df[grid].unique()):]: ax.set_axis_off()

    if title:
        title += f' ({unit})' if unit else ''
        print(title + ':')

    sns.despine()
    plt.show()
    print(f'Years: {", ".join(years.astype(str))}.')
    