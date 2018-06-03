import altair as alt
import pandas as pd

def _build_slope(df, values, time, bars, col, text, filter_in, y_pos=10, width=350, height=400):
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
        y=alt.Y(values, title=None,
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

def slope_comparison(df, values, bars, col, text,
                     bars_w=200, bars_h=515,
                     slope_avg='Average', slope_w=350, slope_h=240, slope_y_pos=10,
                     palette='tableau10'):
    '''
    Plot a slope graph for given variables and year.

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
    click = alt.selection_single(fields=[bars], empty='none', nearest=True)

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
    
    slope_mouse = _build_slope(df, values, None, bars, col, text, mouse, slope_y_pos, slope_w, slope_h)
    slope_click = _build_slope(df, values, None, bars, col, text, click, slope_y_pos, slope_w, slope_h)
    chart = (bars_ci + barsplot) | (slope_mouse & slope_click)

    return chart
