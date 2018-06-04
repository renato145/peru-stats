import pandas as pd, numpy as np
from sklearn.cluster import MiniBatchKMeans

def _pdp_cluster(clusters, df, rows, columns, values):
    df = df.copy()
    df.dropna(axis=1, how='all', inplace=True)
    medians = df.median()
    for col in df: df[col].fillna(medians[col], inplace=True)

    kmeans = MiniBatchKMeans(clusters, random_state=42)
    kmeans.fit(df)
    df_clusters = pd.DataFrame(kmeans.cluster_centers_, columns=df.columns)
    df_clusters[rows] = [f'cluster_{e+1}' for e in range(clusters)]
    df_clusters = df_clusters.set_index(rows).stack().reset_index()
    df_clusters.columns = [rows, columns, values]
    
    return df_clusters

def pdp_clusters(clusters, df, rows, columns, values, variables=None, vars_filter=None):
    df = df.copy()
    group = [rows, columns]
    if vars_filter and variables:
        df = df[df[variables] == vars_filter].drop(variables, axis=1)
    elif vars_filter is None and variables:
        group = [variables] + group

    df = df.set_index(group).unstack()
    df.columns = df.columns.get_level_values(1)

    if len(group) < 3:
        out = _pdp_cluster(clusters, df, rows, columns, values)
    else:
        out = []
        for idx in df.index.get_level_values(0).unique():
            t = _pdp_cluster(clusters, df.loc[idx], rows, columns, values)
            t[variables] = idx
            out.append(t)
        
        out = pd.concat(out, ignore_index=True)

    return out


    # '''
    # Formats dataframe for suitable use in plotting pdp graphs.
    
    # Parameteres
    # -----------
    # df : pandas.DataFrame
    # rows : str
    #     Column name to be use as rows.
    # columns : str
    #     Column name to be use new columns.
    # values : str
    #     Column name to be use for values.
    # variables : str
    #     Column name to distinct variables.
    # vars_filter: str
    #     Filter for df[vars]

    # Returns
    # -------
    # df : pandas.DataFrame
    # '''

    #    df = df.copy()
    # group = [rows, columns]
    # if vars_filter and variables:
    #     df = df[df[variables] == vars_filter].drop(variables, axis=1)
    # elif vars_filter is None and variables:
    #     group = [variables] + group

    # df = df.set_index(group).unstack()
    # df.columns = df.columns.get_level_values(1)
    # df = df.T.reset_index()
