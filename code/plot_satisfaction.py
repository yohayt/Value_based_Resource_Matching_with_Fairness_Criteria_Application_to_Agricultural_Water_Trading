#!/usr/bin/env python
from glob import glob
import numpy as np
import pandas as pd
from pdb import set_trace

from aaviz import plot

def main():
    dfs = []
    for f in glob('../results/satisfaction/sat.*csv'):
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs)
    df = df[df.network.isin(['Touchet10', 'Yakima10'])]
    df.delta = df.delta/100

    touchet = pd.read_excel('../../../water/real/input/touchet/touchet_10_af_per_unit.xlsx')
    touchet = touchet[['WR_Doc_ID', 'total_units']].groupby('WR_Doc_ID').sum()

    yakima = pd.read_excel('../../../water/real/input/yakima/yakima_10_af_per_unit.xlsx')
    yakima = yakima[['WR_Doc_ID', 'total_units']].groupby('WR_Doc_ID').sum()

    df_touchet = df[df.network=='Touchet10'].merge(touchet, left_on='buyer',
            right_on='WR_Doc_ID')
    df_yakima = df[df.network=='Yakima10'].merge(yakima, left_on='buyer',
            right_on='WR_Doc_ID')

    dft = pd.concat([df_touchet, df_yakima])
    dft.satisfaction = dft.sat/dft.total_units

    fig, gs = plot.initiate_figure(x=8, y=2, gs_nrows=1, gs_ncols=1,
            gs_wspace=1.4, fs_fontsize=17)

    colors = plot.COLORS['mathematica']

    ax = plot.subplot(fig=fig, grid=gs[0,0], func='sns.boxplot',
            data=dft, 
            pf_palette=colors[3:5],
            pf_y='satisfaction', pf_x='delta', pf_hue='network',
            lg_visible=True,
            lg_ncol=2,
            lg_loc='center',
            lg_bbox_to_anchor=(.4,1.1),
            ag_ymin=0, ag_ymax=1,
            fs_legend_title='normalsize',
            fs_xlabel='large',
            fs_ylabel='large',
            hatch=True,
            la_xlabel=f'$\delta$',
            la_ylabel='Buyer satisfaction')
    plot.savefig('satisfaction.pdf')

if __name__ == '__main__':
    main()
