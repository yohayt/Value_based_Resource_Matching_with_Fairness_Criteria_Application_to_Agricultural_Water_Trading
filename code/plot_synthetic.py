import pandas as pd

import plot

N = 10
k = 5

def main():
    df = pd.read_csv('results_synthetic.csv')

    lambda_beta_str = '$\lambda,\\beta_{\\textrm{high}}$'

    lbst = list(zip(df['lambda'], df.beta_high))

    df[lambda_beta_str] = lbst

    beta_str = '$\\beta_{\\textrm{high}}$'
    lambda_str = '$\lambda$'
    df = df.rename(columns={'beta_high': beta_str, 'lambda': lambda_str})
    df['density'] = df.num_edges/N/N/k/k

    fig, gs = plot.initiate_figure(x=16, y=2.8, gs_nrows=1, gs_ncols=11,
                                   gs_wspace=2, fs_fontsize=17)

    markers = ['o', 'o', '^', '^', 's', 's']
    style = df[lambda_beta_str] #[lambda_str,beta_str]
    colors = [plot.COLORS['mathematica'][i] for i in [0, 0, 1, 1, 2, 2]]
    pd.set_option('display.max_columns', None)
    print(df.head())

    ax = plot.subplot(fig=fig, grid=gs[0, 0:2], func='sns.lineplot',
                      data=df,
                      pf_x='delta', pf_y='num_sellers', pf_hue=lambda_beta_str,
                      pf_palette=colors,
                      pf_style=style,
                      pf_markers = markers,
                      pf_dashes = ['', (2,2), '', (2,2), '', (2,2)],
                      pf_linewidth=3,
                      ag_xmin=0, ag_xmax=1, ag_ymin=0, ag_ymax=10,
                      lg_visible=False,
                      la_title='(a)',
                      fs_title='normalsize',
                      la_xlabel=f'$\delta$',
                      la_ylabel='Number of sellers')

    ax = plot.subplot(fig=fig, grid=gs[0, 2:5], func='sns.lineplot',
                      data=df,
                      pf_x='delta', pf_y='num_edges', pf_hue=lambda_beta_str,
                      pf_palette=colors,
                      pf_style=style,
                      pf_markers = markers,
                      pf_dashes = ['', (2,2), '', (2,2), '', (2,2)],
                      pf_linewidth=3,
                      ag_xmin=0, ag_xmax=1, ag_ymin=0,ag_ymax=500,
                      lg_visible=False,
                      la_title='(b)',
                      fs_title='normalsize',
                      la_xlabel=f'$\delta$',
                      la_ylabel='Edges in the Res-needs graph')

    ax = plot.subplot(fig=fig, grid=gs[0, 5:8], func='sns.lineplot',
                      data=df,
                      pf_x='delta', pf_y='total_value', pf_hue=lambda_beta_str,
                      pf_palette=colors,
                      pf_style=style,
                      pf_markers = markers,
                      pf_dashes = ['', (2,2), '', (2,2), '', (2,2)],
                      pf_linewidth=3,
                      ag_xmin=0, ag_xmax=1, ag_ymin=0, ag_ymax=1,
                      lg_title=lambda_beta_str,
                      fs_legend_title='small',
                      fs_legend='small',
                      la_title='(c)',
                      fs_title='normalsize',
                      la_xlabel=f'$\delta$',
                      la_ylabel='$\sigma(\mathcal{T}^*)/\sigma_0$')

    ax = plot.subplot(fig=fig, grid=gs[0, 8:11], func='sns.lineplot',
                      data=df,
                      pf_x='delta', pf_y='normalized_welfare', pf_hue=lambda_beta_str,
                      pf_palette=colors,
                      pf_style=style,
                      pf_markers = markers,
                      pf_dashes = ['', (2,2), '', (2,2), '', (2,2)],
                      pf_linewidth=3,
                      ag_xmin=0, ag_xmax=1, ag_ymin=0,
                      lg_visible=False,
                      la_title='(d)',
                      fs_title='normalsize',
                      la_xlabel=f'$\delta$',
                      la_ylabel='wel($\mathcal{T}^*$)$/\sigma_0$')

    plot.savefig('synthetic.pdf')

if __name__ == '__main__':
    main()
