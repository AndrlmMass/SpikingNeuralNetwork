import pandas as pd

for fname in ['GLMM_predictions.xlsx', 'GLMM_predictions_clust.xlsx', 'GLMM_predictions_clust2.xlsx']:
    df = pd.read_excel(f'results/acc_history/mnist/2026.05.24/21/{fname}', engine='openpyxl')
    print(f'\n=== {fname} ===')
    print('Columns:', df.columns.tolist())
    print('reg_target:', df['reg_target'].unique().tolist() if 'reg_target' in df.columns else 'N/A')
    fit_range = f"fit range: {df['fit'].min():.4f} – {df['fit'].max():.4f}" if 'fit' in df.columns else ''
    print(fit_range)
    print(df.head(4).to_string())
