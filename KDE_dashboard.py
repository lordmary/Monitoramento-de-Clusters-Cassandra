import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde

# --- 1. Carregar Dados ---
try:
    df_read = pd.read_csv('20260206_171648721_r_combinado.csv')
    df_write = pd.read_csv('20260206_171648721_w_combinado.csv')
    print("Arquivos carregados via nome padrão.")
except:
        print("ERRO CRÍTICO: Não achei os arquivos CSV.")

# --- 2. Função de Preparação ---
def preparar_dados(df, target_col):
    # Lista de termos que indicam métricas calculadas (o "gabarito")
    vazamento = ['percentile', 'latency', 'average', 'min', 'max']
    
    # Mantemos apenas o Target, a Carga e a Infraestrutura (container_)
    # Removendo qualquer outro percentil ou média que não seja o alvo
    colunas_ok = [col for col in df.columns if col == target_col or 
                  (('container_' in col or 'rate' in col) and 
                   not any(t in col.lower() for t in vazamento))]
    
    df_f = df[colunas_ok].fillna(0)
    
    if 'mean_rate' in df_f.columns: features = ['mean_rate']
    else: features = ['queries_requested']
    
    infra = [c for c in df_f.columns if 'container_' in c and df_f[c].std() > 0]
    X = df_f[features + infra]
    y = df_f[target_col]
    return X, y

# --- 3. Configuração do Plot ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Dashboard KDE - Densidade de Probabilidade (Real vs Predito)', fontsize=16)

configuracoes = [(df_read, 'Leitura', 0), (df_write, 'Escrita', 1)]
targets = ['95th_percentile', 'd_95th_percentile', 'w_95th_percentile']
titulos = ['Padrão (95th)', 'Prefixo d (95th)', 'Prefixo w (95th)']

# --- 4. Execução ---
for df_atual, nome_op, row in configuracoes:
    for col, target in enumerate(targets):
        
        X, y = preparar_dados(df_atual, target)
        
        # --- A CORREÇÃO: shuffle=True ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True, random_state=42)
        
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        # Lógica KDE
        try:
            density_real = gaussian_kde(y_test)
            density_pred = gaussian_kde(y_pred)
            
            xmin = min(y_test.min(), y_pred.min())
            xmax = max(y_test.max(), y_pred.max())
            margin = (xmax - xmin) * 0.1
            xs = np.linspace(xmin - margin, xmax + margin, 200)
            
            ax = axes[row, col]
            ax.plot(xs, density_real(xs), label='Real', color='blue', linewidth=2)
            ax.fill_between(xs, density_real(xs), color='blue', alpha=0.1)
            ax.plot(xs, density_pred(xs), label='Predito', color='orange', linestyle='--', linewidth=2)
            
            ax.set_title(f"{nome_op} - {titulos[col]}\nR²: {r2:.4f}")
            ax.set_xlabel('Latência (ms)')
            if col == 0: ax.set_ylabel('Densidade')
            ax.grid(True, alpha=0.3)
            ax.legend()
        except:
            axes[row, col].text(0.5, 0.5, "Erro no KDE (Dados Constantes?)", ha='center')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('dashboard_kde_corrigido.jpg')
plt.show()