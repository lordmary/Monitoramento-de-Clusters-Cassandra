import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ==============================================================================
# 1. CARREGAR ARQUIVOS
# ==============================================================================
try:
    df_read = pd.read_csv('20260206_171648721_r_combinado.csv')
    df_write = pd.read_csv('20260206_171648721_w_combinado.csv')
    print("Arquivos carregados via nome padrão.")
except:
    print("ERRO CRÍTICO: Não achei os arquivos CSV.")

# ==============================================================================
# 2. FUNÇÃO DE PREPARAÇÃO (ATUALIZADA)
# ==============================================================================
def preparar_dados(df, target_col):
    # --- PASSO NOVO: LIMPEZA DE "GABARITO" (Anti-Data Leakage) ---
    # Lista de termos que indicam que a coluna é um resultado de latência e não uma causa
    termos_proibidos = ['percentile', 'latency', 'average', 'min', 'max']
    
    # Vamos manter apenas as colunas que NÃO têm esses termos
    # EXCETO, é claro, o nosso 'target_col' que é o que queremos prever
    colunas_limpas = []
    for col in df.columns:
        # Se for o target, a gente mantém
        if col == target_col:
            colunas_limpas.append(col)
            continue
        
        # Se for carga (mean_rate), a gente mantém
        if col in ['mean_rate', 'queries_requested']:
            colunas_limpas.append(col)
            continue
            
        # Se for infra (container_), a gente mantém APENAS se não for um percentil perdido
        if 'container_' in col:
            colunas_limpas.append(col)
            continue
            
        # Se não cair em nenhuma regra acima e tiver termo proibido, a gente ignora (deleta)
        # Isso remove colunas como '99th_percentile', 'average_latency', etc.

    df_filtrado = df[colunas_limpas]

    # --- RESTO DA PREPARAÇÃO ---
    if 'mean_rate' in df_filtrado.columns:
        col_carga = 'mean_rate'
    else:
        col_carga = 'queries_requested'
        
    infra_cols = [c for c in df_filtrado.columns if 'container_' in c]
    infra_cols = df_filtrado[infra_cols].select_dtypes(include=[np.number]).columns.tolist()
    infra_cols = [c for c in infra_cols if df_filtrado[c].std() > 0]
    
    features = [col_carga] + infra_cols
    
    X = df_filtrado[features].fillna(0)
    y = df_filtrado[target_col].fillna(0)
    carga = df_filtrado[col_carga].fillna(0)
    
    return X, y, carga

# [O restante do seu código de plotagem continua igual daqui para baixo!]
# ==============================================================================
# 3. CONFIGURAÇÃO DO DASHBOARD
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('Dashboard de Predição Temporal (Random Forest) - Sem Vazamento de Dados', fontsize=16)

configuracoes = [(df_read, 'Leitura', 0), (df_write, 'Escrita', 1)]
targets = ['95th_percentile', 'd_95th_percentile', 'w_95th_percentile']
titulos = ['Padrão (95th)', 'Prefixo d (95th)', 'Prefixo w (95th)']

for df_atual, nome_op, row in configuracoes:
    for col, target in enumerate(targets):
        # O loop e a plotagem permanecem iguais, chamando a função preparar_dados limpa
        try:
            X, y, carga_full = preparar_dados(df_atual, target)
            
            indices = np.arange(len(X))
            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                X, y, indices, test_size=0.30, shuffle=True, random_state=42
            )
            
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            
            y_pred = rf.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            df_result = pd.DataFrame({'Real': y_test, 'Predito': y_pred, 'Index': idx_test})
            df_result = df_result.sort_values(by='Index')
            carga_test = carga_full.iloc[idx_test].loc[df_result['Index']]
            
            ax1 = axes[row, col]
            ax2 = ax1.twinx()
            ax2.plot(range(len(df_result)), carga_test, color='green', linestyle=':', alpha=0.3, label='Carga')
            ax1.plot(range(len(df_result)), df_result['Real'], label='Real', color='blue', alpha=0.8)
            ax1.plot(range(len(df_result)), df_result['Predito'], label='Predito', color='orange', linestyle='--', alpha=0.9)
            ax1.set_title(f"{nome_op} - {titulos[col]}\nR²: {r2:.4f}")
            # ... (restante dos comandos de plotagem que você já tem)
        except Exception as e:
            print(f"Pulei o target {target} pois não o encontrei ou deu erro: {e}")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('dashboard_temporal_final_limpo.jpg')
print("Dashboard Final gerado com sucesso!")
plt.show()