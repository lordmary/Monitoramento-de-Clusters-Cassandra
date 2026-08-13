import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os arquivos
df_w = pd.read_csv("20251023_192015739_w_combinado.csv")
df_r = pd.read_csv("20251023_192015739_r_combinado.csv")

# 2. Identificar os cenários
df_w['Cenario'] = 'Escrita (Write)'
df_r['Cenario'] = 'Leitura (Read)'

# 3. Juntar tudo num único DataFrame
df = pd.concat([df_w, df_r])
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 4. Configurar o visual
sns.set_style("whitegrid")
fig, axes = plt.subplots(3, 2, figsize=(18, 15))
fig.suptitle('Dashboard de Performance: Comparativo Leitura vs Escrita', fontsize=20)

# --- Gráfico 1: Boxplot de Latência (Visão Geral) ---
sns.boxplot(data=df, x='Cenario', y='average_latency', ax=axes[0, 0], palette="Set2")
axes[0, 0].set_title('Distribuição de Latência (Boxplot)')

# --- Gráfico 2: Latência ao Longo do Tempo ---
sns.lineplot(data=df, x='timestamp', y='average_latency', hue='Cenario', ax=axes[0, 1], palette="Set2")
axes[0, 1].set_title('Evolução da Latência Média')
axes[0, 1].tick_params(axis='x', rotation=45)

# --- Gráfico 3: Throughput (Requisições por Segundo) ---
sns.lineplot(data=df, x='timestamp', y='queries_num', hue='Cenario', ax=axes[1, 0], palette="Set2")
axes[1, 0].set_title('Throughput (Queries/seg)')
axes[1, 0].tick_params(axis='x', rotation=45)

# --- Gráfico 4: P99 (Latência de Cauda) ---
sns.lineplot(data=df, x='timestamp', y='99th_percentile', hue='Cenario', ax=axes[1, 1], palette="Set2")
axes[1, 1].set_title('Latência P99 (Pior Caso)')
axes[1, 1].tick_params(axis='x', rotation=45)

# --- Gráfico 5: Distribuição (Corrigido com Zoom) ---
# Aqui está o segredo para não ficar "feio": limitamos o eixo X
limite_x = df['average_latency'].quantile(0.99) * 1.5
sns.kdeplot(data=df, x='average_latency', hue='Cenario', fill=True, ax=axes[2, 0], palette="Set2")
axes[2, 0].set_xlim(0, limite_x) 
axes[2, 0].set_title(f'Densidade de Latência (Zoom até {limite_x:.0f}ms)')

# --- Gráfico 6: Erros ---
if df['errors_occurred'].sum() > 0:
    sns.lineplot(data=df, x='timestamp', y='errors_occurred', hue='Cenario', ax=axes[2, 1], palette="Set2")
    axes[2, 1].set_title('Erros por Segundo (Atenção!)')
else:
    sns.lineplot(data=df, x='timestamp', y='total_connections', hue='Cenario', ax=axes[2, 1], palette="Set2")
    axes[2, 1].set_title('Total de Conexões (Sem Erros)')
axes[2, 1].tick_params(axis='x', rotation=45)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('dashboard_completo.png')
print("Gráfico gerado!")