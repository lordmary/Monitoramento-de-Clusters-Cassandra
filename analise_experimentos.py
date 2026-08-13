import pandas as pd
import glob
import os
import numpy as np
import gc  # Garbage Collector para liberar memória

# --- CONFIGURAÇÃO ---
# Nome exato do arquivo gerado pelo exportar_tudo.py
ARQUIVO_HOST_METRICS = 'metricas_FILTRADAS_node_container.csv'
# Padrão para encontrar seus arquivos de 2026
PADRAO_GRAFANA_FILES = '2026*.csv'
TIMESTAMP_COL = 'timestamp'

def otimizar_dataframe(df):
    """Converte float64 para float32 para economizar 50% de RAM"""
    floats = df.select_dtypes(include=['float64']).columns
    df[floats] = df[floats].astype('float32')
    return df

def carregar_metricas_com_limpeza(arquivo):
    print(f"--- Carregando {arquivo} com otimização de memória ---")
    
    # 1. Identificar colunas inúteis (variância zero ou tudo NaN) lendo em chunks
    try:
        preview = pd.read_csv(arquivo, nrows=5)
    except FileNotFoundError:
        print(f"ERRO CRÍTICO: Não encontrei o arquivo '{arquivo}'.")
        print("Verifique se o exportar_tudo.py terminou e se o nome está correto.")
        exit(1)

    todas_colunas = preview.columns.tolist()
    total_cols = len(todas_colunas)
    
    print(f"Total de colunas detectadas: {total_cols}")
    print("Analisando colunas constantes (sem variação)... aguarde.")
    
    # Lendo o arquivo inteiro em chunks para calcular desvio padrão sem estourar RAM
    # Se o desvio padrão é 0, a coluna é constante e inútil.
    try:
        # Tenta ler em chunks para não travar
        stats = pd.read_csv(arquivo, usecols=lambda x: x != TIMESTAMP_COL, dtype='float32')
        descricao = stats.describe().transpose()
        
        # Colunas onde o desvio padrão (std) é 0 ou NaN são inúteis
        cols_inuteis = descricao[ (descricao['std'] == 0) | (descricao['std'].isna()) ].index.tolist()
        
        print(f"Colunas descartadas (constantes/vazias): {len(cols_inuteis)}")
        print(f"Colunas úteis restantes: {total_cols - len(cols_inuteis)}")
        
        # Liberar memória das estatísticas
        del stats
        del descricao
        gc.collect()
        
        # 2. Carregar o arquivo final apenas com as colunas úteis
        cols_uteis = [c for c in todas_colunas if c not in cols_inuteis]
        
    except Exception as e:
        print(f"Aviso: Não foi possível calcular estatísticas (memória cheia?). Tentando carregar tudo... Erro: {e}")
        cols_uteis = todas_colunas

    print("Carregando dataset limpo...")
    # Aqui é o pulo do gato: carrega SÓ as colunas úteis
    df = pd.read_csv(arquivo, usecols=cols_uteis)
    df = otimizar_dataframe(df)
    
    # Converter timestamp
    if TIMESTAMP_COL in df.columns:
        # Tenta converter para datetime; se falhar, assume que já é numérico e converte
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], unit='s')
        df = df.sort_values(by=TIMESTAMP_COL)
    
    return df

# --- EXECUÇÃO PRINCIPAL ---

if __name__ == "__main__":
    # 1. Carregar Métricas de Infra (O Pesado)
    df_metrics = carregar_metricas_com_limpeza(ARQUIVO_HOST_METRICS)

    # 2. Processar Arquivos da Aplicação
    app_files = glob.glob(PADRAO_GRAFANA_FILES)
    print(f"\nArquivos de aplicação encontrados: {app_files}")

    if not app_files:
        print("Nenhum arquivo 2026*.csv encontrado! Verifique a pasta.")

    for app_file in app_files:
        if '_combinado' in app_file:
            continue # Pula arquivos que a gente já gerou antes
            
        print(f"\nProcessando: {app_file} ...")
        
        try:
            # Carregar App
            df_app = pd.read_csv(app_file)
            
            # Limpeza básica do Grafana (linhas vazias no inicio)
            if len(df_app) > 2 and df_app.iloc[0]['queries_num'] == 0:
                 df_app = df_app.iloc[2:].reset_index(drop=True)

            df_app = otimizar_dataframe(df_app)
            
            if TIMESTAMP_COL in df_app.columns:
                df_app[TIMESTAMP_COL] = pd.to_datetime(df_app[TIMESTAMP_COL], unit='s')
                df_app = df_app.sort_values(by=TIMESTAMP_COL)

            # 3. MERGE ASOF (O Pulo do Gato)
            # Tolerância: Aceita desencontro de até 15 segundos nos relógios
            print("Cruzando dados (Merge)...")
            df_merged = pd.merge_asof(
                df_app, 
                df_metrics, 
                on=TIMESTAMP_COL, 
                direction='nearest',
                tolerance=pd.Timedelta('15s')
            )

            # 4. Limpeza Final e Salvamento
            # Remove linhas onde não houve match (infra vazia naquele segundo)
            col_ref_infra = df_metrics.columns[1] if len(df_metrics.columns) > 1 else None
            if col_ref_infra:
                df_merged = df_merged.dropna(subset=[col_ref_infra]) 
            
            output_file = app_file.replace('.csv', '_combinado.csv')
            df_merged.to_csv(output_file, index=False)
            print(f"--> SUCESSO! Salvo em: {output_file}")
            
            # Limpa memória antes do próximo loop
            del df_merged
            del df_app
            gc.collect()
            
        except Exception as e:
            print(f"Erro ao processar {app_file}: {e}")

    print("\n--- Fim do Processamento ---")