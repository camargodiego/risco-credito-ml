import glob
import os
import kagglehub
import pandas as pd

# 1. Baixa os arquivos usando a função correta de COMPETIÇÃO
print("⏳ Baixando dados da competição do Kaggle...")
kaggle_path = kagglehub.competition_download("home-credit-default-risk")

# 2. Garante que a pasta 'data/' exista
os.makedirs("data", exist_ok=True)

# 3. Mapeia todos os CSVs baixados
csv_files = glob.glob(f"{kaggle_path}/*.csv")
print(
    f"Encontrados {len(csv_files)} arquivos CSV. Iniciando conversão para"
    " Parquet...\n"
)

# 4. Converte cada CSV para Parquet
for filepath in csv_files:
  filename = os.path.basename(filepath)
  table_name = os.path.splitext(filename)[0].lower()
  parquet_path = f"data/{table_name}.parquet"

  print(f"⏳ Processando {filename} -> {parquet_path}...")

  try:
    df = pd.read_csv(filepath, encoding="utf-8")
  except UnicodeDecodeError:
    df = pd.read_csv(filepath, encoding="iso-8859-1")

  # Salva compactado em Snappy Parquet
  df.to_parquet(parquet_path, index=False, compression="snappy")
  print(f"✅ {table_name}.parquet gerado com sucesso!")

print("\n🎉 Conversão concluída! Seus arquivos estão salvos em 'data/'.")