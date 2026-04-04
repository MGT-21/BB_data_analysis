# Projeto de Análise de Dados e Previsão de Fraude/Inadimplência

## Descrição

Este projeto visa analisar dados financeiros para identificar padrões de inadimplência, fraude e outros indicadores de risco. Utilizando uma arquitetura de dados em camadas (Bronze, Silver, Gold), o projeto processa dados brutos, realiza limpeza e transformação, e gera insights através de modelos preditivos.

### Objetivos Principais
- **Previsão de Inadimplência**: Desenvolver um modelo para prever se um cliente será inadimplente com base no perfil da pessoa.
- **Taxa de Aprovação por Região**: Analisar e calcular taxas de aprovação de crédito por diferentes regiões.
- **Detecção de Outliers**: Identificar anomalias nos dados que podem indicar fraudes ou comportamentos atípicos.
- **Análise de Performance do Sistema**: Avaliar o desempenho do sistema de crédito.
- **Segmentação de Clientes**: Classificar clientes em segmentos para melhor targeting.

## Estrutura do Projeto

```
bb_2/
├── config.py                 # Arquivo de configuração
├── README.md                 # Este arquivo
├── data/
│   ├── bronze/               # Dados brutos ingeridos
│   │   └── bronze_databridge.csv
│   ├── silver/               # Dados tratados e limpos
│   │   └── silver_databridge.csv
│   ├── gold/                 # Dados agregados e modelados
│   │   ├── gold_fraude_casos_alto_risco.csv
│   │   └── gold_fraude.csv
│   ├── raw/                  # Dados originais
│   │   ├── databridge_squad19_dictionary.csv
│   │   └── databridge_squad19_sintetico.csv
│   └── eda/                  # Dados para análise exploratória
│       └── eda_databridge.csv
├── notebooks/                # Notebooks Jupyter para análise
│   ├── 01_bronze_ingestao.ipynb      # Ingestão de dados brutos
│   ├── 02-silver_tratamento.ipynb    # Tratamento e limpeza
│   ├── 03_eda.ipynb                  # Análise Exploratória de Dados
│   ├── 04_gold_Fraudes.ipynb         # Modelagem para detecção de fraudes
│   ├── 05_gold_Inadimplencia.ipynb   # Previsão de inadimplência
│   ├── 06_gold_PerformanceSistema.ipynb  # Análise de performance
│   └── 07_gold_SegmentoDeClientes.ipynb   # Segmentação de clientes
└── output/                  # Saídas e resultados
```

## Instalação

1. Clone o repositório:
   ```bash
   git clone <url-do-repositorio>
   cd bb_2
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
   (Certifique-se de que o arquivo `requirements.txt` existe ou crie um com as bibliotecas necessárias, como pandas, numpy, scikit-learn, matplotlib, etc.)

3. Configure o ambiente Python (se necessário):
   - Use um ambiente virtual: `python -m venv env` e ative com `env\Scripts\activate` no Windows.

## Uso

1. **Ingestão de Dados**: Execute o notebook `01_bronze_ingestao.ipynb` para carregar dados brutos na camada Bronze.

2. **Tratamento**: Use `02-silver_tratamento.ipynb` para limpar e transformar os dados na camada Silver.

3. **Análise Exploratória**: Realize EDA com `03_eda.ipynb` para entender os dados.

4. **Modelagem**:
   - Fraudes: `04_gold_Fraudes.ipynb`
   - Inadimplência: `05_gold_Inadimplencia.ipynb`
   - Performance: `06_gold_PerformanceSistema.ipynb`
   - Segmentação: `07_gold_SegmentoDeClientes.ipynb`

5. Os resultados são salvos na pasta `output/` e na camada Gold.

## Dados

- **Fonte**: Dados sintéticos do Databridge Squad 19.
- **Dicionário**: `databridge_squad19_dictionary.csv` contém a descrição das variáveis.
- **Processamento**: Segue arquitetura Medallion (Bronze → Silver → Gold).

## Tecnologias Utilizadas

- Python
- Jupyter Notebooks
- Pandas, NumPy
- Scikit-learn (para modelagem)
- Matplotlib/Seaborn (para visualizações)

## Contribuição

1. Faça um fork do projeto.
2. Crie uma branch para sua feature: `git checkout -b feature/nova-feature`.
3. Commit suas mudanças: `git commit -m 'Adiciona nova feature'`.
4. Push para a branch: `git push origin feature/nova-feature`.
5. Abra um Pull Request.

## Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## Contato

Para dúvidas ou sugestões, entre em contato com a equipe do Squad 19.