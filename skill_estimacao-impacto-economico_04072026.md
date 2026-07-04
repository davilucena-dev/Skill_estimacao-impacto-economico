---
name: estimacao-impacto-economico
description: >
  Use esta skill quando precisar estimar o impacto econômico regional de um
  investimento, choque de demanda, política pública ou implantação de
  empreendimento em qualquer setor e qualquer região brasileira. Ativa quando
  o usuário pedir: cálculo de efeitos diretos/indiretos/induzidos, simulação
  de choque de demanda, multiplicadores de produção/renda/emprego, modelo
  de Leontief aberto ou fechado (com famílias), decomposição de impactos,
  ou relatório de impacto regional. Requer uma MIP regionalizada como
  insumo (pode vir da skill regionalizacao-mip). Entrega vetores de impacto
  setorial, multiplicadores e top setores beneficiados.
---
# Skill: Estimação de Impacto Econômico Regional

## Propósito
Simular choques de demanda final sobre a economia de uma região (estado, microrregião ou município) utilizando o modelo de insumo-produto de Leontief — versões aberta e fechada (com famílias). Calcula efeitos diretos, indiretos e induzidos, multiplicadores de produção/renda/emprego, e identifica os setores mais beneficiados. Genérica para qualquer setor (indústria, serviços, agropecuária, mineração etc.) e qualquer região com MIP regionalizada disponível.

## O que esta skill NÃO faz
- Não regionaliza a MIP — parte de uma matriz de coeficientes técnicos já regionalizada (use a skill `regionalizacao-mip`)
- Não constrói modelos CGE completos — apenas o modelo linear de Leontief
- Não coleta dados de emprego ou produção — recebe esses dados como parâmetro
- Não faz decomposição estrutural (SDA) — apenas simulação de choque em um ponto no tempo

## Fontes
- Referência: `Estimação de Impacto Econômico.pdf` (Relatório Técnico AgenteNazaré — Davi Lucena da Silva, UFV, 2026)
- Leontief, W. (1936). *Quantitative Input and Output Relations in the Economic System of the United States.*
- Miller, R. E. & Blair, P. D. (2009). *Input-Output Analysis: Foundations and Extensions* (2nd ed.). Cambridge University Press.
- Guilhoto, J. J. M. (2010). *Input-Output Analysis: Theory and Foundations.* MPRA Paper 32566.

---

## 1. Instalação de Dependências

```bash
pip install numpy pandas openpyxl scipy
```

---

## 2. Pré-requisitos

### 2.1 Insumos necessários

| Insumo | Descrição | Fonte possível |
|---|---|---|
| **Matriz A_reg** (n x n) | Matriz de coeficientes técnicos regionalizada e balanceada | Skill `regionalizacao-mip` ou MIP oficial do estado |
| **Vetor f_produto** (m,) | Consumo das famílias por produto (R$) | Tabela de Usos (Ibge — MIP), aba "02" |
| **Matriz D** (n x m) | Market share: participação de cada atividade na produção de cada produto | MIP Ibge, aba "13" |
| **Vetor de produção X** (n,) | Produção total por setor (R$) | MIP Ibge, Tabela de Recursos |
| **Vetor delta_f** (n,) | Choque de demanda final a simular (R$ por setor) | Definido pelo usuário (ex: R$ 500 MM no setor automotivo) |
| **Vetor de coeficientes de trabalho l** (n,) opcional | Renda do trabalho por unidade produzida (R$/R$) | Remunerações na Tabela de Usos, ou estimado como fração do VA |

**Níveis de agregação suportados:** 12, 20 ou 67 setores (qualquer nível da MIP Ibge).

### 2.2 Entendimento do modelo

```
Modelo Aberto (Leontief):
  x = (I - A)^{-1} × f = L × f
  Efeito total = direto + indireto

Modelo Fechado (com famílias):
  A_fechado = [[A,  h],
               [l^T, 0]]
  Onde:
    h = coeficiente de consumo das famílias (coluna extra)
    l^T = coeficiente de renda do trabalho (linha extra)
  
  Efeito total = direto + indireto + induzido
```

---

## 3. Fluxo Passo a Passo

### 3.1 Carregar a matriz regionalizada

```python
import numpy as np
import pandas as pd

def carregar_matriz_regional(caminho_arquivo, nivel=12):
    """
    Carrega a matriz A regionalizada a partir de arquivo.
    
    Suporta formatos:
    - .npy (numpy array)
    - .xlsx (Excel, primeira aba)
    - .csv
    
    Parâmetros:
        caminho_arquivo (str): Caminho do arquivo
        nivel (int): Nível de agregação (12, 20 ou 67)
    
    Retorna:
        ndarray: Matriz A_reg (n x n)
    """
    if caminho_arquivo.endswith('.npy'):
        A = np.load(caminho_arquivo)
    elif caminho_arquivo.endswith('.xlsx'):
        A = pd.read_excel(caminho_arquivo, header=None).values
    elif caminho_arquivo.endswith('.csv'):
        A = pd.read_csv(caminho_arquivo, header=None).values
    else:
        raise ValueError(f"Formato não reconhecido: {caminho_arquivo}")
    
    n_esperado = nivel
    assert A.shape == (n_esperado, n_esperado), \
        f"Matriz tem dimensão {A.shape}, esperada ({n_esperado}x{n_esperado})"
    
    return A.astype(float)
```

### 3.2 Construir a Inversa de Leontief (Modelo Aberto)

```python
def calcular_inversa_leontief(A):
    """
    Calcula a inversa de Leontief L = (I - A)^{-1}.
    
    Usa np.linalg.solve em vez de np.linalg.inv para maior
    estabilidade numérica: resolve (I - A) @ L = I.
    
    Parâmetros:
        A (ndarray): Matriz de coeficientes técnicos (n x n)
    
    Retorna:
        ndarray: Matriz L (n x n) — inversa de Leontief
    
    Levanta:
        np.linalg.LinAlgError se (I - A) for singular
    """
    n = A.shape[0]
    I = np.eye(n)
    M = I - A
    
    # Verificar determinante antes de inverter
    det = np.linalg.det(M)
    if abs(det) < 1e-10:
        raise np.linalg.LinAlgError(
            f"Matriz (I-A) é singular ou mal-condicionada: det = {det:.2e}"
        )
    
    # Resolver (I - A) @ L = I (mais estável que inverter diretamente)
    L = np.linalg.solve(M, I)
    return L
```

### 3.3 Converter choque de demanda: Produto → Atividade

```python
def converter_choque_produto_para_atividade(delta_f_produto, D):
    """
    Converte um choque de demanda do espaço-produto para o espaço-atividade.
    
    f_atividade = D @ f_produto
    
    Onde D_ij = participação da atividade i na produção do produto j.
    
    Parâmetros:
        delta_f_produto (array): Choque em cada produto (m,)
        D (ndarray): Matriz D (n x m) — market share
    
    Retorna:
        array: Choque em cada atividade (n,)
    """
    return D @ delta_f_produto
```

### 3.4 Simular choque — Modelo Aberto

```python
def simular_choque_aberto(L, delta_f):
    """
    Simula choque de demanda no modelo aberto de Leontief.
    
    Δx = L × Δf
    
    Parâmetros:
        L (ndarray): Inversa de Leontief (n x n)
        delta_f (array): Vetor de choque na demanda final (n,)
    
    Retorna:
        dict: Efeitos direto, indireto e total, e vetores de produção
    
    Exemplo:
        >>> L = np.array([[1.2, 0.3], [0.2, 1.1]])
        >>> delta_f = np.array([100, 0])
        >>> resultado = simular_choque_aberto(L, delta_f)
        >>> resultado['efeito_total']
        array([120.,  20.])
    """
    # Efeito total: Δx_total = L @ Δf
    delta_x_total = L @ delta_f
    
    # Efeito direto: é o próprio choque
    efeito_direto = delta_f.copy()
    
    # Efeito indireto: total - direto
    efeito_indireto = delta_x_total - efeito_direto
    
    # Produção inicial e final
    # (Nota: se X0 não for fornecido, calcula-se apenas a variação)
    
    return {
        'efeito_direto': efeito_direto,
        'efeito_indireto': efeito_indireto,
        'efeito_total': delta_x_total,
        'multiplicador_aberto': float(delta_x_total.sum() / delta_f.sum())
    }
```

### 3.5 Converter consumo das famílias: Produto → Atividade

```python
def converter_consumo_familias(h_produto, D):
    """
    Converte o vetor de consumo das famílias do espaço-produto (m)
    para o espaço-atividade (n).
    
    h_atividade = D @ h_produto
    
    Parâmetros:
        h_produto (array): Consumo das famílias por produto (m,)
        D (ndarray): Matriz D (n x m) — market share
    
    Retorna:
        array: Consumo das famílias por atividade (n,)
    """
    return D @ h_produto
```

### 3.6 Estimar coeficientes de renda do trabalho

```python
def estimar_coeficiente_trabalho(A, alfa=0.6):
    """
    Estima o coeficiente de renda do trabalho (l_j) para cada setor.
    
    l_j = α × (1 - Σ_i a_ij)
    
    Onde:
      - (1 - Σ_i a_ij) = coeficiente de valor adicionado do setor j
      - α = parcela dos salários no valor adicionado
    
    Parâmetros:
        A (ndarray): Matriz de coeficientes técnicos (n x n)
        alfa (float): Parcela dos salários no VA (default 0.6)
                      Faixa típica Brasil: 0,58–0,63 (SCN/IBGE 2010-2020)
    
    Retorna:
        array: Coeficiente de renda do trabalho por setor (n,)
    """
    coef_va = 1 - A.sum(axis=0)  # VA por unidade produzida
    l = alfa * coef_va
    return l


def extrair_remuneracoes_reais(sheet_usos, coluna_remuneracao, producao_total):
    """
    Extrai o coeficiente de trabalho REAL da Tabela de Usos da MIP.
    
    l_j_real = Remunerações_do_setor_j / Produção_total_do_setor_j
    
    Parâmetros:
        sheet_usos: Planilha "02" da MIP (Usos)
        coluna_remuneracao (int): Índice da coluna de remunerações
        producao_total (array): Vetor de produção total por setor (n,)
    
    Retorna:
        array: Coeficiente de renda do trabalho real (n,)
    """
    # Extrair valores de remuneração (linha específica na Tabela de Usos)
    # Adaptar conforme a estrutura do arquivo .xls da MIP
    remuneracoes = []
    for r in range(5, 5 + len(producao_total)):
        val = sheet_usos.cell_value(r, coluna_remuneracao)
        if isinstance(val, (int, float)):
            remuneracoes.append(val)
        else:
            remuneracoes.append(0.0)
    
    remuneracoes = np.array(remuneracoes)
    with np.errstate(divide='ignore', invalid='ignore'):
        l_real = np.where(producao_total > 0, remuneracoes / producao_total, 0)
    return l_real
```

### 3.7 Construir modelo fechado (com famílias)

```python
def construir_modelo_fechado(A, h, l):
    """
    Expande a matriz A para incluir as famílias como setor adicional.
    
    A_fechado = [[A,  h],
                 [l^T, 0]]
    
    Onde:
      - h: vetor de coeficientes de consumo das famílias (coluna n+1)
      - l^T: vetor de coeficientes de renda do trabalho (linha n+1)
    
    Parâmetros:
        A (ndarray): Matriz de coeficientes técnicos (n x n)
        h (array): Coeficiente de consumo das famílias (n,)
        l (array): Coeficiente de renda do trabalho (n,)
    
    Retorna:
        ndarray: Matriz expandida ((n+1) x (n+1))
    """
    n = A.shape[0]
    A_closed = np.zeros((n + 1, n + 1))
    
    # Bloco da matriz original
    A_closed[:n, :n] = A
    
    # Coluna das famílias (consumo)
    A_closed[:n, n] = h
    
    # Linha das famílias (renda do trabalho)
    A_closed[n, :n] = l
    
    # Canto inferior direito: 0 (famílias não consomem de si mesmas)
    A_closed[n, n] = 0.0
    
    return A_closed
```

### 3.8 Simular choque — Modelo Fechado

```python
def simular_choque_fechado(A, h, l, delta_f, X0=None):
    """
    Simula choque de demanda no modelo fechado (com famílias).
    
    O modelo fechado captura três efeitos:
      - Direto: o choque inicial na demanda final
      - Indireto: compras inter-setoriais ativadas pelo choque
      - Induzido: consumo adicional das famílias financiado pela
                  renda do trabalho gerada nos efeitos direto e indireto
    
    Parâmetros:
        A (ndarray): Matriz de coeficientes técnicos (n x n)
        h (array): Coeficiente de consumo das famílias (n,)
        l (array): Coeficiente de renda do trabalho (n,)
        delta_f (array): Vetor de choque na demanda final (n,)
        X0 (array, opcional): Vetor de produção inicial (n,)
    
    Retorna:
        dict: Efeitos direto, indireto, induzido, total e multiplicadores
    """
    n = A.shape[0]
    
    # --- Modelo Aberto (para decompor indireto vs. induzido) ---
    L_aberto = calcular_inversa_leontief(A)
    resultado_aberto = simular_choque_aberto(L_aberto, delta_f)
    delta_x_aberto = resultado_aberto['efeito_total']
    
    # --- Modelo Fechado ---
    A_closed = construir_modelo_fechado(A, h, l)
    
    # Expandir o choque: colocar 0 na linha das famílias
    delta_f_closed = np.zeros(n + 1)
    delta_f_closed[:n] = delta_f
    
    # Calcular inversa expandida
    L_closed = calcular_inversa_leontief(A_closed)
    
    # Impacto total no sistema fechado
    delta_x_closed = L_closed @ delta_f_closed
    
    # Decomposição dos efeitos
    efeito_direto = delta_f.copy()
    efeito_indireto = delta_x_aberto[:n] - efeito_direto
    efeito_induzido = delta_x_closed[:n] - delta_x_aberto[:n]
    efeito_total_fechado = delta_x_closed[:n]
    
    # Multiplicadores
    soma_choque = delta_f.sum()
    if soma_choque > 0:
        mult_aberto = float(delta_x_aberto.sum() / soma_choque)
        mult_fechado = float(delta_x_closed[:n].sum() / soma_choque)
        mult_renda = mult_fechado - mult_aberto + 1  # impacto induzido / choque
    else:
        mult_aberto = 0.0
        mult_fechado = 0.0
        mult_renda = 0.0
    
    return {
        'efeito_direto': efeito_direto,
        'efeito_indireto': efeito_indireto,
        'efeito_induzido': efeito_induzido,
        'efeito_total_aberto': delta_x_aberto,
        'efeito_total_fechado': efeito_total_fechado,
        'multiplicador_aberto': mult_aberto,
        'multiplicador_renda': mult_renda,
        'multiplicador_fechado': mult_fechado,
        'L_aberto': L_aberto,
        'L_fechado': L_closed
    }
```

### 3.9 Identificar top setores beneficiados

```python
def top_setores_impacto(efeito_total, nomes_setores=None, top_n=5):
    """
    Identifica os setores mais beneficiados pelo choque.
    
    Parâmetros:
        efeito_total (array): Vetor de impacto total (n,)
        nomes_setores (list, opcional): Nomes dos setores
        top_n (int): Número de setores a listar
    
    Retorna:
        pd.DataFrame: Tabela com setores ordenados por impacto
    """
    if nomes_setores is None:
        nomes_setores = [f"Setor {i}" for i in range(len(efeito_total))]
    
    # Criar DataFrame
    df = pd.DataFrame({
        'setor': nomes_setores,
        'impacto_r': efeito_total,
        'participacao_pct': efeito_total / efeito_total.sum() * 100
    })
    
    # Ordenar do maior para o menor (excluir o próprio setor do choque if zero)
    df = df.sort_values('impacto_r', ascending=False).head(top_n)
    df.index = range(1, top_n + 1)
    
    return df
```

### 3.10 Calcular impacto no emprego

```python
def calcular_impacto_emprego(delta_x, coeficientes_emprego):
    """
    Converte variação na produção em variação no emprego.
    
    ΔE = ê × Δx
    
    Onde ê_i = emprego por unidade de produção no setor i.
    
    Parâmetros:
        delta_x (array): Variação na produção por setor (n,)
        coeficientes_emprego (array): Empregos por R$ produzido (n,)
    
    Retorna:
        array: Variação no emprego por setor (n,)
    
    Exemplo:
        # Coeficiente: 10 empregos / R$ 1 milhão produzido
        coef = np.array([10e-6, 5e-6])
        delta_x = np.array([100e6, 50e6])  # R$ 100 MM e R$ 50 MM
        calcular_impacto_emprego(delta_x, coef)
        # array([1000.,  250.])
    """
    return coeficientes_emprego * delta_x
```

### 3.11 Relatório completo de impacto

```python
def gerar_relatorio_impacto(resultado, top_setores, nome_regiao="Região",
                            nome_choque="Choque", moeda="R$"):
    """
    Gera um dicionário com o relatório completo dos resultados.
    
    Parâmetros:
        resultado (dict): Retorno de simular_choque_fechado()
        top_setores (pd.DataFrame): Top setores impactados
        nome_regiao (str): Nome da região analisada
        nome_choque (str): Descrição do choque
        moeda (str): Símbolo da moeda
    
    Retorna:
        dict: Relatório estruturado para exportação
    """
    soma_direto = resultado['efeito_direto'].sum()
    soma_indireto = resultado['efeito_indireto'].sum()
    soma_induzido = resultado['efeito_induzido'].sum()
    soma_total_aberto = resultado['efeito_total_aberto'].sum()
    soma_total_fechado = resultado['efeito_total_fechado'].sum()
    
    relatorio = {
        'resumo': {
            'regiao': nome_regiao,
            'choque': nome_choque,
            'moeda': moeda,
            'data_execucao': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
        },
        'efeitos': {
            'direto': {
                'valor': soma_direto,
                'participacao_pct': round(soma_direto / soma_total_fechado * 100, 2)
            },
            'indireto': {
                'valor': soma_indireto,
                'participacao_pct': round(soma_indireto / soma_total_fechado * 100, 2)
            },
            'induzido': {
                'valor': soma_induzido,
                'participacao_pct': round(soma_induzido / soma_total_fechado * 100, 2)
            },
            'total_aberto': soma_total_aberto,
            'total_fechado': soma_total_fechado
        },
        'multiplicadores': {
            'aberto': round(resultado['multiplicador_aberto'], 4),
            'renda': round(resultado['multiplicador_renda'], 4),
            'fechado': round(resultado['multiplicador_fechado'], 4)
        },
        'top_setores_indiretos': top_setores.to_dict(orient='records')
    }
    
    return relatorio
```

---

## 4. Exemplo Completo

```python
# ===========================================================================
# Exemplo: Simulação de choque de R$ 500 milhões no setor automotivo (MG)
# Nível 67 setores
# ===========================================================================

import numpy as np
import pandas as pd

# --- Dados simulados para demonstração ---
# (Numa execução real, estes dados viriam da MIP e RAIS)

n = 67  # número de setores
np.random.seed(42)

# Matriz A regionalizada (simulada)
A_reg = np.random.dirichlet(np.ones(n), size=n) * 0.3
np.fill_diagonal(A_reg, A_reg.diagonal() * 0.8)  # reduz auto-consumo

# Consumo das famílias (espaço-atividade, simulado)
h = np.random.uniform(0.01, 0.05, size=n)
h = h / h.sum() * 0.3  # ~30% do VA vai para consumo familiar

# Coeficiente de renda do trabalho
l = estimar_coeficiente_trabalho(A_reg, alfa=0.6)

# Choque: R$ 500 milhões no setor automotivo
# Setor automotivo (código 2991) ≈ índice 27 no nível 67
delta_f = np.zeros(n)
delta_f[27] = 500_000_000  # R$ 500 milhões

# --- Executar simulação ---
resultado = simular_choque_fechado(
    A=A_reg,
    h=h,
    l=l,
    delta_f=delta_f
)

# --- Top setores indiretos ---
nomes = [f"Setor_{i}" for i in range(n)]
nomes[27] = "Automotivo (2991)"

top5 = top_setores_impacto(
    resultado['efeito_indireto'] + resultado['efeito_induzido'],
    nomes_setores=nomes,
    top_n=5
)

# --- Relatório ---
relatorio = gerar_relatorio_impacto(
    resultado, top5,
    nome_regiao="Minas Gerais",
    nome_choque="R$ 500 MM no setor automotivo (2991)"
)

print("=== RESUMO DOS RESULTADOS ===")
print(f"Região: {relatorio['resumo']['regiao']}")
print(f"Choque: {relatorio['resumo']['choque']}")
print()
print("--- Efeitos ---")
for efeito, dados in relatorio['efeitos'].items():
    if isinstance(dados, dict):
        print(f"  {efeito}: R$ {dados['valor']/1e6:.2f} MM ({dados['participacao_pct']:.1f}%)")
print(f"  Total (aberto): R$ {relatorio['efeitos']['total_aberto']/1e6:.2f} MM")
print(f"  Total (fechado): R$ {relatorio['efeitos']['total_fechado']/1e6:.2f} MM")
print()
print("--- Multiplicadores ---")
for k, v in relatorio['multiplicadores'].items():
    print(f"  {k}: {v:.4f}")
print()
print("--- Top 5 setores indiretos+induzidos ---")
print(top5.to_string())
```

---

## 5. Tabelas de Referência

### 5.1 Decomposição dos efeitos

| Efeito | Modelo | Descrição | Fórmula |
|---|---|---|---|
| **Direto** | Aberto + Fechado | O choque inicial na demanda final | Δf |
| **Indireto** | Aberto + Fechado | Compras inter-setoriais ativadas pela cadeia de fornecedores | (L - I) @ Δf |
| **Induzido** | Fechado apenas | Consumo adicional das famílias com a renda gerada nos efeitos direto+indireto | L_closed[:n] @ Δf_closed - L @ Δf |

### 5.2 Multiplicadores

| Multiplicador | Interpretação | Exemplo |
|---|---|---|
| **Aberto** | Para cada R$ 1 de demanda final, quanto a economia regional produz no total (direto + indireto) | 1,50 → R$ 1 → R$ 1,50 |
| **Renda** | Impacto induzido dividido pelo choque inicial | 2,05 → efeito induzido = 2,05 × choque |
| **Fechado** | Para cada R$ 1 de demanda final, produção total incluindo efeito induzido | 3,55 → R$ 1 → R$ 3,55 |

### 5.3 Parâmetro α (parcela salarial no VA)

| Contexto | α sugerido | Fonte |
|---|---|---|
| Padrão Brasil (média 2010-2020) | 0,60 | SCN/IBGE |
| Indústria de transformação | 0,55–0,65 | IBGE |
| Serviços intensivos em trabalho | 0,65–0,75 | IBGE |
| Agropecuária | 0,50–0,60 | IBGE |
| **Extrair valor real da MIP** | **Remunerações / VA** | **Tabela de Usos (recomendado)** |

### 5.4 Níveis da MIP Ibge

| Nível | Setores | Tamanho arquivo | Uso recomendado |
|---|---|---|---|
| 12 | 12 × 12 | ~132 KB | Visão geral, choques macro |
| 20 | 20 × 20 | ~176 KB | Análise intermediária |
| 67 | 127 × 67 | ~1,2 MB | Impactos setoriais específicos (ex: automotivo, mineração) |

---

## 6. Regras

### O que SEMPRE fazer
- Verificar se det(I - A) > 1e-10 **antes** de inverter a matriz
- Usar `np.linalg.solve(M, I)` em vez de `np.linalg.inv(M)` por estabilidade
- Converter choques do espaço-produto para espaço-atividade usando a matriz D
- Documentar o valor de α (parcela salarial) e sua fonte
- Separar efeitos em direto, indireto e induzido sempre que usar modelo fechado
- Validar a matriz A_reg antes de qualquer simulação (requer skill `regionalizacao-mip`)
- Registrar data/hora de execução, parâmetros e fontes no relatório

### O que NUNCA fazer
- Nunca inverter (I - A) sem verificar o determinante — se for ≈ 0, a inversa é instável
- Nunca usar coeficiente de trabalho genérico quando os dados de remuneração estão disponíveis na MIP
- Nunca somar efeitos de modelos aberto e fechado como se fossem independentes
- Nunca apresentar multiplicadores sem especificar se são abertos ou fechados
- Nunca ignorar vazamentos regionais — o multiplicador regional é SEMPRE menor que o nacional

### Quando algo falhar
- **det(I - A) ≈ 0**: verificar se alguma coluna soma ≥ 1 (setor 100% dependente de insumos); ajustar coeficientes suspeitos e rebalancear
- **Multiplicador < 1**: verificar se a matriz A é válida (coeficientes não-negativos, soma colunas < 1); possível erro na regionalização
- **L convergência numérica**: para matrizes grandes (67+), usar `scipy.linalg.solve` em vez de `np.linalg.solve`
- **Efeito induzido negativo**: verificar se h (consumo das famílias) e l (coeficiente de trabalho) são consistentes; h deve ser positivo e l deve estar entre 0 e 1
- **Dados de remuneração ausentes na MIP**: usar estimativa com α = 0,60 e documentar a limitação

---

## 7. Pipeline Decisório

```
1. RECEBER CHOQUE
   ├── Qual setor? → Identificar código CNAE/atividade
   ├── Qual valor? → Definir Δf em R$
   └── Qual região? → Carregar A_reg correspondente

2. PREPARAR DADOS
   ├── A_reg existe e é válida? → Validar (det, col_sums, não-negatividade)
   ├── Consumo famílias (h) disponível? → Extrair da MIP ou estimar
   ├── Coef. trabalho (l) disponível? → Extrair remunerações ou estimar com α
   └── Choque em produto ou atividade? → Se produto: converter via D

3. MODELO ABERTO
   ├── Calcular L = (I - A)^{-1}
   ├── Δx_aberto = L @ Δf
   └── Decompor: direto = Δf, indireto = Δx_aberto - Δf

4. MODELO FECHADO
   ├── Construir A_fechado = [[A, h], [l^T, 0]]
   ├── Calcular L_fechado
   ├── Δx_fechado = L_fechado @ Δf_fechado
   └── Decompor: induzido = Δx_fechado - Δx_aberto

5. RESULTADOS
   ├── Calcular multiplicadores (aberto, renda, fechado)
   ├── Identificar top N setores
   ├── Se disponível: converter Δx em Δemprego
   └── Gerar relatório
```

---

## 8. Glossário

| Termo | Definição |
|---|---|
| **MIP** | Matriz Insumo-Produto |
| **A** | Matriz de coeficientes técnicos (a_ij = insumo do setor i para produzir 1 unidade do setor j) |
| **L** | Inversa de Leontief (I - A)⁻¹ — captura efeitos diretos + indiretos |
| **Δf** | Vetor de choque na demanda final |
| **Δx** | Variação total na produção |
| **Efeito direto** | O próprio choque inicial |
| **Efeito indireto** | Compras adicionais entre setores ao longo da cadeia |
| **Efeito induzido** | Consumo adicional das famílias com a renda gerada |
| **Matriz D** | Market share: participação de cada atividade na produção de cada produto (n × m) |
| **h** | Coeficiente de consumo das famílias (coluna adicional no modelo fechado) |
| **l** | Coeficiente de renda do trabalho (linha adicional no modelo fechado) |
| **VA** | Valor Adicionado |
| **α** | Parcela dos salários no valor adicionado |
| **Multiplicador aberto** | Razão entre Δx_total (aberto) e o choque inicial |
| **Multiplicador fechado** | Razão entre Δx_total (fechado) e o choque inicial |
