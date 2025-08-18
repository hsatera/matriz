import streamlit as st
import pandas as pd
import re
import plotly.express as px
import nltk
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import os

# Baixar recursos do NLTK se necessário.
# A função nltk.download() já verifica se os dados existem antes de baixar,
# então o bloco try/except não é mais necessário e causava erro em versões recentes.
nltk.download('stopwords', quiet=True)

# --- Configuração da Página Streamlit ---
st.set_page_config(layout="wide", page_title="Filtro de Artigos Científicos com Resumo")

st.title("📚 MATRIZ: Revisão de Escopo")
st.markdown("Use os filtros na barra lateral para encontrar artigos específicos e veja o resumo da distribuição nos gráficos.")

# --- Carregamento e Pré-processamento dos Dados ---
@st.cache_data
def load_data(file_path):
    """
    Carrega os dados do arquivo CSV e preenche valores ausentes.
    Usa st.cache_data para otimizar o carregamento de dados.
    """
    df = pd.read_csv(file_path)
    # Preenche colunas com valores padrão se estiverem ausentes
    df['notes'] = df['notes'].fillna('')
    df['abstract'] = df['abstract'].fillna('')
    if 'authors' not in df.columns:
        df['authors'] = 'Não informado'
    if 'url' not in df.columns:
        df['url'] = ''
    # Garante que 'url' e 'PDF files' são do tipo string após preencher NaNs
    df['url'] = df['url'].astype(str)
    if 'PDF files' in df.columns: # Verifica se a coluna existe antes de processar
        df['PDF files'] = df['PDF files'].fillna('').astype(str)
    else:
        df['PDF files'] = '' # Adiciona uma coluna vazia se não existir
    return df

try:
    df = load_data("selecionados.csv")
except FileNotFoundError:
    st.error("Erro: O arquivo 'selecionados.csv' não foi encontrado. Por favor, certifique-se de que o arquivo está no mesmo diretório.")
    st.stop() # Interrompe a execução do aplicativo se o arquivo não for encontrado

# --- Função para extrair labels ---
def extract_labels(label_string):
    """
    Extrai as labels de uma string formatada como 'RAYYAN-LABELS: ...'.
    Retorna uma lista de strings de labels.
    """
    if pd.isna(label_string): # Verifica se a string é NaN (Not a Number)
        return []
    # Usa regex para encontrar a parte das labels
    match = re.search(r'RAYYAN-LABELS:\s*(.+)', label_string)
    if match:
        labels_part = match.group(1).strip()
        # Limpa a string de caracteres indesejados
        labels_part = labels_part.replace('""', '').replace('{', '').replace('}', '')
        # Encontra todas as labels separadas por ',' ou '|'
        labels = re.findall(r'[^,\|]+', labels_part)
        # Retorna uma lista de labels limpas e sem espaços em branco
        return [label.strip() for label in labels if label.strip()]
    return []

# Aplica a função para criar uma nova coluna com as labels parseadas
df['Parsed_Labels'] = df['notes'].apply(extract_labels)

# --- Filtros Disponíveis (Listas de opções para os multiselects) ---
PAISES = ['Angola', 'Argentina', 'Brasil', 'Cuba', 'Portugal', 'Uruguai']
METODOLOGIAS = ['Estudo misto (quali-quanti)', 'Estudo Qualitativo', 'Estudo quantitativo', 'Revisão literatura']
TIPOS_ESTUDO = ['Definição Ações Coletivas Cuidado', 'Reflexão teórica Ações Coletivas de Cuidado', 'Relato Experiência Ações Coletivas de Cuidado']
EIXOS_MATRIZ = ['Doenças crônicas', 'Saúde Mental', 'Saúde Bucal', 'Infância e Adolescência', 'Gênero e Sexualidade', 'Deficiência Intelectual']
IDIOMAS = ['Língua Espanhola', 'Língua Inglesa', 'Língua Portuguesa']

# --- Barra Lateral de Filtros ---
st.sidebar.header("⚙️ Filtros")

# Cria os widgets de multiselect na barra lateral para cada categoria de filtro
selected_paises = st.sidebar.multiselect("País", PAISES)
selected_metodologias = st.sidebar.multiselect("Metodologia", METODOLOGIAS)
selected_tipos_estudo = st.sidebar.multiselect("Tipo de Estudo", TIPOS_ESTUDO)
selected_eixos_matriz = st.sidebar.multiselect("Eixo Matriz", EIXOS_MATRIZ)
selected_idiomas = st.sidebar.multiselect("Idioma", IDIOMAS)

# --- Lógica de Filtragem ---
filtered_df = df.copy() # Cria uma cópia do DataFrame original para aplicar os filtros

def check_labels(article_labels, selected_options):
    """
    Verifica se alguma das labels de um artigo corresponde às opções selecionadas.
    Retorna True se houver correspondência ou se nenhuma opção foi selecionada.
    """
    if not selected_options: # Se nenhuma opção foi selecionada, todos os artigos passam
        return True
    article_labels = article_labels if isinstance(article_labels, list) else [] # Garante que é uma lista
    # Converte labels e opções selecionadas para minúsculas para comparação case-insensitive
    article_labels_lower = [label.lower().strip() for label in article_labels]
    selected_options_lower = [option.lower().strip() for option in selected_options]
    # Retorna True se qualquer opção selecionada estiver nas labels do artigo
    return any(option in article_labels_lower for option in selected_options_lower)

# Aplica os filtros iterativamente ao DataFrame
for filtro, selecao in zip(
    ['País', 'Metodologia', 'Tipo de Estudo', 'Eixo Matriz', 'Idioma'],
    [selected_paises, selected_metodologias, selected_tipos_estudo, selected_eixos_matriz, selected_idiomas]
):
    if selecao: # Aplica o filtro apenas se alguma opção foi selecionada para a categoria
        filtered_df = filtered_df[filtered_df['Parsed_Labels'].apply(lambda x: check_labels(x, selecao))]

# --- Seção: Gráficos de Resumo no Topo (Revertido para o layout original) ---
if not filtered_df.empty:
    st.header("📊 Visão Geral dos Resultados Filtrados")
    st.subheader(f"Total de Artigos Encontrados: {len(filtered_df)}") # Exibe o total de artigos

    filters_to_plot = {
        "País": PAISES,
        "Metodologia": METODOLOGIAS,
        "Tipo de Estudo": TIPOS_ESTUDO,
        "Eixo Matriz": EIXOS_MATRIZ,
        "Idioma": IDIOMAS
    }

    def plot_summary_component(data_frame, filter_name, options_list):
        """
        Prepara os dados e gera o gráfico de resumo para uma categoria específica.
        Escolhe entre gráfico de barras e pizza com base na categoria.
        """
        counts = {}
        for labels_list in data_frame['Parsed_Labels']:
            if isinstance(labels_list, list):
                for label in labels_list:
                    original_label = next((opt for opt in options_list if opt.lower().strip() == label.lower().strip()), None)
                    if original_label:
                        counts[original_label] = counts.get(original_label, 0) + 1

        if counts:
            summary_df = pd.DataFrame(counts.items(), columns=['Categoria', 'Contagem'])
            summary_df = summary_df.sort_values(by='Contagem', ascending=False)

            # Escolhe o tipo de gráfico com base na categoria
            if filter_name in ["Metodologia", "Tipo de Estudo"]:
                fig = px.pie(summary_df, values='Contagem', names='Categoria',
                             title=f'Distribuição por {filter_name}',
                             color_discrete_sequence=px.colors.sequential.Viridis,
                             hole=0.3) # Gráfico de donut
                fig.update_traces(textposition='inside', textinfo='percent+label',
                                  marker=dict(line=dict(color='#000000', width=1)))
            else:
                fig = px.bar(summary_df, x='Categoria', y='Contagem',
                             title=f'Distribuição por {filter_name}',
                             labels={'Contagem': 'Nº Artigos', 'Categoria': filter_name},
                             color='Contagem', color_continuous_scale=px.colors.sequential.Viridis)
                fig.update_layout(xaxis_title_text='', yaxis_title_text='Número de Artigos')

            fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
            return fig
        return None

    # Exibe os gráficos em colunas (2 por linha)
    filter_names = list(filters_to_plot.keys())
    for i in range(0, len(filter_names), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(filter_names):
                name = filter_names[i + j]
                fig = plot_summary_component(filtered_df, name, filters_to_plot[name])
                if fig:
                    cols[j].plotly_chart(fig, use_container_width=True)

    # --- Seção da WordCloud ---
    st.header("☁️ Nuvem de Palavras dos Resumos Filtrados")

    # Conjuntos gramaticais a excluir para a WordCloud
    PRONOMES = {
        'eu', 'tu', 'ele', 'ela', 'nós', 'vós', 'eles', 'elas',
        'me', 'te', 'se', 'nos', 'vos', 'lhe', 'lhes',
        'mim', 'ti', 'si', 'comigo', 'contigo', 'consigo',
        'meu', 'minha', 'meus', 'minhas',
        'teu', 'tua', 'teus', 'tuas',
        'seu', 'sua', 'seus', 'suas',
        'nosso', 'nossa', 'nossos', 'nossas',
        'vosso', 'vossa', 'vossos', 'vossas',
        'este', 'esta', 'estes', 'estas',
        'esse', 'essa', 'esses', 'essas',
        'aquele', 'aquela', 'aqueles', 'aquelas',
        'isto', 'isso', 'aquilo',
        'alguém', 'ninguém', 'tudo', 'nada', 'cada',
        'quem', 'qual', 'quais', 'quanto', 'quantos', 'quanta', 'quantas',
        'algum', 'alguma', 'alguns', 'algumas',
        'nenhum', 'nenhuma', 'uns', 'umas'
    }

    PREPOSICOES = {
        'a', 'ante', 'após', 'até', 'com', 'contra', 'de', 'desde',
        'em', 'entre', 'para', 'per', 'perante', 'por', 'sem',
        'sob', 'sobre', 'trás'
    }

    ADVERBIOS = {
        'aqui', 'ali', 'lá', 'cá', 'acolá', 'aí', 'perto', 'longe', 'dentro', 'fora',
        'acima', 'abaixo', 'atrás', 'adiante', 'antes', 'depois', 'cedo', 'tarde',
        'logo', 'já', 'ainda', 'sempre', 'nunca', 'jamais', 'eventualmente',
        'provavelmente', 'possivelmente', 'certamente', 'sim', 'não', 'também',
        'ainda', 'só', 'apenas', 'quase', 'muito', 'pouco', 'bastante', 'demais',
        'tanto', 'quanto', 'mais', 'menos', 'mal', 'bem', 'melhor', 'pior',
        'devagar', 'depressa', 'calmamente', 'rapidamente'
    }

    CONTRAIDOS = {
        'do', 'da', 'dos', 'das', 'no', 'na', 'nos', 'nas',
        'pelo', 'pela', 'pelos', 'pelas',
        'deste', 'desta', 'desses', 'dessas', 'neste', 'nesta', 'nestes', 'nestas',
        'daquele', 'daquela', 'daqueles', 'daquelas', 'à', 'às', 'ao', 'aos'
    }

    def filtrar_palavras(texto):
        """
        Filtra palavras de um texto, removendo stopwords, pronomes, preposições,
        advérbios e contrações em português.
        """
        stop_words = set(stopwords.words('portuguese'))
        palavras_excluir = stop_words.union(PRONOMES, PREPOSICOES, ADVERBIOS, CONTRAIDOS)

        tokenizer = RegexpTokenizer(r'\b\w+\b') # Tokeniza por palavras alfanuméricas
        palavras = tokenizer.tokenize(texto.lower()) # Converte para minúsculas

        return [
            palavra for palavra in palavras
            if palavra.isalpha() # Garante que é uma palavra alfabética
            and len(palavra) >= 4 # Filtra palavras com 4 ou mais caracteres
            and palavra not in palavras_excluir # Exclui palavras da lista de exclusão
        ]

    def gerar_wordcloud(palavras_filtradas):
        """
        Gera uma WordCloud a partir de uma lista de palavras filtradas.
        Retorna um objeto Figure do Matplotlib.
        """
        if not palavras_filtradas:
            return None # Retorna None se não houver palavras para gerar a nuvem

        texto_filtrado = ' '.join(palavras_filtradas)
        wordcloud = WordCloud(
            width=800,
            height=400, # Ajustado para melhor visualização no Streamlit
            background_color='white',
            stopwords=STOPWORDS, # Usa o conjunto de stopwords padrão da WordCloud
            max_words=200,
            colormap='viridis',
            collocations=False # Para evitar que a WordCloud crie pares de palavras
        ).generate(texto_filtrado)

        fig, ax = plt.subplots(figsize=(10, 5), facecolor=None) # Cria uma figura e eixos
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off") # Remove os eixos
        plt.tight_layout(pad=0)
        return fig

    # Agrega todos os resumos dos artigos filtrados
    all_abstracts_text = " ".join(filtered_df['abstract'].dropna().tolist())

    if all_abstracts_text.strip():
        palavras_filtradas = filtrar_palavras(all_abstracts_text)
        if palavras_filtradas:
            wordcloud_fig = gerar_wordcloud(palavras_filtradas)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig) # Exibe a WordCloud no Streamlit
                plt.close(wordcloud_fig) # Fecha a figura para liberar memória
            else:
                st.info("Nenhuma palavra significativa encontrada para gerar a nuvem de palavras.")
        else:
            st.info("Nenhuma palavra significativa encontrada para gerar a nuvem de palavras após a filtragem.")
    else:
        st.info("Nenhum resumo disponível nos artigos filtrados para gerar a nuvem de palavras.")


    # --- Resultados Detalhados dos Artigos ---
    st.subheader(f"🔎 Artigos Encontrados ({len(filtered_df)})")

    # Itera sobre cada linha do DataFrame filtrado para exibir os detalhes do artigo
    for _, row in filtered_df.iterrows():
        st.markdown("---") # Separador visual entre os artigos
        st.markdown(f"**Título:** {row['title']}")
        st.markdown(f"**Autores:** {row['authors']}")
        st.markdown(f"**Ano:** {int(row['year']) if pd.notna(row['year']) else 'Não informado'}")
        st.markdown(f"**Labels:** {', '.join(row['Parsed_Labels']) if row['Parsed_Labels'] else 'Nenhum'}")

        # Expander para mostrar o resumo (abstract)
        if row['abstract'].strip():
            with st.expander("Ver Resumo"):
                st.write(row['abstract'])

        # Link para o artigo, se disponível
        if row['url'] and row['url'].strip():
            st.markdown(f"[🔗 Link para o Artigo]({row['url']})")

        # Informação sobre PDF disponível, se houver
        if row['PDF files'] and row['PDF files'].strip():
            st.info(f"📄 PDF disponível: {row['PDF files']}")
else:
    st.info("Nenhum artigo encontrado com os filtros selecionados. Ajuste os filtros na barra lateral.")

st.markdown("---")

# --- Próximos Passos e Sugestões ---
st.header("💡 Próximos Passos e Sugestões para Aprimoramento")
st.markdown("""
Para tornar esta aplicação ainda mais poderosa e interativa, considere as seguintes melhorias:

* **Gráficos de Treemap/Sunburst:** Explore o uso de `plotly.express.treemap` ou `plotly.express.sunburst` para visualizar dados hierárquicos, por exemplo, a distribuição de "Eixos Matriz" dentro de cada "País". Isso pode revelar padrões interessantes em dados aninhados.
* **Filtros Interativos nos Gráficos:** Implemente a funcionalidade de clicar em uma barra ou fatia do gráfico para que os artigos correspondentes sejam automaticamente filtrados na lista abaixo. Isso adicionaria uma camada de interatividade e exploração de dados.
* **Tabela Interativa para Artigos:** Em vez de exibir os artigos um a um, utilize `st.dataframe` ou bibliotecas como `streamlit-aggrid` para apresentar os resultados em uma tabela interativa. Isso permitiria aos usuários pesquisar, classificar e paginar os artigos de forma mais eficiente.
* **Nuvem de Palavras:** Para visualizar os termos mais frequentes nos títulos ou resumos dos artigos, uma nuvem de palavras (`wordcloud`) pode ser uma adição visualmente interessante. Isso exigiria um pré-processamento de texto adicional para extrair e contar as palavras.
* **Exportação de Dados:** Adicione um botão para permitir que os usuários exportem os artigos filtrados (por exemplo, em formato CSV ou Excel) para análise posterior.
* **Funcionalidade de Busca por Texto:** Inclua uma caixa de texto para que os usuários possam pesquisar artigos por palavras-chave no título, resumo ou autores.
""")

st.caption("🔧 Desenvolvido com Streamlit e Plotly")
