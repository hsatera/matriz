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
import io

# Baixar recursos do NLTK se necessário.
# A função nltk.download() já verifica se os dados existem antes de baixar.
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
    """
    df = pd.read_csv(file_path)
    df['notes'] = df['notes'].fillna('')
    df['abstract'] = df['abstract'].fillna('')
    df['keywords'] = df['keywords'].fillna('Não informado')
    if 'authors' not in df.columns:
        df['authors'] = 'Não informado'
    if 'url' not in df.columns:
        df['url'] = ''
    df['url'] = df['url'].astype(str)
    if 'PDF files' in df.columns:
        df['PDF files'] = df['PDF files'].fillna('').astype(str)
    else:
        df['PDF files'] = ''
    return df

try:
    df = load_data("selecionados.csv")
except FileNotFoundError:
    st.error("Erro: O arquivo 'selecionados.csv' não foi encontrado. Por favor, certifique-se de que o arquivo está no mesmo diretório.")
    st.stop()

# --- Funções de Extração ---
def extract_labels(label_string):
    """
    Extrai as labels de uma string formatada como 'RAYYAN-LABELS: ...'.
    """
    if pd.isna(label_string):
        return []
    match = re.search(r'RAYYAN-LABELS:\s*(.+)', label_string)
    if match:
        labels_part = match.group(1).strip().replace('""', '').replace('{', '').replace('}', '')
        labels = re.findall(r'[^,\|]+', labels_part)
        return [label.strip() for label in labels if label.strip()]
    return []

# Lista fixa de revisores fornecida pelo usuário
all_reviewers = ['andré', 'daniela', 'luana', 'henrique', 'carolina', 'humberto', 'mauricio', 'rodrigo']

# Cria uma nova coluna 'Reviewers' no DataFrame original, verificando a presença dos nomes na coluna 'notes'
df['Reviewers'] = df['notes'].apply(lambda s: [rev for rev in all_reviewers if rev.lower() in s.lower()])
df['Parsed_Labels'] = df['notes'].apply(extract_labels)

# Adiciona uma nova coluna 'Is_Perola' baseada na presença de 'Pérola' na coluna 'notes'
df['Is_Perola'] = df['notes'].apply(lambda s: 'Sim' if 'pérola' in s.lower() else 'Não')

# --- Definição das Opções de Filtro ---
# Opções para "Sem Label"
SEM_PAIS = "País não especificado"
SEM_METODOLOGIA = "Metodologia não especificada"
SEM_TIPO_ESTUDO = "Tipo de Estudo não especificado"
SEM_EIXO_MATRIZ = "Eixo Matriz não especificado"
SEM_IDIOMA = "Idioma não especificado"

# Listas de opções para os gráficos (sem a opção "Sem Label")
PAISES = ['Angola', 'Argentina', 'Brasil', 'Cuba', 'Portugal', 'Uruguai']
METODOLOGIAS = ['Estudo misto (quali-quanti)', 'Estudo Qualitativo', 'Estudo quantitativo', 'Revisão literatura']
TIPOS_ESTUDO = ['Definição Ações Coletivas Cuidado', 'Reflexão teórica Ações Coletivas de Cuidado', 'Relato Experiência Ações Coletivas de Cuidado']
EIXOS_MATRIZ = ['Doenças crônicas', 'Saúde Mental', 'Saúde Bucal', 'Infância e Adolescência', 'Gênero e Sexualidade', 'Deficiência Intelectual']
IDIOMAS = ['Língua Espanhola', 'Língua Inglesa', 'Língua Portuguesa']
PEROLA_OPTIONS = ['Sim', 'Não']

# Listas de opções para os widgets da barra lateral (com a opção "Sem Label")
PAISES_OPTIONS = [SEM_PAIS] + PAISES
METODOLOGIAS_OPTIONS = [SEM_METODOLOGIA] + METODOLOGIAS
TIPOS_ESTUDO_OPTIONS = [SEM_TIPO_ESTUDO] + TIPOS_ESTUDO
EIXOS_MATRIZ_OPTIONS = [SEM_EIXO_MATRIZ] + EIXOS_MATRIZ
IDIOMAS_OPTIONS = [SEM_IDIOMA] + IDIOMAS
PEROLA_FILTER_OPTIONS = ['Ambos'] + PEROLA_OPTIONS

# --- Barra Lateral de Filtros ---
st.sidebar.header("⚙️ Filtros")

selected_reviewers = st.sidebar.multiselect("Revisor(a)", all_reviewers)
selected_paises = st.sidebar.multiselect("País", PAISES_OPTIONS)
selected_metodologias = st.sidebar.multiselect("Metodologia", METODOLOGIAS_OPTIONS)
selected_tipos_estudo = st.sidebar.multiselect("Tipo de Estudo", TIPOS_ESTUDO_OPTIONS)
selected_eixos_matriz = st.sidebar.multiselect("Eixo Matriz", EIXOS_MATRIZ_OPTIONS)
selected_idiomas = st.sidebar.multiselect("Idioma", IDIOMAS_OPTIONS)
selected_perola = st.sidebar.selectbox("Pérola", PEROLA_FILTER_OPTIONS, index=0)

# --- Lógica de Filtragem Avançada ---
filtered_df = df.copy()

def apply_filter_logic(df, selected_options, all_category_options, no_label_string):
    """
    Aplica a lógica de filtro para uma categoria, lidando com a seleção de "Sem Label".
    """
    if not selected_options:
        return df

    sem_label_selected = no_label_string in selected_options
    specific_labels_selected = [opt for opt in selected_options if opt != no_label_string]
    all_category_options_lower = {opt.lower().strip() for opt in all_category_options}

    def check_row(article_labels):
        article_labels_lower = {label.lower().strip() for label in article_labels}
        
        # Condição 1: O artigo NÃO tem nenhuma label desta categoria
        has_no_category_label = not any(label in all_category_options_lower for label in article_labels_lower)
        
        # Condição 2: O artigo tem pelo menos uma das labels específicas selecionadas
        has_specific_label = any(label.lower().strip() in article_labels_lower for label in specific_labels_selected)

        # Lógica final
        if sem_label_selected and specific_labels_selected:
            return has_no_category_label or has_specific_label
        elif sem_label_selected:
            return has_no_category_label
        elif specific_labels_selected:
            return has_specific_label
        return True

    return df[df['Parsed_Labels'].apply(check_row)]

# Aplica o filtro de revisores
if selected_reviewers:
    filtered_df = filtered_df[filtered_df['Reviewers'].apply(lambda x: any(rev in x for rev in selected_reviewers))]

# Aplica os outros filtros um a um
filtered_df = apply_filter_logic(filtered_df, selected_paises, PAISES, SEM_PAIS)
filtered_df = apply_filter_logic(filtered_df, selected_metodologias, METODOLOGIAS, SEM_METODOLOGIA)
filtered_df = apply_filter_logic(filtered_df, selected_tipos_estudo, TIPOS_ESTUDO, SEM_TIPO_ESTUDO)
filtered_df = apply_filter_logic(filtered_df, selected_eixos_matriz, EIXOS_MATRIZ, SEM_EIXO_MATRIZ)
filtered_df = apply_filter_logic(filtered_df, selected_idiomas, IDIOMAS, SEM_IDIOMA)

# Aplica o filtro de Pérola
if selected_perola != 'Ambos':
    filtered_df = filtered_df[filtered_df['Is_Perola'] == selected_perola]

# --- Função para preparar o DataFrame para exportação ---
def create_export_df(df_filtered):
    """
    Cria um novo DataFrame com as colunas específicas para exportação.
    As colunas são reordenadas conforme a solicitação do usuário.
    """
    export_data = []
    for _, row in df_filtered.iterrows():
        # Extrai labels específicas para as colunas
        country = next((label for label in row['Parsed_Labels'] if label in PAISES), 'Não informado')
        study_type = next((label for label in row['Parsed_Labels'] if label in TIPOS_ESTUDO), 'Não informado')
        methodology = next((label for label in row['Parsed_Labels'] if label in METODOLOGIAS), 'Não informado')
        eixo_matriz = next((label for label in row['Parsed_Labels'] if label in EIXOS_MATRIZ), 'Não informado')
        idioma = next((label for label in row['Parsed_Labels'] if label in IDIOMAS), 'Não informado')
        
        export_row = {
            'Título estudo': row['title'],
            'Autor(es)': row['authors'],
            'Ano': int(row['year']) if pd.notna(row['year']) else 'Não informado',
            'País': country,
            'Tipo de estudo': study_type,
            'Metodologia': methodology,
            'Eixo Matriz': eixo_matriz,
            'Idioma': idioma,
            'Principais resultados': '',
            'Pérola': row['Is_Perola'],
            'Revisor(es)': ', '.join(row['Reviewers']) if row['Reviewers'] else 'Nenhum'
        }
        export_data.append(export_row)
    
    # Cria o DataFrame com a ordem de colunas desejada
    columns_order = [
        'Título estudo', 'Autor(es)', 'Ano', 'País', 'Tipo de estudo',
        'Metodologia', 'Eixo Matriz', 'Idioma', 'Principais resultados', 'Pérola', 'Revisor(es)'
    ]
    
    return pd.DataFrame(export_data, columns=columns_order)

# --- Seção: Gráficos de Resumo no Topo ---
if not filtered_df.empty:
    st.header("📊 Visão Geral dos Resultados Filtrados")
    
    # Prepara os dados para exportação para XLSX
    df_export = create_export_df(filtered_df)
    output = io.BytesIO()
    df_export.to_excel(output, index=False, engine='openpyxl')
    processed_data = output.getvalue()

    st.download_button(
        label="📥 Baixar lista filtrada (.xlsx)",
        data=processed_data,
        file_name='artigos_filtrados.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    
    st.subheader(f"Total de Artigos Encontrados: {len(filtered_df)}")

    filters_to_plot = {
        "País": PAISES,
        "Metodologia": METODOLOGIAS,
        "Tipo de Estudo": TIPOS_ESTUDO,
        "Eixo Matriz": EIXOS_MATRIZ,
        "Idioma": IDIOMAS
    }

    def plot_summary_component(data_frame, filter_name, options_list):
        counts = {}
        for labels_list in data_frame['Parsed_Labels']:
            if isinstance(labels_list, list):
                for label in labels_list:
                    original_label = next((opt for opt in options_list if opt.lower().strip() == label.lower().strip()), None)
                    if original_label:
                        counts[original_label] = counts.get(original_label, 0) + 1
        
        if counts:
            summary_df = pd.DataFrame(counts.items(), columns=['Categoria', 'Contagem']).sort_values(by='Contagem', ascending=False)
            
            if filter_name in ["Metodologia", "Tipo de Estudo"]:
                fig = px.pie(summary_df, values='Contagem', names='Categoria', title=f'Distribuição por {filter_name}',
                             color_discrete_sequence=px.colors.sequential.Viridis, hole=0.3)
                fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1)))
            else:
                fig = px.bar(summary_df, x='Categoria', y='Contagem', title=f'Distribuição por {filter_name}',
                             labels={'Contagem': 'Nº Artigos', 'Categoria': filter_name},
                             color='Contagem', color_continuous_scale=px.colors.sequential.Viridis)
                fig.update_layout(xaxis_title_text='', yaxis_title_text='Número de Artigos')
            
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
            return fig
        return None

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
    PALAVRAS_EXCLUIR = set(stopwords.words('portuguese')).union({
        'eu', 'tu', 'ele', 'ela', 'nós', 'vós', 'eles', 'elas', 'me', 'te', 'se', 'nos', 'vos', 'lhe', 'lhes',
        'mim', 'ti', 'si', 'comigo', 'contigo', 'consigo', 'meu', 'minha', 'meus', 'minhas', 'teu', 'tua',
        'teus', 'tuas', 'seu', 'sua', 'seus', 'suas', 'nosso', 'nossa', 'nossos', 'nossas', 'vosso', 'vossa',
        'vossos', 'vossas', 'este', 'esta', 'estes', 'estas', 'esse', 'essa', 'esses', 'essas', 'aquele',
        'aquela', 'aqueles', 'aquelas', 'isto', 'isso', 'aquilo', 'alguém', 'ninguém', 'tudo', 'nada', 'cada',
        'quem', 'qual', 'quais', 'quanto', 'quantos', 'quanta', 'quantas', 'algum', 'alguma', 'alguns',
        'algumas', 'nenhum', 'nenhuma', 'uns', 'umas', 'a', 'ante', 'após', 'até', 'com', 'contra', 'de',
        'desde', 'em', 'entre', 'para', 'per', 'perante', 'por', 'sem', 'sob', 'sobre', 'trás', 'aqui', 'ali',
        'lá', 'cá', 'acolá', 'aí', 'perto', 'longe', 'dentro', 'fora', 'acima', 'abaixo', 'atrás', 'adiante',
        'antes', 'depois', 'cedo', 'tarde', 'logo', 'já', 'ainda', 'sempre', 'nunca', 'jamais', 'eventualmente',
        'provavelmente', 'possivelmente', 'certamente', 'sim', 'não', 'também', 'só', 'apenas', 'quase',
        'muito', 'pouco', 'bastante', 'demais', 'tanto', 'quanto', 'mais', 'menos', 'mal', 'bem', 'melhor',
        'pior', 'devagar', 'depressa', 'calmamente', 'rapidamente', 'do', 'da', 'dos', 'das', 'no', 'na',
        'nos', 'nas', 'pelo', 'pela', 'pelos', 'pelas', 'deste', 'desta', 'desses', 'dessas', 'neste',
        'nesta', 'nestes', 'nestas', 'daquele', 'daquela', 'daqueles', 'daquelas', 'à', 'às', 'ao', 'aos'
    })

    def filtrar_palavras(texto):
        tokenizer = RegexpTokenizer(r'\b\w+\b')
        palavras = tokenizer.tokenize(texto.lower())
        return [
            palavra for palavra in palavras
            if palavra.isalpha() and len(palavra) >= 4 and palavra not in PALAVRAS_EXCLUIR
        ]

    def gerar_wordcloud(palavras_filtradas):
        if not palavras_filtradas:
            return None
        texto_filtrado = ' '.join(palavras_filtradas)
        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS,
                              max_words=200, colormap='viridis', collocations=False).generate(texto_filtrado)
        fig, ax = plt.subplots(figsize=(10, 5), facecolor=None)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        plt.tight_layout(pad=0)
        return fig

    all_abstracts_text = " ".join(filtered_df['abstract'].dropna().tolist())
    if all_abstracts_text.strip():
        palavras_filtradas = filtrar_palavras(all_abstracts_text)
        if palavras_filtradas:
            wordcloud_fig = gerar_wordcloud(palavras_filtradas)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
                plt.close(wordcloud_fig)
            else:
                st.info("Nenhuma palavra significativa encontrada para gerar a nuvem de palavras.")
        else:
            st.info("Nenhuma palavra significativa encontrada para gerar a nuvem de palavras após a filtragem.")
    else:
        st.info("Nenhum resumo disponível nos artigos filtrados para gerar a nuvem de palavras.")

    # --- Resultados Detalhados dos Artigos ---
    st.subheader(f"🔎 Artigos Encontrados ({len(filtered_df)})")
    for _, row in filtered_df.iterrows():
        st.markdown("---")
        st.markdown(f"**Título:** {row['title']}")
        st.markdown(f"**Autores:** {row['authors']}")
        st.markdown(f"**Ano:** {int(row['year']) if pd.notna(row['year']) else 'Não informado'}")
        st.markdown(f"**Pérola:** {row['Is_Perola']}")
        st.markdown(f"**Revisores:** {', '.join(row['Reviewers']) if row['Reviewers'] else 'Nenhum'}")
        st.markdown(f"**Labels:** {', '.join(row['Parsed_Labels']) if row['Parsed_Labels'] else 'Nenhum'}")

        if row['abstract'].strip():
            with st.expander("Ver Resumo"):
                st.write(row['abstract'])
        
        if row['url'] and row['url'].strip():
            st.markdown(f"[🔗 Link para o Artigo]({row['url']})")
        
        if row['PDF files'] and row['PDF files'].strip():
            st.info(f"📄 PDF disponível: {row['PDF files']}")
else:
    st.info("Nenhum artigo encontrado com os filtros selecionados. Ajuste os filtros na barra lateral.")

st.markdown("---")
# --- Próximos Passos e Sugestões ---
st.header("💡 Próximos Passos e Sugestões para Aprimoramento")
st.markdown("""
* **Gráficos de Treemap/Sunburst:** Explore o uso de `plotly.express.treemap` ou `plotly.express.sunburst` para visualizar dados hierárquicos.
* **Filtros Interativos nos Gráficos:** Implemente a funcionalidade de clicar em uma barra ou fatia do gráfico para filtrar os artigos.
* **Tabela Interativa para Artigos:** Utilize `st.dataframe` para apresentar os resultados em uma tabela interativa com busca e classificação.
* **Funcionalidade de Busca por Texto:** Inclua uma caixa de texto para pesquisar por palavras-chave no título, resumo ou autores.
""")
st.caption("🔧 Desenvolvido com Streamlit e Plotly")
c
