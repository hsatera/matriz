import streamlit as st
import pandas as pd
import re
import plotly.express as px

# --- Configuração da Página Streamlit ---
st.set_page_config(layout="wide", page_title="Filtro de Artigos Científicos com Resumo")

st.title("📚 MATRIZ - Revisão de Escopo")
st.markdown("Use os filtros na barra lateral para encontrar artigos específicos e veja o resumo da distribuição nos gráficos.")

# --- Carregamento e Pré-processamento dos Dados ---
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['notes'] = df['notes'].fillna('')
    df['abstract'] = df['abstract'].fillna('')
    if 'authors' not in df.columns:
        df['authors'] = 'Não informado'
    if 'url' not in df.columns:
        df['url'] = ''
    return df

try:
    df = load_data("selecionados.csv")
except FileNotFoundError:
    st.error("Erro: O arquivo 'selecionados.csv' não foi encontrado.")
    st.stop()

# --- Função para extrair labels ---
def extract_labels(label_string):
    if pd.isna(label_string):
        return []
    match = re.search(r'RAYYAN-LABELS:\s*(.+)', label_string)
    if match:
        labels_part = match.group(1).strip()
        labels_part = labels_part.replace('""', '').replace('{', '').replace('}', '')
        labels = re.findall(r'[^,\|]+', labels_part)
        return [label.strip() for label in labels if label.strip()]
    return []

df['Parsed_Labels'] = df['notes'].apply(extract_labels)

# --- Filtros Disponíveis ---
PAISES = ['Angola', 'Argentina', 'Brasil', 'Cuba', 'Portugal', 'Uruguai']
METODOLOGIAS = ['Estudo misto (quali-quanti)', 'Estudo Qualitativo', 'Estudo quantitativo', 'Revisão literatura']
TIPOS_ESTUDO = ['Definição Ações Coletivas Cuidado', 'Reflexão teórica Ações Coletivas de Cuidado', 'Relato Experiência Ações Coletivas de Cuidado']
EIXOS_MATRIZ = ['Doenças crônicas', 'Saúde Mental', 'Saúde Bucal', 'Infância e Adolescência', 'Gênero e Sexualidade', 'Deficiência Intelectual']
IDIOMAS = ['Língua Espanhola', 'Língua Inglesa', 'Língua Portuguesa']

# --- Barra Lateral de Filtros ---
st.sidebar.header("⚙️ Filtros")

selected_paises = st.sidebar.multiselect("País", PAISES)
selected_metodologias = st.sidebar.multiselect("Metodologia", METODOLOGIAS)
selected_tipos_estudo = st.sidebar.multiselect("Tipo de Estudo", TIPOS_ESTUDO)
selected_eixos_matriz = st.sidebar.multiselect("Eixo Matriz", EIXOS_MATRIZ)
selected_idiomas = st.sidebar.multiselect("Idioma", IDIOMAS)

# --- Lógica de Filtragem ---
filtered_df = df.copy()

def check_labels(article_labels, selected_options):
    if not selected_options:
        return True
    article_labels = article_labels if isinstance(article_labels, list) else []
    article_labels_lower = [label.lower().strip() for label in article_labels]
    selected_options_lower = [option.lower().strip() for option in selected_options]
    return any(option in article_labels_lower for option in selected_options_lower)

# Aplicar filtros
for filtro, selecao in zip(
    ['País', 'Metodologia', 'Tipo de Estudo', 'Eixo Matriz', 'Idioma'],
    [selected_paises, selected_metodologias, selected_tipos_estudo, selected_eixos_matriz, selected_idiomas]
):
    if selecao:
        filtered_df = filtered_df[filtered_df['Parsed_Labels'].apply(lambda x: check_labels(x, selecao))]

# --- Seção: Gráficos de Resumo no Topo ---
if not filtered_df.empty:
    st.header("📊 Visão Geral dos Resultados Filtrados")

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
                    if label.lower().strip() in [opt.lower().strip() for opt in options_list]:
                        original_label = next((opt for opt in options_list if opt.lower().strip() == label.lower().strip()), label)
                        counts[original_label] = counts.get(original_label, 0) + 1

        if counts:
            summary_df = pd.DataFrame(counts.items(), columns=['Categoria', 'Contagem'])
            summary_df = summary_df.sort_values(by='Contagem', ascending=False)
            fig = px.bar(summary_df, x='Categoria', y='Contagem',
                         title=f'{filter_name}',
                         labels={'Contagem': 'Nº Artigos', 'Categoria': filter_name},
                         color='Contagem', color_continuous_scale=px.colors.sequential.Viridis)
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

    # --- Resultados dos Artigos ---
    st.subheader(f"🔎 {len(filtered_df)} Artigos Encontrados")

    for _, row in filtered_df.iterrows():
        st.markdown("---")
        st.markdown(f"**Título:** {row['title']}")
        st.markdown(f"**Autores:** {row['authors']}")
        st.markdown(f"**Ano:** {int(row['year']) if pd.notna(row['year']) else 'Não informado'}")
        st.markdown(f"**Labels:** {', '.join(row['Parsed_Labels']) if row['Parsed_Labels'] else 'Nenhum'}")

        if row['abstract'].strip():
            with st.expander("Ver Resumo"):
                st.write(row['abstract'])

        if row['url'].strip():
            st.markdown(f"[🔗 Link para o Artigo]({row['url']})")

        if pd.notna(row['PDF files']) and row['PDF files'].strip():
            st.info(f"📄 PDF disponível: {row['PDF files']}")
else:
    st.info("Nenhum artigo encontrado com os filtros selecionados.")

st.markdown("---")
st.caption("🔧 Desenvolvido com Streamlit e Plotly")
