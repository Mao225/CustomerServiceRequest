

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openai  # pinned to e.g. openai==0.28
import json
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D


# Load API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]


# ==================== 1) Data Loading & Parsing ====================


@st.cache_data
def load_data():
    file_url = "https://raw.githubusercontent.com/Mao225/CustomerServiceRequest/main/all_clustering_results.xlsx"

    try:
        data = pd.read_excel(file_url, engine='openpyxl')  
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None  

def parse_embedding(embedding_str):
    if isinstance(embedding_str, (np.ndarray, list)):
        return np.array(embedding_str)
    if isinstance(embedding_str, (float, int)):
        return np.array([embedding_str])
    try:
        cleaned_str = embedding_str.strip('[]')
        vals = [float(x) for x in cleaned_str.split()]
        return np.array(vals)
    except:
        return None


# ==================== 2) K-Means (Ward Silhouette) ====================

def compute_silhouette_scores(embedding_matrix, k_min=2, k_max=9):
    scores = []
    if embedding_matrix is not None and embedding_matrix.shape[0] > 1:
        link_mat = linkage(embedding_matrix, method='ward')
        for k in range(k_min, k_max+1):
            clusters = fcluster(link_mat, t=k, criterion='maxclust')
            sc = silhouette_score(embedding_matrix, clusters)
            scores.append({'K': k, 'Silhouette Score': sc})
    return pd.DataFrame(scores)

def get_best_k(df_sil):
    if df_sil.empty:
        return None
    best_idx = df_sil['Silhouette Score'].idxmax()
    return int(df_sil.loc[best_idx, 'K'])

def run_kmeans(embedding_matrix, best_k):
    if best_k and embedding_matrix is not None and embedding_matrix.shape[0] > 1:
        km = KMeans(n_clusters=best_k, random_state=42)
        labels = km.fit_predict(embedding_matrix)
        return labels
    return None


# ==================== 3) DBSCAN ====================

def dbscan_param_search(embeddings, eps_range, min_samples_range):
    results = []
    if embeddings is not None and embeddings.shape[0] > 1:
        for eps_val in eps_range:
            for ms in min_samples_range:
                db = DBSCAN(eps=eps_val, min_samples=ms, metric='cosine')
                labels = db.fit_predict(embeddings)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                cluster_counts = pd.Series(labels).value_counts()
                num_noise = cluster_counts.get(-1, 0)

                # Valid silhouette only if >=2 clusters (not counting noise).
                if n_clusters < 2:
                    sc = -1.0
                else:
                    mask = (labels != -1)
                    if not any(mask):
                        sc = -1.0
                    else:
                        try:
                            sc = silhouette_score(embeddings[mask], labels[mask])
                        except:
                            sc = -1.0

                results.append({
                    'eps': eps_val,
                    'min_samples': ms,
                    'num_clusters': n_clusters,
                    'num_noise': num_noise,
                    'silhouette_score': sc
                })
    df = pd.DataFrame(results)
    df = df.sort_values('silhouette_score', ascending=False)
    return df

def run_dbscan(embedding_matrix, eps_val=0.3, min_samp=5):
    if embedding_matrix is None or embedding_matrix.shape[0] < 2:
        return None
    db = DBSCAN(eps=eps_val, min_samples=min_samp, metric='cosine')
    return db.fit_predict(embedding_matrix)


# ==================== 4) GMM ====================

def compute_gmm_scores(embedding_matrix, k_min=2, k_max=9):
    scores = []
    if embedding_matrix is not None and embedding_matrix.shape[0] > 1:
        for k in range(k_min, k_max+1):
            gmm = GaussianMixture(n_components=k, random_state=42)
            labels = gmm.fit_predict(embedding_matrix)
            if len(np.unique(labels)) > 1:
                sc = silhouette_score(embedding_matrix, labels)
                scores.append({
                    'K': k,
                    'Silhouette Score': sc,
                    'AIC': gmm.aic(embedding_matrix),
                    'BIC': gmm.bic(embedding_matrix)
                })
    return pd.DataFrame(scores)

def run_gmm(embedding_matrix, n_components):
    if n_components and embedding_matrix is not None and embedding_matrix.shape[0] > 1:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        labels = gmm.fit_predict(embedding_matrix)
        probabilities = gmm.predict_proba(embedding_matrix)
        return labels, probabilities, gmm
    return None, None, None


# ==================== 5) BGMM ====================

def compute_bgmm_scores(embedding_matrix, k_min=2, k_max=9):
    scores = []
    optimal_models = {}
    if embedding_matrix is not None and embedding_matrix.shape[0] > 1:
        for k in range(k_min, k_max+1):
            bgmm = BayesianGaussianMixture(
                n_components=k,
                covariance_type='full',
                random_state=42,
                weight_concentration_prior_type='dirichlet_process'
            )
            bgmm.fit(embedding_matrix)
            labels = bgmm.predict(embedding_matrix)
            optimal_models[k] = {"model": bgmm, "labels": labels}
            if len(np.unique(labels)) > 1:
                sc = silhouette_score(embedding_matrix, labels)
                scores.append({
                    'K': k,
                    'Silhouette Score': sc,
                    'Lower Bound': bgmm.lower_bound_,
                    'n_effective_components': (bgmm.weights_ > 0.01).sum()
                })
    return pd.DataFrame(scores), optimal_models

def run_bgmm(embedding_matrix, n_components):
    if n_components and embedding_matrix is not None and embedding_matrix.shape[0] > 1:
        bgmm = BayesianGaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=42,
            weight_concentration_prior_type='dirichlet_process'
        )
        bgmm.fit(embedding_matrix)
        labels = bgmm.predict(embedding_matrix)
        probabilities = bgmm.predict_proba(embedding_matrix)
        return labels, probabilities, bgmm
    return None, None, None


# ==================== 6) Plotting (PCA, cluster sizes) ====================

def plot_clusters_3d(filtered_data, embedding_column, cluster_column, title):
    if filtered_data.empty:
        st.error("No data available for plotting.")
        return
    X_list = []
    L_list = []
    for emb_str, lab in zip(filtered_data[embedding_column], filtered_data[cluster_column]):
        arr = parse_embedding(emb_str)
        if arr is not None:
            X_list.append(arr)
            L_list.append(int(lab))
    if not X_list:
        st.error("No valid embeddings for PCA.")
        return
    X = np.vstack(X_list)
    labs = np.array(L_list)
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(X)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labs)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    non_noise_labels = unique_labels[unique_labels != -1]
    non_noise_colors = colors[unique_labels != -1]
    for label, color in zip(non_noise_labels, non_noise_colors):
        mask = (labs == label)
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            reduced[mask, 2],
            c=[color],
            label=f"Cluster {label}",
            s=50,
            alpha=0.8
        )
    if -1 in unique_labels:
        noise_mask = (labs == -1)
        ax.scatter(
            reduced[noise_mask, 0],
            reduced[noise_mask, 1],
            reduced[noise_mask, 2],
            c='lightgray',
            label="Noise",
            s=50,
            alpha=0.5
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)

def plot_cluster_sizes(data, cluster_column, title):
    if data.empty:
        st.warning("No data available for plotting.")
        return
    c_min = int(data[cluster_column].min())
    c_max = int(data[cluster_column].max())
    rng = range(c_min, c_max+1)

    cluster_counts = data[cluster_column].value_counts()
    cluster_sizes = pd.Series(index=rng, data=0)
    cluster_sizes.update(cluster_counts)
    cluster_sizes = cluster_sizes.sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    non_noise_sizes = cluster_sizes[cluster_sizes.index != -1]
    bars = ax.bar(range(len(non_noise_sizes)), non_noise_sizes.values,
                  color='skyblue', edgecolor='black')
    if -1 in cluster_sizes.index:
        noise_size = cluster_sizes.loc[-1]
        noise_bar = ax.bar([-0.5], [noise_size], color='lightgray', edgecolor='black', alpha=0.5)

    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Count")

    xticks = list(range(len(non_noise_sizes)))
    xticklabels = list(non_noise_sizes.index)
    if -1 in cluster_sizes.index:
        xticks = [-0.5] + xticks
        xticklabels = ['Noise'] + xticklabels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h, str(int(h)),
                ha='center', va='bottom')
    if -1 in cluster_sizes.index:
        h = noise_bar[0].get_height()
        ax.text(noise_bar[0].get_x() + noise_bar[0].get_width()/2., h,
                str(int(h)), ha='center', va='bottom')

    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)


# ==================== 7) Similarity & Representative ====================

def analyze_cluster_similarity(X, labs):
    clusters = {}
    for c in np.unique(labs):
        clusters[c] = X[labs==c]

    intra_dict = {}
    for c, arr in clusters.items():
        if len(arr) > 1:
            sim_mat = cosine_similarity(arr)
            tri = np.triu_indices_from(sim_mat,1)
            val = sim_mat[tri]
            intra_dict[c] = np.mean(val)
        else:
            intra_dict[c] = 1.0

    c_list = list(clusters.keys())
    n = len(c_list)
    mat = np.zeros((n,n))
    for i,c_i in enumerate(c_list):
        e_i = clusters[c_i]
        for j,c_j in enumerate(c_list):
            if i==j:
                mat[i,j] = np.nan
            else:
                e_j = clusters[c_j]
                sims = cosine_similarity(e_i, e_j)
                mat[i,j] = np.mean(sims)
    return intra_dict, c_list, mat

def extract_representative_requests(data, embeddings, cluster_labels, n_representatives=5, method='silhouette'):
    from sklearn.metrics import silhouette_samples
    representatives = {}
    if data.empty:
        return representatives
    data_copy = data.copy()
    if method == 'silhouette':
        silhouette_vals = silhouette_samples(embeddings, cluster_labels)
        data_copy['Silhouette'] = silhouette_vals

    unique_clusters = np.unique(cluster_labels)
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue
        cluster_points = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_points) == 0:
            continue
        cluster_data = data_copy.iloc[cluster_points]
        if cluster_data.empty:
            continue
        if method == 'silhouette':
            subset = cluster_data.nlargest(n_representatives, 'Silhouette')
            c_indices = subset.index
        else:
            c_indices = cluster_data.index[:n_representatives]
        if 'Message' in data_copy.columns:
            reps = data_copy.loc[c_indices, 'Message'].tolist()
            representatives[int(cluster_id)] = reps
    return representatives

def show_all_clusters_insights(data, embedding_column, cluster_column, param_display):
    emb_valid = data.dropna(subset=[embedding_column])
    if emb_valid.empty:
        st.warning("No valid embeddings to extract rep requests.")
        return
    embeddings = np.vstack(emb_valid[embedding_column])
    labels = data.loc[emb_valid.index, cluster_column].values
    reps = extract_representative_requests(
        emb_valid, embeddings, labels, n_representatives=5
    )
    for cid in sorted(reps.keys()):
        with st.expander(f"Cluster {cid}"):
            for i, msg in enumerate(reps[cid], 1):
                st.markdown(f"**{i}.** {msg}")


# ==================== 8) Time Estimation Editor ====================

def create_combined_metrics_editor(data, cluster_column, silhouette_score, weight_silhouette=0.7):
    if data.empty or cluster_column not in data.columns:
        return None, 0, 0

    cluster_sizes = data[cluster_column].value_counts().sort_index()
    cluster_labels = [f"Cluster {int(idx)}" for idx in cluster_sizes.index]

    df = pd.DataFrame({
        'Cluster': cluster_labels,
        'Request Count': cluster_sizes.values,
        'Face Validation (1-5)': [0] * len(cluster_sizes),
        'Processing Time per Request (minutes)': [0] * len(cluster_sizes)
    })

    st.markdown("**Scale Explanations:**")
    st.markdown("- Face Validation (1-5): 1 = Poor cluster quality, 5 = Excellent cluster quality")
    st.markdown("- Processing Time per Request (minutes): estimated manual time for 1 request")

    edited_df = st.data_editor(
        df,
        hide_index=True,
        column_config={
            "Cluster": st.column_config.TextColumn("Cluster", disabled=True),
            "Request Count": st.column_config.NumberColumn("Request Count", disabled=True),
            "Face Validation (1-5)": st.column_config.NumberColumn("Face Validation (1-5)", min_value=0, max_value=5),
            "Processing Time per Request (minutes)": st.column_config.NumberColumn("Processing Time (min)", min_value=0, max_value=120)
        }
    )

    weight_face = 1 - weight_silhouette
    # normalize face validation to [0..1]
    face_scores = (edited_df['Face Validation (1-5)'] - 1) / 4.0
    edited_df['Weight Factor'] = (
        weight_silhouette * silhouette_score
        + weight_face * face_scores
    )
    edited_df['Time Savings (minutes)'] = (
        edited_df['Request Count']
        * edited_df['Processing Time per Request (minutes)']
        * edited_df['Weight Factor']
    )

    total_time_saving = edited_df['Time Savings (minutes)'].sum()
    st.session_state['total_time_saving'] = total_time_saving

    st.markdown("**Weighted Factor Explanation:** ")
    st.markdown("- Combines silhouette score (measures cluster quality) and face validation (human evaluation of cluster quality)")
    st.markdown("- It is computed as: `(weight silhouette * silhouette score) + (weight face validation * face validation score)`")
    st.markdown("- `weight silhouette` is set to 0.7 and `weight face validation` is 0.3")


    st.dataframe(edited_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(edited_df['Cluster'], edited_df['Time Savings (minutes)'], width=0.8)
    max_height = max(edited_df['Time Savings (minutes)'].max(), 1)
    for bar in bars:
        height = bar.get_height()
        idx = bars.index(bar)
        label_info = [
            f"Count: {edited_df['Request Count'].iloc[idx]}",
            f"Weight: {edited_df['Weight Factor'].iloc[idx]:.3f}",
            f"Saved: {height:.1f} min"
        ]
        y_pos = height + max_height * 0.05
        for i, line in enumerate(label_info):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                y_pos - i * (max_height * 0.02),
                line,
                ha='center',
                va='bottom',
                fontsize=7
            )

    ax.set_ylim(top=max_height*1.2)
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Time Savings (minutes)')
    ax.set_title('Estimated Time Savings per Cluster')
    plt.tight_layout()
    st.pyplot(fig)

    return edited_df, silhouette_score, total_time_saving


# ==================== 9) Model Analysis sections ====================

# ==================== 9.1) K-Means ====================
def show_model_analysis_kmeans(model_name,
                               data,
                               embedding_column,
                               silhouette_df,
                               best_k,
                               cluster_column):
    left_col, right_col = st.columns([0.6, 0.4])

    with left_col:
        valid_k_values = sorted(silhouette_df["K"].unique())
        default_selection = "Show All"
        k_options = [default_selection] + [str(x) for x in valid_k_values]

        chosen_k = st.sidebar.selectbox(
            "Number of Clusters",
            k_options,
            key=f"{model_name}_kmeans_cluster_selector"
        )

        if chosen_k != default_selection:
            current_k = int(chosen_k)
            if embedding_column in data.columns:
                embeddings = np.vstack(data[embedding_column].dropna())
                if embeddings.shape[0] > 0:
                    new_labels = run_kmeans(embeddings, current_k)
                    if new_labels is not None:
                        data[cluster_column] = np.nan
                        data.loc[data[embedding_column].dropna().index, cluster_column] = new_labels
            filtered_silhouette_df = silhouette_df[silhouette_df["K"] == current_k].copy()
            filtered_data = data.dropna(subset=[cluster_column]).copy()
        else:
            current_k = best_k
            filtered_silhouette_df = silhouette_df.copy()
            filtered_data = data.copy()

        param_display = f"(K={current_k})"

        st.subheader("K-Means Clustering Scores")
        if not filtered_silhouette_df.empty:
            best_idx = filtered_silhouette_df["Silhouette Score"].idxmax()
            filtered_silhouette_df["IsBest?"] = ""
            filtered_silhouette_df.loc[best_idx, "IsBest?"] = "<-- BEST"
            st.dataframe(filtered_silhouette_df, use_container_width=True)

        col_pca, col_dist = st.columns(2)
        with col_pca:
            st.write(f"#### PCA Visualization {param_display}")
            if not filtered_data.empty:
                plot_clusters_3d(filtered_data, embedding_column, cluster_column, None)

        with col_dist:
            st.write(f"#### Cluster Size Distribution {param_display}")
            if not filtered_data.empty:
                plot_cluster_sizes(filtered_data, cluster_column, None)

        st.write(f"#### Intra/InterCluster Similarity {param_display}")
        if not filtered_data.empty:
            valid_df = filtered_data.dropna(subset=[embedding_column, cluster_column])
            E_list, L_list = [], []
            for _, row in valid_df.iterrows():
                arr = parse_embedding(row[embedding_column])
                lab = row[cluster_column]
                if arr is not None:
                    E_list.append(arr)
                    L_list.append(int(lab))

            if E_list:
                X = np.vstack(E_list)
                Y = np.array(L_list)
                intraD, c_list, mat = analyze_cluster_similarity(X, Y)

                col_intra, col_inter = st.columns([0.3, 0.7])
                with col_intra:
                    st.write("**Intra Cluster Similarity**")
                    row_list = [{"Cluster": f"Cluster {cid}", "Similarity": val}
                                for cid, val in intraD.items()]
                    df_intra = pd.DataFrame(row_list)
                    st.table(df_intra)

                with col_inter:
                    st.write("**Inter Cluster Similarity Matrix**")
                    index_labels = [f"Cluster {c}" for c in c_list]
                    df_int = pd.DataFrame(mat, index=index_labels, columns=index_labels)
                    st.dataframe(df_int.style.background_gradient(cmap="Blues", axis=None))

        st.write(f"#### Time Savings Estimation {param_display}")
        if not filtered_data.empty and not filtered_silhouette_df.empty:
            row_match = filtered_silhouette_df[filtered_silhouette_df['K'] == current_k]
            if not row_match.empty:
                current_silhouette = row_match['Silhouette Score'].iloc[0]
            else:
                current_silhouette = 0.0

            create_combined_metrics_editor(filtered_data, cluster_column, current_silhouette)

    with right_col:
        st.subheader(f"Representative Requests {param_display}")
        if not filtered_data.empty:
            show_all_clusters_insights(filtered_data, embedding_column, cluster_column, param_display)
        else:
            st.warning("No data available.")

# ==================== 9.2) DBSCAN ====================
def show_model_analysis_dbscan(
    model_name,
    data,
    embedding_column,
    dbscan_df,
    best_eps,
    best_ms,
    cluster_labels,
    cluster_column
):
    left_col, right_col = st.columns([0.6, 0.4])

    with left_col:

        unique_eps = sorted(dbscan_df['eps'].unique())
        unique_ms = sorted(dbscan_df['min_samples'].unique())

        best_idx_overall = dbscan_df["silhouette_score"].idxmax() if not dbscan_df.empty else None
        best_eps_overall = dbscan_df.loc[best_idx_overall, "eps"] if best_idx_overall is not None else 0.3
        best_ms_overall = dbscan_df.loc[best_idx_overall, "min_samples"] if best_idx_overall is not None else 5

        formatted_eps = [f"{eps:.1f}" for eps in unique_eps]
        formatted_ms = [str(ms) for ms in unique_ms]

        selected_eps = st.sidebar.selectbox(
            "Filter by EPS values",
            ["Show All"] + formatted_eps,
            index=0,
            key=f"{model_name}_eps_filter"
        )
        selected_ms = st.sidebar.selectbox(
            "Filter by min_samples",
            ["Show All"] + formatted_ms,
            index=0,
            key=f"{model_name}_ms_filter"
        )

        st.subheader("DBSCAN Parameter Results")
        filtered_df = dbscan_df.copy()
        if selected_eps != "Show All":
            filtered_df = filtered_df[filtered_df['eps'] == float(selected_eps)]
        if selected_ms != "Show All":
            filtered_df = filtered_df[filtered_df['min_samples'] == int(selected_ms)]

        if not filtered_df.empty:
            best_idx = filtered_df["silhouette_score"].idxmax()
            filtered_df["isBest"] = ""
            filtered_df.loc[best_idx, "isBest"] = "<-- BEST"
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.warning("No DBSCAN results for these filters.")
            fallback_df = dbscan_df.copy()
            if not fallback_df.empty:
                best_idx = fallback_df["silhouette_score"].idxmax()
                fallback_df["isBest"] = ""
                fallback_df.loc[best_idx, "isBest"] = "<-- BEST"
                st.dataframe(fallback_df, use_container_width=True)

        # Current parameters
        if selected_eps == "Show All" and selected_ms == "Show All":
            current_eps = best_eps_overall
            current_ms = best_ms_overall
            param_display = f"(Best eps={current_eps:.1f}, min_samples={current_ms})"
            if best_idx_overall is not None:
                current_silhouette = dbscan_df.loc[best_idx_overall, "silhouette_score"]
            else:
                current_silhouette = 0.0
        else:
            current_eps = float(selected_eps) if selected_eps != "Show All" else best_eps_overall
            current_ms = int(selected_ms) if selected_ms != "Show All" else best_ms_overall
            param_display = f"(eps={current_eps:.1f}, min_samples={current_ms})"
            match_rows = filtered_df[
                (filtered_df['eps'] == current_eps) &
                (filtered_df['min_samples'] == current_ms)
            ]
            if not match_rows.empty:
                current_silhouette = match_rows.iloc[0]['silhouette_score']
            else:
                current_silhouette = 0.0

        valid_embeddings = data[embedding_column].dropna()
        if len(valid_embeddings) > 0:
            current_labels = run_dbscan(np.vstack(valid_embeddings), eps_val=current_eps, min_samp=current_ms)
            if current_labels is not None:
                data[cluster_column] = np.nan
                data.loc[valid_embeddings.index, cluster_column] = current_labels
                data[cluster_column] = data[cluster_column].astype('Int64')
                filtered_data = data.dropna(subset=[cluster_column]).copy()

                col_pca, col_dist = st.columns(2)
                with col_pca:
                    st.subheader(f"PCA Visualization {param_display}")
                    plot_clusters_3d(filtered_data, embedding_column, cluster_column, None)
                with col_dist:
                    st.subheader(f"Cluster Size Distribution {param_display}")
                    plot_cluster_sizes(filtered_data, cluster_column, None)

                st.subheader(f"Intra/Inter Cluster Similarity {param_display}")
                valid_df = filtered_data.dropna(subset=[embedding_column, cluster_column])
                if not valid_df.empty:
                    E_list, L_list = [], []
                    for _, row in valid_df.iterrows():
                        arr = parse_embedding(row[embedding_column])
                        lab = row[cluster_column]
                        if arr is not None:
                            E_list.append(arr)
                            L_list.append(int(lab))
                    if E_list:
                        X = np.vstack(E_list)
                        Y = np.array(L_list, dtype=int)
                        intraD, c_list, mat = analyze_cluster_similarity(X, Y)

                        col_intra, col_inter = st.columns(2)
                        with col_intra:
                            st.write("**Intra Cluster Similarity**")
                            row_list = [{"Cluster": f"Cluster {cid}", "Similarity": val}
                                        for cid, val in intraD.items()]
                            df_intra = pd.DataFrame(row_list)
                            st.table(df_intra)
                        with col_inter:
                            st.write("**Inter Cluster Similarity**")
                            index_labels = [f"Cluster {c}" for c in c_list]
                            df_int = pd.DataFrame(mat, index=index_labels, columns=index_labels)
                            st.dataframe(df_int.style.background_gradient(cmap="Blues", axis=None))
                else:
                    st.warning("No valid data for similarity analysis.")

                st.subheader(f"Time Savings Estimation {param_display}")
                if not filtered_data.empty:
                    # Exclude noise if desired
                    filtered_data_no_noise = filtered_data[filtered_data[cluster_column] != -1]
                    if not filtered_data_no_noise.empty:
                        create_combined_metrics_editor(
                            filtered_data_no_noise, cluster_column, current_silhouette
                        )
                    else:
                        st.warning("All points are noise; no clusters to analyze.")
            else:
                st.warning("DBSCAN returned None with these params.")
        else:
            st.warning("No valid embeddings for DBSCAN.")

    with right_col:
        st.subheader(f"Representative Requests {param_display}")
        if 'filtered_data' in locals() and not filtered_data.empty:
            show_all_clusters_insights(filtered_data, embedding_column, cluster_column, param_display)
        else:
            st.warning("No data available for representative requests.")

# ==================== 9.3) GMM ====================
def show_model_analysis_gmm(
    model_name,
    data,
    embedding_column,
    gmm_scores_df,
    best_k,
    cluster_column
):
    left_col, right_col = st.columns([0.6, 0.4])
    with left_col:

        valid_k_values = sorted(gmm_scores_df["K"].unique())
        default_selection = "Show All"
        k_options = [default_selection] + [str(x) for x in valid_k_values]

        chosen_k = st.sidebar.selectbox(
            "Number of Components",
            k_options,
            key=f"{model_name}_gmm_cluster_selector"
        )
        if chosen_k != default_selection:
            current_k = int(chosen_k)
            embeddings = np.vstack(data[embedding_column].dropna())
            if embeddings.shape[0] > 0:
                new_labels, probabilities, model = run_gmm(embeddings, current_k)
                if new_labels is not None:
                    data[cluster_column] = np.nan
                    data.loc[data[embedding_column].dropna().index, cluster_column] = new_labels
            filtered_scores_df = gmm_scores_df[gmm_scores_df["K"] == current_k].copy()
            filtered_data = data.dropna(subset=[cluster_column]).copy()
        else:
            current_k = best_k
            filtered_scores_df = gmm_scores_df.copy()
            filtered_data = data.copy()

        param_display = f"(Components={current_k})"
        st.subheader("GMM Clustering Scores")
        if not filtered_scores_df.empty:
            best_idx = filtered_scores_df["Silhouette Score"].idxmax()
            filtered_scores_df["IsBest?"] = ""
            filtered_scores_df.loc[best_idx, "IsBest?"] = "<-- BEST"
            st.dataframe(filtered_scores_df, use_container_width=True)
        else:
            st.warning("No GMM scores for these components.")

        col_pca, col_dist = st.columns(2)
        with col_pca:
            st.subheader(f"PCA Visualization {param_display}")
            if not filtered_data.empty:
                plot_clusters_3d(filtered_data, embedding_column, cluster_column, None)

        with col_dist:
            st.subheader(f"Cluster Size Distribution {param_display}")
            if not filtered_data.empty:
                plot_cluster_sizes(filtered_data, cluster_column, None)

        st.subheader(f"Intra/Inter Cluster Similarity {param_display}")
        if not filtered_data.empty:
            valid_df = filtered_data.dropna(subset=[embedding_column, cluster_column])
            E_list, L_list = [], []
            for _, row in valid_df.iterrows():
                arr = parse_embedding(row[embedding_column])
                lab = row[cluster_column]
                if arr is not None:
                    E_list.append(arr)
                    L_list.append(int(lab))
            if E_list:
                X = np.vstack(E_list)
                Y = np.array(L_list)
                intraD, c_list, mat = analyze_cluster_similarity(X, Y)
                col_intra, col_inter = st.columns(2)
                with col_intra:
                    st.write("**Intra Cluster Similarity**")
                    row_list = [{"Cluster": f"Cluster {cid}", "Similarity": val} for cid, val in intraD.items()]
                    df_intra = pd.DataFrame(row_list)
                    st.table(df_intra)
                with col_inter:
                    st.write("**Inter Cluster Similarity**")
                    index_labels = [f"Cluster {c}" for c in c_list]
                    df_int = pd.DataFrame(mat, index=index_labels, columns=index_labels)
                    st.dataframe(df_int.style.background_gradient(cmap="Blues", axis=None))

        st.subheader(f"Time Savings Estimation {param_display}")
        if not filtered_data.empty and not filtered_scores_df.empty:
            row_match = filtered_scores_df[filtered_scores_df['K'] == current_k]
            if not row_match.empty:
                current_silhouette = row_match['Silhouette Score'].iloc[0]
            else:
                current_silhouette = 0.0
            create_combined_metrics_editor(filtered_data, cluster_column, current_silhouette)

    with right_col:
        st.subheader(f"Representative Requests {param_display}")
        if not filtered_data.empty:
            show_all_clusters_insights(filtered_data, embedding_column, cluster_column, param_display)
        else:
            st.warning("No data available.")

# ==================== 9.4) BGMM ====================
def show_model_analysis_bgmm(
    model_name,
    data,
    embedding_column,
    bgmm_scores_df,
    best_k,
    cluster_column
):
    left_col, right_col = st.columns([0.6, 0.4])
    with left_col:

        valid_k_values = sorted(bgmm_scores_df["K"].unique())
        default_selection = "Show All"
        k_options = [default_selection] + [str(x) for x in valid_k_values]

        chosen_k = st.sidebar.selectbox(
            "Number of Components",
            k_options,
            key=f"{model_name}_bgmm_cluster_selector"
        )
        if chosen_k != default_selection:
            current_k = int(chosen_k)
            embeddings = np.vstack(data[embedding_column].dropna())
            if embeddings.shape[0] > 0:
                new_labels, probabilities, _ = run_bgmm(embeddings, current_k)
                if new_labels is not None:
                    data[cluster_column] = np.nan
                    data.loc[data[embedding_column].dropna().index, cluster_column] = new_labels
            filtered_scores_df = bgmm_scores_df[bgmm_scores_df["K"] == current_k].copy()
            filtered_data = data.dropna(subset=[cluster_column]).copy()
            param_display = f"(K={current_k})"
        else:
            current_k = best_k
            filtered_scores_df = bgmm_scores_df.copy()
            filtered_data = data.copy()
            param_display = f"(K={current_k})"

        st.subheader("BGMM Clustering Scores")
        if not filtered_scores_df.empty:
            best_idx = filtered_scores_df["Silhouette Score"].idxmax()
            filtered_scores_df["IsBest?"] = ""
            filtered_scores_df.loc[best_idx, "IsBest?"] = "<-- BEST"
            st.dataframe(filtered_scores_df, use_container_width=True)
        else:
            st.warning("No BGMM scores for these components.")

        col_pca, col_dist = st.columns(2)
        with col_pca:
            st.subheader(f"PCA Visualization {param_display}")
            if not filtered_data.empty:
                plot_clusters_3d(filtered_data, embedding_column, cluster_column, None)

        with col_dist:
            st.subheader(f"Cluster Size Distribution {param_display}")
            if not filtered_data.empty:
                plot_cluster_sizes(filtered_data, cluster_column, None)

        st.subheader(f"Intra/Inter Cluster Similarity {param_display}")
        if not filtered_data.empty:
            valid_df = filtered_data.dropna(subset=[embedding_column, cluster_column])
            E_list, L_list = [], []
            for _, row in valid_df.iterrows():
                arr = parse_embedding(row[embedding_column])
                lab = row[cluster_column]
                if arr is not None:
                    E_list.append(arr)
                    L_list.append(int(lab))
            if E_list:
                X = np.vstack(E_list)
                Y = np.array(L_list, dtype=int)
                intraD, c_list, mat = analyze_cluster_similarity(X, Y)
                col_intra, col_inter = st.columns(2)
                with col_intra:
                    st.write("**Intra Cluster Similarity**")
                    row_list = [{"Cluster": f"Cluster {cid}", "Similarity": val} for cid, val in intraD.items()]
                    df_intra = pd.DataFrame(row_list)
                    st.table(df_intra)
                with col_inter:
                    st.write("**Inter Cluster Similarity**")
                    index_labels = [f"Cluster {c}" for c in c_list]
                    df_int = pd.DataFrame(mat, index=index_labels, columns=index_labels)
                    st.dataframe(df_int.style.background_gradient(cmap="Blues", axis=None))

        st.subheader(f"Time Savings Estimation {param_display}")
        if not filtered_data.empty and not filtered_scores_df.empty:
            row_match = filtered_scores_df[filtered_scores_df['K'] == current_k]
            if not row_match.empty:
                current_silhouette = row_match['Silhouette Score'].iloc[0]
            else:
                current_silhouette = 0.0
            create_combined_metrics_editor(filtered_data, cluster_column, current_silhouette)
        else:
            st.warning("No data for time savings analysis.")

    with right_col:
        st.subheader(f"Representative Requests {param_display}")
        if not filtered_data.empty:
            show_all_clusters_insights(filtered_data, embedding_column, cluster_column, param_display)
        else:
            st.warning("No data available.")



# ==================== 10) Dynamic Chat Interface for Analysis Results ====================

def analyze_request_themes(requests: list) -> dict:
    if not requests:
        return {
            'primary_topic': '',
            'request_type': '',
            'complexity': '',
            'key_characteristics': [],
            'automation_potential': ''
        }

    combined_requests = "".join([f"- {req}" for req in requests])

    prompt = (
        "Analyze these customer service requests and identify patterns:"
        f"{combined_requests}"
        "Provide a JSON response with:"
        "- primary_topic: Main subject/area"
        "- request_type: Type of action requested"
        "- complexity: 'Low', 'Medium', or 'High'"
        "- key_characteristics: List of notable patterns"
        "- automation_potential: Assessment of automation possibility"
        "Respond only with the JSON."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing customer service patterns."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"GPT-4 analysis error: {str(e)}")
        return {
            'primary_topic': 'General Request',
            'request_type': 'General Inquiry',
            'complexity': 'Medium',
            'key_characteristics': [],
            'automation_potential': 'Unknown'
        }

def analyze_and_respond(user_input: str, data: pd.DataFrame, cluster_results: dict) -> str:
    context = {
        'method': cluster_results['method'],
        'silhouette': cluster_results['silhouette'],
        'patterns': cluster_results['patterns'],
        'similarities': cluster_results['similarities'],
        'time_savings': cluster_results['time_savings']
    }

    prompt = (
        "As a clustering analysis expert, answer this question about the clustering results:"
        f"Question: {user_input}"
        "Context:"
        f"{json.dumps(context, indent=2)}"
        "Provide a clear, concise answer focusing on the specific aspects asked about."
        "If the question is unclear, suggest related aspects that might be helpful."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in explaining clustering analysis results."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"GPT-4 response error: {str(e)}")
        return (
            "I encountered an error analyzing your question. You can ask about:"
            "- Overall clustering insights"
            "- Specific clusters"
            "- Time savings and efficiency"
            "- Similarities between clusters"
        )

def chat_interface(data: pd.DataFrame, cluster_results: dict):
    st.header("Ask Questions About the Analysis")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.container():
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")

    # Get user input
    user_input = st.text_input(
        "Ask about the clustering results:",
        key="user_input",
        placeholder="e.g., 'What are the main insights?' or 'Tell me about cluster 2'"
    )

    # Process new input
    if user_input and user_input != st.session_state.get("last_user_input", ""):
        st.session_state["last_user_input"] = user_input
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            response = analyze_and_respond(user_input, data, cluster_results)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_msg = (
                "I encountered an error analyzing your question. "
                "Please try rephrasing or ask about a different aspect."
            )
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            print(f"Chat interface error: {str(e)}")

        st.rerun()

# ==================== 11) Process Functions for Each Method ====================

def process_kmeans(data, mat, emb_valid):
    df_sil = compute_silhouette_scores(mat, 2, 9)
    best_k = None
    if not df_sil.empty:
        best_k = get_best_k(df_sil)
        labs = run_kmeans(mat, best_k)
        if labs is not None:
            labs = labs.astype(int)
            data['Kmeans_Model'] = np.nan
            data.loc[emb_valid.index, 'Kmeans_Model'] = labs

    show_model_analysis_kmeans(
        "Model 1",
        data,
        "Embeddings",
        df_sil,
        best_k,
        "Kmeans_Model"
    )

    silhouette_for_summary = 0.0
    if best_k and not df_sil.empty:
        row_match = df_sil[df_sil['K'] == best_k]
        if not row_match.empty:
            silhouette_for_summary = row_match["Silhouette Score"].iloc[0]

    return "Kmeans_Model", f"(K={best_k})" if best_k else "", silhouette_for_summary


def process_dbscan(data, mat, emb_valid, eps_range_dbscan, min_samples_range_dbscan):
    dbscan_df = dbscan_param_search(mat, eps_range_dbscan, min_samples_range_dbscan)
    if not dbscan_df.empty:
        best_idx = dbscan_df["silhouette_score"].idxmax()
        best_eps = dbscan_df.loc[best_idx, "eps"]
        best_ms = dbscan_df.loc[best_idx, "min_samples"]
        silhouette_for_summary = dbscan_df.loc[best_idx, "silhouette_score"]
    else:
        st.warning("No DBSCAN results available. Please adjust the parameter ranges or check your data.")
        return None, "", 0.0

    show_model_analysis_dbscan(
        "Model 1",
        data,
        "Embeddings",
        dbscan_df,
        best_eps,
        best_ms,
        None,  # cluster_labels is handled internally
        "DBSCAN_Model"
    )

    return "DBSCAN_Model", f"(eps={best_eps}, min_samples={best_ms})", silhouette_for_summary


def process_gmm(data, mat, emb_valid):
    gmm_scores_df = compute_gmm_scores(mat, k_min=2, k_max=9)
    best_k = None
    if not gmm_scores_df.empty:
        best_idx = gmm_scores_df["Silhouette Score"].idxmax()
        best_k = int(gmm_scores_df.loc[best_idx, "K"])

        # Run GMM immediately with best_k so "GMM_Model" always exists
        labs, probabilities, model = run_gmm(mat, best_k)
        if labs is not None:
            labs = labs.astype(int)
            data["GMM_Model"] = np.nan
            data.loc[emb_valid.index, "GMM_Model"] = labs

    show_model_analysis_gmm(
        "Model 1",
        data,
        "Embeddings",
        gmm_scores_df,
        best_k,
        "GMM_Model"
    )

    silhouette_for_summary = 0.0
    if best_k and not gmm_scores_df.empty:
        row_match = gmm_scores_df[gmm_scores_df['K'] == best_k]
        if not row_match.empty:
            silhouette_for_summary = row_match["Silhouette Score"].iloc[0]

    return "GMM_Model", f"(K={best_k})" if best_k else "", silhouette_for_summary


def process_bgmm(data, mat, emb_valid):
    bgmm_scores_df, optimal_models = compute_bgmm_scores(mat, k_min=2, k_max=9)
    best_k = None
    if not bgmm_scores_df.empty:
        best_idx = bgmm_scores_df["Silhouette Score"].idxmax()
        best_k = int(bgmm_scores_df.loc[best_idx, "K"])

        # Immediately run BGMM with best_k so "BGMM_Model" always exists
        labs, probabilities, model = run_bgmm(mat, best_k)
        if labs is not None:
            labs = labs.astype(int)
            data["BGMM_Model"] = np.nan
            data.loc[emb_valid.index, "BGMM_Model"] = labs

    show_model_analysis_bgmm(
        "Model 1",
        data,
        "Embeddings",
        bgmm_scores_df,
        best_k,
        "BGMM_Model"
    )

    silhouette_for_summary = 0.0
    if best_k and not bgmm_scores_df.empty:
        row_match = bgmm_scores_df[bgmm_scores_df['K'] == best_k]
        if not row_match.empty:
            silhouette_for_summary = row_match["Silhouette Score"].iloc[0]

    return "BGMM_Model", f"(K={best_k})" if best_k else "", silhouette_for_summary


# ==================== 12) Process Clustering Results for Chat Interface ====================

def process_clustering_results(valid_df, cluster_col, cluster_patterns, cluster_similarities,
                               time_savings, method_model1, param_display, silhouette_for_summary):
    E_list, L_list = [], []
    for _, row in valid_df.iterrows():
        arr = parse_embedding(row["Embeddings"])
        lab = row[cluster_col]
        if arr is not None:
            E_list.append(arr)
            L_list.append(int(lab))  # cast cluster label to plain int

    if E_list:
        X = np.vstack(E_list)
        Y = np.array(L_list)
        intraD, c_list, mat = analyze_cluster_similarity(X, Y)

        # ----------------------
        # 1) Fix 'intra' keys
        # ----------------------
        cluster_similarities['intra'] = {}
        for k, v in intraD.items():
            k_str = str(int(k))
            cluster_similarities['intra'][k_str] = v

        # ----------------------
        # 2) Fix 'inter' keys
        # ----------------------
        cluster_similarities['inter'] = {}
        for i, c1 in enumerate(c_list):
            for j, c2 in enumerate(c_list):
                if i < j:
                    key_str = f"{int(c1)}-{int(c2)}"
                    cluster_similarities['inter'][key_str] = mat[i, j]

        # Extract patterns from representative requests
        reps = extract_representative_requests(valid_df, X, Y, n_representatives=5)

        # Calculate cluster sizes as percentages
        cluster_sizes = valid_df[cluster_col].value_counts()
        total_requests = cluster_sizes.sum()

        # ----------------------
        # 3) Fix cluster_id keys in patterns
        # ----------------------
        for cluster_id, requests in reps.items():
            if cluster_id != -1:  # Skip noise points
                size_percentage = (cluster_sizes.get(cluster_id, 0) / total_requests) * 100
                themes = analyze_request_themes(requests)
                cluster_id_str = str(int(cluster_id))
                cluster_patterns[cluster_id_str] = {
                    'size_percentage': size_percentage,
                    'themes': themes,
                    'automation_potential': 'high' if intraD.get(cluster_id, 0) > 0.7 else 'medium'
                }

        # ----------------------
        # 4) Fix cluster_id keys in time_savings
        # ----------------------
        if 'total_time_saving' in st.session_state:
            total_saving = st.session_state['total_time_saving']
            for cluster_id in cluster_sizes.index:
                if cluster_id != -1:  # Skip noise
                    size_factor = cluster_sizes[cluster_id] / cluster_sizes.sum()
                    similarity_factor = intraD.get(cluster_id, 0)
                    confidence = 'high' if similarity_factor > 0.7 else 'medium'
                    cluster_id_str = str(int(cluster_id))
                    time_savings[cluster_id_str] = {
                        'minutes': total_saving * size_factor * similarity_factor,
                        'confidence': confidence
                    }

        # Initialize the chat interface
        chat_interface(
            data=valid_df,
            cluster_results={
                'method': method_model1,
                'parameters': param_display,
                'silhouette': silhouette_for_summary,
                'similarities': cluster_similarities,
                'patterns': cluster_patterns,
                'time_savings': time_savings
            }
        )
    else:
        st.warning("No valid embeddings for analysis")


# ==================== 13) Main Function ====================

def main():
    st.set_page_config(
        layout="wide",
        page_title="Customer Service Requests Clustering",
        initial_sidebar_state="expanded"
    )
    st.title("Customer Service Requests Clustering Results")

    data = load_data()
    data['Embeddings'] = data['Embeddings'].apply(parse_embedding)

    st.sidebar.header("Filters")
    method_model1 = st.sidebar.selectbox(
        "Clustering Method",
        ["K-Means", "GMM", "BGMM", "DBSCAN"],
        index=0
    )

    # Define parameter ranges for DBSCAN
    eps_range_dbscan = np.arange(0.2, 0.6, 0.1)
    min_samples_range_dbscan = range(5, 15, 2)

    # Prepare embedding matrix
    emb_valid = data['Embeddings'].dropna()
    mat = np.vstack(emb_valid) if not emb_valid.empty else None

    # Initialize variables for cluster analysis
    cluster_col = None
    param_display = ""
    silhouette_for_summary = 0.0
    cluster_patterns = {}
    time_savings = {}
    cluster_similarities = {'intra': {}, 'inter': {}}

    # Process clustering based on selected method
    if method_model1 == "K-Means":
        cluster_col, param_display, silhouette_for_summary = process_kmeans(data, mat, emb_valid)
    elif method_model1 == "GMM":
        cluster_col, param_display, silhouette_for_summary = process_gmm(data, mat, emb_valid)
    elif method_model1 == "BGMM":
        cluster_col, param_display, silhouette_for_summary = process_bgmm(data, mat, emb_valid)
    else:  # DBSCAN
        cluster_col, param_display, silhouette_for_summary = process_dbscan(
            data, mat, emb_valid, eps_range_dbscan, min_samples_range_dbscan
        )

    # Process clustering results for chat interface
    if cluster_col and cluster_col in data.columns:
        valid_df = data.dropna(subset=[cluster_col, "Embeddings"])
        if not valid_df.empty:
            process_clustering_results(valid_df, cluster_col, cluster_patterns,
                                       cluster_similarities, time_savings, method_model1,
                                       param_display, silhouette_for_summary)
        else:
            st.warning("No valid data for clustering analysis")
    else:
        st.warning("Please select a clustering method")


if __name__ == "__main__":
    main()
