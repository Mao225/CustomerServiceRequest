

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openai  # openai==0.28
import time
import json
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
import re
from typing import Dict, List, Optional, Union
import traceback

# ==================== 1) Data Loading & Parsing ====================


@st.cache_data
def load_data():
    return pd.read_excel('all_clustering_results.xlsx')

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
        if 'Updated_Message' in data_copy.columns:
            reps = data_copy.loc[c_indices, 'Updated_Message'].tolist()
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

    st.markdown("**Time Savings Calculation:**")
    st.markdown("- Time Savings = Request Count × Processing Time per Request × Weight Factor")

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

# ==================== K-Means ====================
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

# ==================== DBSCAN ====================
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

# ==================== GMM ====================
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

# ==================== BGMM ====================
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
# Format helper function
def format_gpt_response(gpt_reply: str) -> str:
    """
    Convert GPT's plain text bullets to Markdown-friendly formatting
    and ensure adequate spacing between paragraphs.
    """
    formatted = gpt_reply.replace("•", "\n-")  # Convert '•' to markdown bullet
    formatted = re.sub(r'[ \t]+\n', '\n', formatted)
    formatted = re.sub(r'\n{1}', '\n\n', formatted)
    return formatted


# Global cache to avoid repeated GPT calls:
gpt_cache = {}

def get_cached_response(question: str, cluster_results: dict) -> Optional[str]:
    """
    Enhanced caching system that properly distinguishes between different types of queries.
    Returns cached responses when available, generates new ones when needed.
    """
    # For main insights questions, ALWAYS use the detailed analysis
    if any(phrase in question.lower() for phrase in [
        "main insight", "main insights", "key insight", "overview", 
        "summary", "analyze", "analysis", "theme", "pattern"
    ]):
        # Return None to force detailed analysis generation
        return None
    
    # Safely extract values with defaults
    method = cluster_results.get('method', 'Unknown')
    silhouette = cluster_results.get('silhouette', 0.0)
    parameters = cluster_results.get('parameters', '')
    
    # We now include the full parameter display in the cache key to differentiate
    # between different configurations of the same method
    base_key_parts = (method, parameters, f"{silhouette:.4f}")
    
    # Check for cluster-specific question
    cluster_match = re.search(r"cluster\s+(\d+)", question.lower())
    if cluster_match:
        cluster_id = cluster_match.group(1)
        cache_key = ("cluster_analysis",) + base_key_parts + (cluster_id,)
        if cache_key in gpt_cache:
            return gpt_cache[cache_key]
        return None

    # Check for similarity analysis questions
    if any(phrase in question.lower() for phrase in [
        "similarity", "cohesion", "coherence", "related", "relationship",
        "connection", "overlap", "distance"
    ]):
        cache_key = ("similarity_analysis",) + base_key_parts
        if cache_key in gpt_cache:
            return gpt_cache[cache_key]
        return None

    # Handle silhouette score questions
    if any(phrase in question.lower() for phrase in ["silhouette", "quality", "validation", "score"]):
        cache_key = ("quality_analysis",) + base_key_parts
        if cache_key in gpt_cache:
            return gpt_cache[cache_key]
        return None

    # Handle time savings questions
    if any(phrase in question.lower() for phrase in ["time", "saving", "efficiency", "ROI", "benefit"]):
        cache_key = ("time_analysis",) + base_key_parts
        if cache_key in gpt_cache:
            return gpt_cache[cache_key]
        return None

    # No cache hit found - force detailed analysis
    return None


def generate_gpt_theme_analysis(
    cluster_patterns, 
    method, 
    silhouette, 
    cluster_similarities, 
    time_savings,
    cluster_id=None,
    parameters=""  # Add parameters here
):
    """
    Enhanced theme analysis that includes business impact and plant-specific logic.
    Now includes the clustering parameters in the cache key.
    """
    # Build unique cache key with parameters included
    cache_key = ("theme_analysis", method, parameters, f"{silhouette:.4f}", str(cluster_id))
    if cache_key in gpt_cache:
        return gpt_cache[cache_key]

    # Prepare data for GPT
    if cluster_id is None:
        question_prompt = "Analyze all clusters for insights and modularization opportunities."
        selected_clusters = cluster_patterns
    else:
        question_prompt = f"Analyze Cluster {cluster_id} specifically for modularization opportunities."
        selected_clusters = {cluster_id: cluster_patterns.get(cluster_id, {})}

    # Gather cluster data including time savings
    cluster_data = {}
    for cid, info in selected_clusters.items():
        if isinstance(info, dict):
            cluster_data[cid] = {
                'requests': info.get('requests', []),
                'size_percentage': info.get('size_percentage', 0),
                'intra_similarity': cluster_similarities.get('intra', {}).get(str(cid), 0),
                'time_savings': time_savings.get(str(cid), {})
            }

    system_prompt = """You are an industrial equipment analytics expert analyzing service request clusters to identify automation, modernization, and standardization opportunities.

Present your analysis in a clear, readable format with appropriate spacing and visual structure:

1. Clustering Quality Assessment
   • Interpret the silhouette score and what it means for automation potential
   • Explain cluster cohesion in simple terms with concrete implications
   • Clearly state whether clusters are well-separated or overlapping
   • Use short sentences and avoid technical jargon where possible

2. Request Domain Patterns (Present as distinct bullet points)
   • Identify ALL major product types and request categories present in the clusters
   • For each identified product/request type, note specific patterns and frequencies
   • Look for repetitive elements across different request types
   • Highlight any emerging trends or evolving customer needs visible in the data

3. Modularization Opportunities (Explore all possibilities)
   • Product Standardization: Identify ANY frequently requested product variants that could be added to standard portfolio (Except GMH product number extension)
   • Component Modularization: Note opportunities for creating standardized components across product lines
   • Portfolio Optimization: Suggest product categories that could be consolidated or expanded
   • Explicitly distinguish between different types of modernization for each cluster

4. Process Automation Opportunities
   • Workflow Automation: Identify repeatable processes across ANY domain (such as GMH product number extension)
   • Documentation Standardization: Note patterns in how technical information is requested/provided
   • Quote/Price Automation: Identify standardizable pricing elements
   • Design Automation: Note opportunities for parametric or template-based design
   • For each opportunity, estimate implementation complexity and potential impact

5. Implementation Recommendations
   • Present prioritized steps with clear rationale
   • Organize technical requirements by domain
   • Use numbered lists for sequential steps
   • Keep paragraphs short (3-5 lines maximum)

IMPORTANT: Use proper spacing between sections and bullet points. Use bold for important conclusions or findings. You MUST include ALL 5 sections in the order specified with the exact titles shown."""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{question_prompt}\n\n"
                        f"Method: {method} {parameters}\n"  # Now includes parameters
                        f"Silhouette Score: {silhouette:.2f}\n"
                        f"Cluster Data:\n{json.dumps(cluster_data, indent=2)}\n"
                        "Note: Pay special attention to all potentially repeating patterns "
                        "that could indicate automation or standardization opportunities."
                    )
                }
            ],
            max_tokens=800,
            temperature=0.2,
            timeout=20
        )
        gpt_reply = response.choices[0].message.content
        
        # Verify response completeness
        required_sections = [
            "1. Clustering Quality Assessment", 
            "2. Request Domain Patterns", 
            "3. Modernization Opportunities", 
            "4. Process Automation Opportunities",
            "5. Implementation Recommendations"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in gpt_reply:
                missing_sections.append(section)
                
        if missing_sections:
            gpt_reply += "\n\nNote: Some sections appear to be missing: " + ", ".join(missing_sections) + ". Please ask for clarification if needed."
            
        gpt_cache[cache_key] = gpt_reply
        return gpt_reply
        
    except Exception as e:
        print(f"GPT API error: {e}")
        return "Error generating complete analysis. Please try again."

def generate_gpt_similarity_analysis(cluster_similarities, method, silhouette, parameters=""):
    """
    Generate analysis of cluster similarity and quality metrics with focus on automation potential.
    Now includes parameters in the cache key.
    """
    cache_key = ("similarity_analysis", method, parameters, f"{silhouette:.4f}")
    if cache_key in gpt_cache:
        return gpt_cache[cache_key]

    # Extract and format similarity data
    intra_sim = cluster_similarities.get('intra', {})
    inter_sim = cluster_similarities.get('inter', {})
    intra_text = "\n".join([f"Cluster {k}: {v:.2f}" for k, v in intra_sim.items()])
    inter_text = "\n".join([f"Clusters {k}: {v:.2f}" for k, v in inter_sim.items()])

    if not intra_sim and not inter_sim:
        return "No similarity scores are available for analysis."

    system_prompt = """You are an industrial equipment analytics expert evaluating cluster similarity metrics.

Present your analysis in a clear, readable format with appropriate spacing. You MUST FOLLOW THIS EXACT FORMAT with these section headings in this order:

1. Similarity Metrics Interpretation
   • Explain what the silhouette score indicates about overall clustering quality
   • Interpret intra-cluster similarity scores in plain language
   • Assess what inter-cluster similarity reveals about cluster separation
   • Use short sentences and concrete examples

2. Cluster Quality Assessment
   • Identify the strongest (most cohesive) and weakest clusters
   • Explain which clusters have clear boundaries versus overlapping ones
   • Assess which clusters are most suitable for automation
   • Note any clusters that might need refinement or subdivision

3. Automation Implications
   • Connect similarity scores to specific automation potential
   • Explain which types of requests could be most reliably automated
   • Identify where standardization would be most effective
   • Suggest concrete next steps based on similarity patterns

IMPORTANT: Use proper spacing between sections. Use bold for important values or conclusions. Keep explanations concise and focused on practical implications. You MUST include ALL 3 sections in the order specified with the exact titles shown."""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Analyze similarity metrics for clustering results:\n\n"
                        f"Method: {method} {parameters}\n"  # Now includes parameters
                        f"Silhouette Score: {silhouette:.2f}\n"
                        f"\nIntra-Cluster Similarity:\n{intra_text}\n"
                        f"\nInter-Cluster Similarity:\n{inter_text}\n"
                        "Focus on implications for automation potential across different request domains."
                    )
                }
            ],
            max_tokens=600,
            temperature=0.2,  # Reduced from 0.3 to improve consistency
            timeout=20
        )
        gpt_reply = response.choices[0].message.content
        
        # Verify response completeness
        required_sections = [
            "1. Similarity Metrics Interpretation", 
            "2. Cluster Quality Assessment", 
            "3. Automation Implications"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in gpt_reply:
                missing_sections.append(section)
                
        if missing_sections:
            gpt_reply += "\n\nNote: Some sections appear to be missing: " + ", ".join(missing_sections) + ". Please ask for clarification if needed."
            
        gpt_cache[cache_key] = gpt_reply
        return gpt_reply
    except Exception as e:
        print(f"GPT API error: {e}")
        return "Unable to analyze similarity scores at the moment."



def analyze_and_respond(user_input: str, data: pd.DataFrame, cluster_results: dict) -> str:
    """
    Main dispatcher for handling user questions about clustering results.
    Dynamically handles parameters for all clustering methods.
    """
    try:
        # Extract key information safely with defaults
        method = cluster_results.get('method', 'Unknown')
        parameters = cluster_results.get('parameters', '')
        silhouette = cluster_results.get('silhouette', 0.0)
        
        # Extract parameters based on the clustering method
        new_parameters = {}
        
        # For K-Means, GMM, BGMM - look for K/components parameter
        k_match = re.search(r'[kK]\s*[=:]\s*(\d+)', user_input.lower())
        components_match = re.search(r'components?\s*[=:]\s*(\d+)', user_input.lower())
        
        # Alternative patterns
        if not k_match:
            k_match = re.search(r'[kK]\s+of\s+(\d+)', user_input.lower())
        if not components_match:
            components_match = re.search(r'components?\s+of\s+(\d+)', user_input.lower())
            
        # For DBSCAN - look for eps and min_samples
        eps_match = re.search(r'eps\s*[=:]\s*(\d+\.\d+)', user_input.lower())
        min_samples_match = re.search(r'min_samples\s*[=:]\s*(\d+)', user_input.lower())
        
        # Alternative patterns
        if not eps_match:
            eps_match = re.search(r'eps\s+of\s+(\d+\.\d+)', user_input.lower())
        if not min_samples_match:
            min_samples_match = re.search(r'min_samples?\s+of\s+(\d+)', user_input.lower())
        
        # Process the parameters based on clustering method
        need_regenerate = False
        model_col = None
        
        if method in ["K-Means", "GMM", "BGMM"] and (k_match or components_match):
            # For K-based methods, extract K/components value
            k_value = int(k_match.group(1) if k_match else components_match.group(1))
            new_parameters["k"] = k_value
            need_regenerate = True
            # Set the appropriate model column
            if method == "K-Means":
                model_col = "Kmeans_Model"
            elif method == "GMM":
                model_col = "GMM_Model"
            else:  # BGMM
                model_col = "BGMM_Model"
            
        elif method == "DBSCAN" and (eps_match or min_samples_match):
            # For DBSCAN, extract eps and min_samples
            if eps_match:
                new_parameters["eps"] = float(eps_match.group(1))
            if min_samples_match:
                new_parameters["min_samples"] = int(min_samples_match.group(1))
            need_regenerate = True
            model_col = "DBSCAN_Model"
        
        # Regenerate clustering if needed
        if need_regenerate and model_col:
            # Get valid embeddings
            valid_embeddings = data["Embeddings"].dropna()
            if len(valid_embeddings) > 0:
                embedding_matrix = np.vstack(valid_embeddings)
                labels = None
                
                # Regenerate clustering based on method
                if method == "K-Means" and "k" in new_parameters:
                    k = new_parameters["k"]
                    labels = run_kmeans(embedding_matrix, k)
                        
                elif method == "GMM" and "k" in new_parameters:
                    k = new_parameters["k"]
                    labels, _, _ = run_gmm(embedding_matrix, k)
                        
                elif method == "BGMM" and "k" in new_parameters:
                    k = new_parameters["k"]
                    labels, _, _ = run_bgmm(embedding_matrix, k)
                        
                elif method == "DBSCAN":
                    # Extract current parameters if not provided
                    current_eps_match = re.search(r'eps=(\d+\.\d+)', parameters)
                    current_ms_match = re.search(r'min_samples=(\d+)', parameters)
                    
                    current_eps = float(current_eps_match.group(1)) if current_eps_match else 0.3
                    current_ms = int(current_ms_match.group(1)) if current_ms_match else 5
                    
                    # Use new parameters if provided, otherwise keep current
                    eps_val = new_parameters.get("eps", current_eps)
                    min_samp = new_parameters.get("min_samples", current_ms)
                    
                    # Run DBSCAN with the parameters
                    labels = run_dbscan(embedding_matrix, eps_val=eps_val, min_samp=min_samp)
                    if labels is not None:
                        parameters = f"(eps={eps_val}, min_samples={min_samp})"
                
                # If we successfully regenerated labels, update everything
                if labels is not None:
                    # Update data with new labels
                    data[model_col] = np.nan
                    data.loc[valid_embeddings.index, model_col] = labels
                    
                    # Calculate new silhouette score
                    if method == "DBSCAN":
                        # For DBSCAN, exclude noise points in silhouette calculation
                        mask = (labels != -1)
                        if sum(mask) > 1 and len(np.unique(labels[mask])) >= 2:
                            silhouette = silhouette_score(embedding_matrix[mask], labels[mask])
                        else:
                            silhouette = 0.0
                    else:
                        # For other methods, use all points
                        if len(np.unique(labels)) >= 2:
                            silhouette = silhouette_score(embedding_matrix, labels)
                        else:
                            silhouette = 0.0
                    
                    # Update parameter string for K-based methods
                    if method in ["K-Means", "GMM", "BGMM"] and "k" in new_parameters:
                        parameters = f"(K={new_parameters['k']})"
                    
                    # Update cluster_results with new parameters and silhouette
                    cluster_results['parameters'] = parameters
                    cluster_results['silhouette'] = silhouette
                    
                    # Clear cache entries related to this method
                    global gpt_cache
                    keys_to_remove = []
                    for key in gpt_cache.keys():
                        if isinstance(key, tuple) and len(key) > 1 and key[1] == method:
                            keys_to_remove.append(key)
                    for key in keys_to_remove:
                        gpt_cache.pop(key, None)
                    
                    # Regenerate patterns and similarities based on new clustering
                    valid_df = data.dropna(subset=[model_col, "Embeddings"])
                    if not valid_df.empty:
                        E_list, L_list = [], []
                        for _, row in valid_df.iterrows():
                            arr = parse_embedding(row["Embeddings"])
                            lab = row[model_col]
                            if arr is not None:
                                E_list.append(arr)
                                L_list.append(int(lab))
                        
                        if E_list:
                            X = np.vstack(E_list)
                            Y = np.array(L_list)
                            intraD, c_list, mat = analyze_cluster_similarity(X, Y)
                            
                            # Update intra similarity dictionary
                            intra_dict = {}
                            for k, v in intraD.items():
                                k_str = str(int(k))
                                intra_dict[k_str] = v
                            
                            # Update inter similarity dictionary
                            inter_dict = {}
                            for i, c1 in enumerate(c_list):
                                for j, c2 in enumerate(c_list):
                                    if i < j:
                                        key_str = f"{int(c1)}-{int(c2)}"
                                        inter_dict[key_str] = mat[i, j]
                            
                            # Update the similarities in cluster_results
                            cluster_results['similarities'] = {
                                'intra': intra_dict,
                                'inter': inter_dict
                            }
                            
                            # Extract representative requests for each cluster
                            reps = extract_representative_requests(valid_df, X, Y, n_representatives=5)
                            cluster_sizes = valid_df[model_col].value_counts()
                            total_requests = cluster_sizes.sum()
                            
                            # Update patterns
                            patterns_dict = {}
                            for cluster_id, requests in reps.items():
                                if cluster_id == -1:
                                    continue  # Skip noise clusters
                                cluster_id_str = str(int(cluster_id))
                                size_percentage = (cluster_sizes.get(cluster_id, 0) / total_requests) * 100
                                
                                patterns_dict[cluster_id_str] = {
                                    'size_percentage': size_percentage,
                                    'requests': requests,
                                    'automation_potential': (
                                        'high' if intraD.get(cluster_id, 0) > 0.7
                                        else 'medium' if intraD.get(cluster_id, 0) > 0.5
                                        else 'low'
                                    ),
                                    'intra_similarity': intraD.get(cluster_id, 0)
                                }
                            
                            # Update patterns in cluster_results
                            cluster_results['patterns'] = patterns_dict
                            
                            # Optionally recalculate time savings if necessary
                            if 'total_time_saving' in st.session_state:
                                total_saving = st.session_state['total_time_saving']
                                time_savings = {}
                                for cl_id in cluster_sizes.index:
                                    if cl_id == -1:
                                        continue  # Skip noise
                                    size_factor = cluster_sizes[cl_id] / cluster_sizes.sum()
                                    similarity_factor = intraD.get(cl_id, 0)
                                    confidence = 'high' if similarity_factor > 0.7 else 'medium'
                                    cluster_id_str = str(int(cl_id))
                                    time_savings[cluster_id_str] = {
                                        'minutes': total_saving * size_factor * similarity_factor,
                                        'confidence': confidence
                                    }
                                cluster_results['time_savings'] = time_savings
        
        # Check if user is asking about a specific cluster
        cluster_match = re.search(r"cluster\s+(\d+)", user_input.lower())
        
        # Create method header
        if cluster_match:
            specific_cluster = cluster_match.group(1)
            method_header = f"\n**Analysis for {method} cluster {specific_cluster} {parameters} (Silhouette={silhouette:.2f}):**\n\n"
        else:
            method_header = f"\n**Analysis for {method} {parameters} (Silhouette={silhouette:.2f}):**\n\n"

        # Special case for main insights/patterns questions
        if any(phrase in user_input.lower() for phrase in [
            "main insight", "main insights", "key insight", "overview", 
            "summary", "analyze", "analysis", "theme", "pattern"
        ]):
            detailed_response = generate_gpt_theme_analysis(
                cluster_results.get('patterns', {}),
                method,
                silhouette,
                cluster_results.get('similarities', {}),
                cluster_results.get('time_savings', {}),
                parameters=parameters
            )
            return method_header + detailed_response

        # Special case for similarity questions
        if any(phrase in user_input.lower() for phrase in [
            "similarity", "cohesion", "coherence", "related", "relationship",
            "connection", "overlap", "distance"
        ]):
            similarity_response = generate_gpt_similarity_analysis(
                cluster_results.get('similarities', {}),
                method,
                silhouette,
                parameters=parameters
            )
            return method_header + similarity_response

        # Try to get cached response for other question types
        cached_answer = get_cached_response(user_input, cluster_results)
        if cached_answer:
            return method_header + cached_answer

        # Specific cluster analysis
        if cluster_match:
            cluster_id = cluster_match.group(1)
            cluster_response = generate_gpt_theme_analysis(
                cluster_results.get('patterns', {}),
                method,
                silhouette,
                cluster_results.get('similarities', {}),
                cluster_results.get('time_savings', {}),
                cluster_id=cluster_id,
                parameters=parameters
            )
            return method_header + cluster_response

        # Default to general analysis
        return method_header + generate_gpt_theme_analysis(
            cluster_results.get('patterns', {}),
            method,
            silhouette,
            cluster_results.get('similarities', {}),
            cluster_results.get('time_savings', {}),
            parameters=parameters
        )
        
    except Exception as e:
        print(f"Error in analyze_and_respond: {str(e)}")
        import traceback
        traceback.print_exc()
        return "I apologize, but I encountered an error while analyzing. Please try again."
        

def chat_interface(data: pd.DataFrame, cluster_results: dict, method_model1=None, param_display=""):
    """
    Implements an interactive chat interface for analyzing clustering results.
    """
    st.header("Ask Questions About the Analysis")

    # Initialize session states if not exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_processed" not in st.session_state:
        st.session_state.last_processed = set()

    # Create container for chat
    chat_container = st.container()
    
    # Display existing chat history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Get user input
    user_input = st.text_input(
        "Ask about these clustering results:",
        key=f"user_input_{method_model1}_{param_display}" if method_model1 else "user_input",  # fallback
        placeholder="e.g., 'What are the main insights?' or 'Tell me about cluster 2'"
    )

    # Process new input if not already processed
    if user_input and user_input not in st.session_state.last_processed:
        with chat_container:
            # Show user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Process and show assistant response
            with st.chat_message("assistant"):
                try:
                    # Create status indicator
                    with st.status("Analyzing...", expanded=False) as status:
                        # Get response from your analyze_and_respond() logic
                        response = analyze_and_respond(user_input, data, cluster_results)
                        
                        # Mark success
                        status.update(label="Analysis complete!", state="complete")
                        
                        # Format the GPT reply for readability
                        formatted_response = format_gpt_response(response)

                        # Display the formatted Markdown text
                        st.markdown(formatted_response)

                        # Update conversation history
                        st.session_state.messages.extend([
                            {"role": "user", "content": user_input},
                            {"role": "assistant", "content": response}
                        ])
                        st.session_state.last_processed.add(user_input)

                except Exception as e:
                    # Handle errors gracefully
                    status.update(label="Analysis failed", state="error")
                    st.error("I apologize, but I encountered an error. Please try again.")
                    print(f"Error in chat interface: {str(e)}")
                    traceback.print_exc()
                        
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

def generate_gpt_theme_analysis_for_cluster(requests, cluster_id):
    """
    Dynamically calls GPT to analyze themes for a specific cluster.
    We KEEP this function in case we want to call it on-demand (e.g., in the chat),
    but we do NOT invoke it automatically in process_clustering_results.
    """
    if not requests:
        return "No requests available to analyze."

    messages = [
        {
            "role": "system",
            "content": "You are a customer service analytics expert. Identify key themes from customer requests."
        },
        {
            "role": "user",
            "content": (
                f"Analyze the themes in these customer requests for Cluster {cluster_id}:\n\n"
                f"{json.dumps(requests)}\n\n"
                "Summarize the primary topics, request types, and complexity levels."
            )
        }
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=250,
            temperature=0.2,
            timeout=10
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"GPT API error: {str(e)}")
        return "Error: GPT theme analysis could not be retrieved."


def process_clustering_results(
    valid_df, 
    cluster_col, 
    cluster_patterns, 
    cluster_similarities,
    time_savings, 
    method_model1, 
    param_display, 
    silhouette_for_summary
):
    """
    This function processes numeric cluster results (like similarity) 
    but does NOT call GPT automatically for each cluster. 
    GPT calls can happen later if the user asks a question in the chat.
    """
    E_list, L_list = [], []
    for _, row in valid_df.iterrows():
        arr = parse_embedding(row["Embeddings"])
        lab = row[cluster_col]
        if arr is not None:
            E_list.append(arr)
            L_list.append(int(lab))  # cast cluster label to plain int

    if E_list:
        # 1) Compute cluster similarity metrics (no GPT calls here)
        X = np.vstack(E_list)
        Y = np.array(L_list)
        intraD, c_list, mat = analyze_cluster_similarity(X, Y)

        # Fill in 'intra' similarity
        cluster_similarities['intra'] = {}
        for k, v in intraD.items():
            k_str = str(int(k))
            cluster_similarities['intra'][k_str] = v

        # Fill in 'inter' similarity
        cluster_similarities['inter'] = {}
        for i, c1 in enumerate(c_list):
            for j, c2 in enumerate(c_list):
                if i < j:
                    key_str = f"{int(c1)}-{int(c2)}"
                    cluster_similarities['inter'][key_str] = mat[i, j]

        # 2) Extract representative requests (still no GPT calls)
        reps = extract_representative_requests(valid_df, X, Y, n_representatives=5)
        cluster_sizes = valid_df[cluster_col].value_counts()
        total_requests = cluster_sizes.sum()

        # 3) Store basic cluster info (size, reps) in 'cluster_patterns' 
        #    WITHOUT calling GPT
        for cluster_id, requests in reps.items():
            if cluster_id == -1: 
                continue  # skip noise
            cluster_id_str = str(int(cluster_id))
            size_percentage = (cluster_sizes.get(cluster_id, 0) / total_requests) * 100
            
            cluster_patterns[cluster_id_str] = {
                'size_percentage': size_percentage,
                'requests': requests,
                'automation_potential': (
                    'high' if intraD.get(cluster_id, 0) > 0.7
                    else 'medium' if intraD.get(cluster_id, 0) > 0.5
                    else 'low'
                ),
                'intra_similarity': intraD.get(cluster_id, 0)  # Added for better GPT analysis
            }

        # 4) Optionally store time-savings data (no GPT calls)
        if 'total_time_saving' in st.session_state:
            total_saving = st.session_state['total_time_saving']
            for cl_id in cluster_sizes.index:
                if cl_id == -1:
                    continue  # skip noise
                size_factor = cluster_sizes[cl_id] / cluster_sizes.sum()
                similarity_factor = intraD.get(cl_id, 0)
                confidence = 'high' if similarity_factor > 0.7 else 'medium'
                cluster_id_str = str(int(cl_id))
                time_savings[cluster_id_str] = {
                    'minutes': total_saving * size_factor * similarity_factor,
                    'confidence': confidence
                }

        # 5) Finally, launch the chat interface
        #    The user can trigger GPT calls by asking questions.
        chat_interface(
            data=valid_df,
            cluster_results={
                'method': method_model1,
                'parameters': param_display,
                'silhouette': silhouette_for_summary,
                'similarities': cluster_similarities,
                'patterns': cluster_patterns,
                'time_savings': time_savings
            },
            method_model1=method_model1,  # Pass the method_model1 parameter
            param_display=param_display   # Pass the param_display parameter
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

    # Initialize session state for configuration tracking
    if "current_config" not in st.session_state:
        st.session_state.current_config = {
            "method": None, 
            "params": "", 
            "silhouette": 0.0,
            "cluster_col": None
        }

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

    # Create the new configuration
    new_config = {
        "method": method_model1,
        "params": param_display,
        "silhouette": silhouette_for_summary,
        "cluster_col": cluster_col
    }
    
    # Check if configuration has changed
    # Get the old config values safely with a default value for comparison
    old_method = st.session_state.current_config.get("method")
    old_params = st.session_state.current_config.get("params", "")
    old_silhouette = st.session_state.current_config.get("silhouette", 0.0)
    old_cluster_col = st.session_state.current_config.get("cluster_col")
    
    config_changed = (old_method != method_model1 or
                      old_params != param_display or
                      old_silhouette != silhouette_for_summary or
                      old_cluster_col != cluster_col)
    
    if config_changed:
        # Clear the GPT cache when configuration changes
        global gpt_cache
        gpt_cache = {}
        # Update the session state
        st.session_state.current_config = new_config
        # Reset chat history when configuration changes
        if "messages" in st.session_state:
            st.session_state.messages = []
        if "last_processed" in st.session_state:
            st.session_state.last_processed = set()

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
