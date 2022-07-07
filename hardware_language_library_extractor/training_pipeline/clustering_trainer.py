import os
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.multiprocessing import Pool, set_start_method
import scipy.cluster.hierarchy as shc

from hardware_language_library_extractor.training_pipeline.config import TRAINING_DATA_BASE_PATH, \
    HARDWARE_SENTENCE_EMBEDDINGS, LANGUAGE_SENTENCE_EMBEDDINGS, LIBRARY_SENTENCE_EMBEDDINGS, MODELS_FOLDER_BASE_PATH, \
    CLUSTER_IMAGES, OUTPUT_FOLDER_BASE_PATH

from hardware_language_library_extractor.common.util import create_folder

file_names = [HARDWARE_SENTENCE_EMBEDDINGS, LANGUAGE_SENTENCE_EMBEDDINGS, LIBRARY_SENTENCE_EMBEDDINGS]


def load_sentence_embeddings(file_name):
    df = pd.DataFrame.from_records(np.loadtxt(os.path.join(TRAINING_DATA_BASE_PATH, file_name)))
    return df


def print_cluster_assignments(df, cluster_assignment):
    for i in range(1, df.shape[0]):
        print('{} => {}'.format(df[i:], cluster_assignment[i]))


def get_clusters(df, num_clusters=5):
    clustering_model = MiniBatchKMeans(n_clusters=num_clusters)
    clustering_model.fit(df)
    cluster_assignment = clustering_model.labels_
    print('predicted cluster value: {}'.format(clustering_model.predict(df[:1])))
    print(clustering_model.cluster_centers_)
    print("Silhouette Coefficient: %0.3f" % silhouette_score(df, cluster_assignment, sample_size=df.shape[0]))
    print_cluster_assignments(df, cluster_assignment)
    return clustering_model


def pca(df):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df)
    principal_df = pd.DataFrame(data=principal_components, columns=[0, 1])
    return principal_df


def draw_dendogram(principal_df, feature):
    plt.figure(figsize=(8, 8))
    plt.title('Visualising the dendogram of the {} data'.format(feature))
    dendrogram = shc.dendrogram((shc.linkage(principal_df, method='ward')))
    plt.show()


def agglomerative_clustering(x, principal_df, file_name, n_clusters):
    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    cluster.fit_predict(x)
    plt.figure(figsize=(10, 7))
    plt.scatter(principal_df[0], principal_df[1], c=cluster.labels_, cmap='rainbow')
    plt.title('Visualizing the cluster of the {} data'.format(file_name.split('_')[0]))
    plt.show()


def plot_clusters(x, principal_df, file_name):
    range_n_clusters = [2, 3, 4, 5, 6]
    all_cluster_labels = dict()
    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(x) + (n_clusters + 1) * 10])
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=10, batch_size=1000)
        cluster_labels = clusterer.fit_predict(x)
        all_cluster_labels[n_clusters] = cluster_labels.tolist()
        pickle.dump(clusterer,
                    open(os.path.join(MODELS_FOLDER_BASE_PATH, '{}_cluster{}.sav'.format(file_name.split('_')[0],
                                                                                         n_clusters)), 'wb'))
        silhouette_avg = silhouette_score(x, cluster_labels)
        print("For {} & n_clusters =".format(file_name.split('_')[0]), n_clusters, "The average silhouette_score is :",
              silhouette_avg)
        sample_silhouette_values = silhouette_samples(x, cluster_labels)
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                              edgecolor=color, alpha=0.7)
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10  # 10 for the 0 samples
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(principal_df[0], principal_df[1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')
        centers = clusterer.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data ""with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
        plt.savefig(os.path.join(OUTPUT_FOLDER_BASE_PATH, CLUSTER_IMAGES, '{}_cluster{}.png'.format(
            file_name.split('_')[0], n_clusters)))
    plt.show()
    return all_cluster_labels


def write_json_output_to_file(output, path):
    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(output, outfile, sort_keys=False, indent=4)


def driver(file_name):
    df = load_sentence_embeddings(file_name)
    principal_df = pca(df)
    # draw_dendogram(principal_df, file_name.split('_')[0])
    agglomerative_clustering(df, principal_df, file_name, 3)
    all_cluster_labels = plot_clusters(df, principal_df, file_name)
    # writeJsonOutputToFile(all_cluster_labels, os.path.join(base_path, file_name.split('_')[0] + cluster_mapping))
    # clustering_model = get_clusters(df)


def main():
    create_folder(path=OUTPUT_FOLDER_BASE_PATH, name=CLUSTER_IMAGES, recursive=True)
    try:
        set_start_method('spawn')
        with Pool(processes=3) as pool:
            pool.map(driver, file_names)
    except RuntimeError:
        pass


if __name__ == '__main__':
    main()
