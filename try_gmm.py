import pandas
import  matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from sklearn import metrics

dataset = pandas.read_csv("dataset.csv", header=None)

print(dataset.head())

plt.scatter(dataset[0],dataset[1])
plt.savefig("scatter.png")

print("Silhouette score for kmean: ")
for i in range(5):
	n = i+2
	kmc_machine = KMeans(n_clusters = n)
	kmc_machine.fit(dataset)
	kmc_results = kmc_machine.predict(dataset)
	plt.scatter(dataset[0], dataset[1], c=kmc_results)
	plt.savefig("scatter_kmean" + str(n) + ".png")
	print("n =" + str(n) + ": " + str(metrics.silhouette_score(dataset, kmc_results)))

print("Silhouette score for gmm: ")
for i in range(5):
	n=i+2
	gmm_machine = GaussianMixture(n_components=n)
	gmm_machine.fit(dataset)
	gmm_results = gmm_machine.predict(dataset)
	plt.scatter(dataset[0], dataset[1], c=gmm_results)
	plt.savefig("scatter_gmm" + str(n) + ".png")
	print("n =" + str(n) + ": " + str(metrics.silhouette_score(dataset, gmm_results)))



