"""This module contains an example of a Gaussian Mixture Model (GMM) trained with maximum likelihood estimation.
Usually Expectation Maximization is used, but for the sake of learning, we'll use gradient descent."""

from typing import Tuple, Optional
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import multivariate_normal
import numpy as np
from matplotlib import pyplot as plt

DIMS = 2


class GaussianMixtureModel(nn.Module):
    def __init__(self, num_clusters: int, dims: int = DIMS) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.dims = dims
        self.means = nn.Parameter(T.randn(num_clusters, dims))
        self.covariances = nn.Parameter(T.randn(num_clusters) ** 2)
        self.weights = nn.Parameter(T.ones(num_clusters) / num_clusters)

    def forward(self, x: T.Tensor, labels: Optional[T.Tensor]) -> T.Tensor:
        densities = T.stack([
            self.weights[i]
            * T.exp(multivariate_normal.MultivariateNormal(self.means[i], T.eye(self.dims) * self.exp(self.covariances[i])).log_prob(x))
            for i in range(self.num_clusters)
        ], dim=-1)
        loss = - T.log(densities.mean()) + (1-self.weights.mean())
        return loss



def _generate_data(
    n_datapoints: int, num_clusters: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a mixture dataset from num_clusters of n_datapoints.
    """
    mixtures = np.random.uniform(size=num_clusters)
    mixtures = mixtures / sum(mixtures)
    means = np.random.normal(size=(num_clusters, DIMS)) * 2
    print("True means:", means)
    covs = np.random.normal(size=(num_clusters, DIMS, DIMS)) ** 2
    labels = []
    data = []
    for i, (mean, cov) in enumerate(zip(means, covs)):
        n_from_mixture = int(mixtures[i] * n_datapoints)
        data_from_dist = np.random.multivariate_normal(mean, cov, size=n_from_mixture)
        data.append(data_from_dist)
        labels.append(np.repeat(i, repeats=n_from_mixture))
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    return data, labels


def _plot(data: np.ndarray, labels: np.ndarray) -> None:
    _ = plt.figure(figsize=(12, 8))
    plt.scatter(x=data[:, 0], y=data[:, 1], c=labels)
    plt.savefig("fig.png")
    plt.close()


def main() -> None:
    data, labels = _generate_data(1000, 3)
    _plot(data, labels)
    model = GaussianMixtureModel(3)
    optimizer = optim.AdamW(model.parameters())
    print(model.means)
    for _ in range(1000):
        optimizer.zero_grad()
        loss = model(T.tensor(data), T.tensor(labels))
        loss.backward()
        optimizer.step()
    print(model.means)

    print(loss)


if __name__ == "__main__":
    main()
