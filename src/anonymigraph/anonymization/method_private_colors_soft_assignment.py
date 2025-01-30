import logging

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from anonymigraph.anonymization._method_private_colors.ccm_sampler import CCMSampler
from anonymigraph.anonymization.abstract_anonymizer import AbstractAnonymizer
from anonymigraph.utils import calculate_katz

logger = logging.getLogger(__name__)


class LamScheduler:
    def __init__(self, initial_lam=1e-3, factor=1.2, patience=40, threshold=1e-2):
        self.initial_lam = initial_lam
        self.lam = initial_lam
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.num_bad_epochs = 0
        self.best_loss = None

    def step(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
        else:
            if current_loss < self.best_loss - self.threshold:
                self.best_loss = current_loss
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs > self.patience:
                self._adjust_lam()
                self.num_bad_epochs = 0
                self.best_loss = None

    def _adjust_lam(self):
        self.lam *= self.factor
        # logger.info(f"LamScheduler: Increasing Entropy Regularizer to {self.lam}")


class SoftColorOptimizer:
    def __init__(
        self,
        G,
        k_max=30,
        w=1,
        use_katz_utility=True,
        alpha=0.01,
        beta=1,
        eps_utility=1e-5,
        eps_privacy=1e-7,
        lr=0.008,
        patience=40,
        threshold=1e-1,
        initial_lam=1e-3,
        factor=1.2,
        device="cpu",
        seed=4,
        use_entropy_reg=True,
    ):
        self.G = G

        self.k_max = k_max  # number of max possible colors
        self.w = w  # linearization parameter w

        self.use_entropy_reg = use_entropy_reg

        self.use_katz_utility = use_katz_utility
        self.alpha = alpha
        self.beta = beta

        self.eps_utility = eps_utility
        self.eps_privacy = eps_privacy

        self.lr = lr
        self.patience = patience
        self.threshold = threshold
        self.initial_lam = initial_lam
        self.factor = factor

        self.seed = seed

        self.device = device
        self._prepare_data()
        self._initialize_parameters()
        self.loss_log = []

    def _prepare_data(self):
        A_scipy = nx.adjacency_matrix(self.G).astype(np.float64)
        self.A_scipy = A_scipy
        self.n = A_scipy.shape[0]

        I, J = A_scipy.nonzero()
        mask = I < J

        self.I = torch.from_numpy(I[mask]).long().to(self.device)
        self.J = torch.from_numpy(J[mask]).long().to(self.device)

        crow_indices = torch.from_numpy(A_scipy.indptr).to(torch.int64)
        col_indices = torch.from_numpy(A_scipy.indices).to(torch.int64)
        values = torch.from_numpy(A_scipy.data).to(torch.float32)
        self.A_pytorch = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=A_scipy.shape)

        katz_centrality = calculate_katz(self.A_scipy, alpha=self.alpha, beta=self.beta)
        self.katz_centrality = torch.from_numpy(katz_centrality).float().to(self.device)
        self.katz_centrality_norm = self.katz_centrality.norm() ** 2

    def _initialize_parameters(self):
        torch.manual_seed(self.seed)
        self.H_logits = torch.empty(self.n, self.k_max, device=self.device, requires_grad=True)
        nn.init.normal_(self.H_logits, std=0.8)

    def fit(self, max_epochs=int(1e10), epoch_report_frequency=5):
        optimizer = optim.Adam([self.H_logits], lr=self.lr)
        entropy_scheduler = LamScheduler(
            initial_lam=self.initial_lam, patience=self.patience, threshold=self.threshold, factor=self.factor
        )

        for epoch in range(max_epochs):
            H = F.softmax(self.H_logits, dim=1)
            AH = torch.sparse.mm(self.A_pytorch.to(self.device), H)

            if self.use_katz_utility:
                loss_U = self.utility_katz(H)
            else:
                loss_U = self.utility_short_term(AH, H)

            loss_P = self.privacy_undirected(AH, H, self.I, self.J)
            entropy = self.entropy(H)
            total_loss = loss_U + self.w * loss_P + entropy_scheduler.lam * entropy

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            self.loss_log.append(total_loss.item())

            entropy_scheduler.step(total_loss.item())

            if epoch % epoch_report_frequency == 0:
                logger.info(
                    f"Epoch {epoch}, Total Loss: {total_loss.item():.4f}, "
                    f"Loss U: {loss_U:.4f}, Loss P: {loss_P:.4f}, "
                    f"Entropy: {entropy:.4f}, Entropy Lam: {entropy_scheduler.lam}"
                )

            if not self.use_entropy_reg:
                # break as soon as entropy reg increases
                if entropy_scheduler.lam > entropy_scheduler.initial_lam:
                    break

            if entropy < 0.1:
                break

        logger.info(
            f"Epoch {epoch}, Total Loss: {total_loss.item():.4f}, "
            f"Loss U: {loss_U:.4f}, Loss P: {loss_P:.4f}, "
            f"Entropy: {entropy:.4f}, Entropy Lam: {entropy_scheduler.lam}"
        )

        self.H_final = F.softmax(self.H_logits, dim=1).detach().cpu().numpy()
        self.clusters = torch.argmax(self.H_logits, dim=1).detach().cpu().numpy()

        unique_labels = np.unique(self.clusters)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        self.colors = {
            node: label_mapping[self.clusters[node]] for node in list(self.G.nodes())
        }  # relabel and remove excess clusters

        used_clusters = unique_labels.shape[0]
        max_num_clusters = H.shape[1]
        logger.info(f"used clusters / max_clusters: {used_clusters}/{max_num_clusters}")

        if used_clusters == max_num_clusters:
            logger.info("WARNING: used clusters are equal to max possible clusters. Increase max possible clusters.")

    def utility_katz(self, H):
        Stx_sqr = (H.T @ self.katz_centrality) ** 2
        n = H.sum(dim=0) + self.eps_utility
        loss_U = self.katz_centrality_norm - (Stx_sqr / n).sum()
        return loss_U

    def entropy(self, H):
        return -torch.sum(H * torch.log(H + 1e-10))

    def utility_short_term(self, AH, H):
        Stx_sqr = (H.T @ AH) ** 2
        n = H.sum(dim=0) + self.eps_utility
        loss_U = (AH**2).sum() - (Stx_sqr / n).sum()
        return loss_U

    def privacy_undirected(self, AH, H, I, J):
        HtAH = H.T @ AH
        loss_P = (((AH[I] * H[J]) @ (HtAH + self.eps_privacy).reciprocal()) * AH[J] * H[I]).sum()
        return loss_P

    def plot_partition_matrix(self):
        import matplotlib.pyplot as plt

        plt.imshow(self.H_final, aspect="auto", cmap="viridis")
        plt.xlabel("Color")
        plt.ylabel("Node Idx")
        plt.title("Partition Matrix Heatmap")
        plt.show()


class SoftColorAnonymizer(AbstractAnonymizer):
    def __init__(
        self,
        k_max=30,
        w=1,
        use_katz_utility=True,
        alpha=0.01,
        beta=1,
        eps_utility=1e-5,
        eps_privacy=1e-7,
        lr=0.008,
        patience=40,
        threshold=1e-1,
        initial_lam=1e-3,
        factor=1.2,
        device="cpu",
        seed=4,
        use_entropy_reg=True,
    ):
        """
        Implements the anonymizer based on our technique when using the Katz or Short-term utility loss and using an gradient-based (soft assignment) optimization approach.
        """

        self.k_max = k_max  # number of max possible colors
        self.w = w  # privacy parameter w

        self.use_entropy_reg = use_entropy_reg

        # Centralities
        self.use_katz_utility = use_katz_utility  # if true uses katz centrality otherwise short term
        self.alpha = alpha
        self.beta = beta

        # Epsilons used to avoid dividing by zero in utility and privacy functions
        self.eps_utility = eps_utility
        self.eps_privacy = eps_privacy

        # Optimization hyperparameters
        self.lr = lr
        self.patience = patience
        self.threshold = threshold
        self.initial_lam = initial_lam
        self.factor = factor

        self.seed = seed

        self.device = device

    def anonymize(self, G: nx.Graph, random_seed=None, epoch_report_frequency=10, max_epochs=int(1e8)) -> nx.Graph:
        soft_optim_katz = SoftColorOptimizer(
            G,
            k_max=self.k_max,
            w=self.w,
            use_katz_utility=self.use_katz_utility,
            use_entropy_reg=self.use_entropy_reg,
            alpha=self.alpha,
            beta=self.beta,
            eps_utility=self.eps_utility,
            eps_privacy=self.eps_privacy,
            lr=self.lr,
            patience=self.patience,
            threshold=self.threshold,
            initial_lam=self.initial_lam,
            factor=self.factor,
            device=self.device,
            seed=random_seed,
        )

        soft_optim_katz.fit(max_epochs=max_epochs, epoch_report_frequency=epoch_report_frequency)

        A, A_prime = CCMSampler(G, soft_optim_katz.colors, use_pseudograph_to_sample=False).sample(seed=random_seed)

        # Create networkx anonyimzed graph from A_prime (using same node order as G)
        Ga = nx.from_scipy_sparse_array(A_prime)
        order = list(G.nodes())
        relabel_map = {i: order[i] for i in range(len(order))}
        G_new = nx.relabel_nodes(Ga, relabel_map)
        return G_new
