import torch


def compute_neighbor_distances(query: torch.Tensor, reference: torch.Tensor, n_neighbors: int = 5, chunk_size: int | None = None) -> torch.Tensor:
    chunk_size = chunk_size or query.shape[0]
    neighbor_distances = []

    for start in range(0, query.shape[0], chunk_size):
        end = start + chunk_size
        distances = torch.cdist(query[start:end], reference)
        neighbor_distances.append(
            torch.kthvalue(distances, n_neighbors + 1, dim=-1).values
        )

    return torch.cat(neighbor_distances, dim=0)
