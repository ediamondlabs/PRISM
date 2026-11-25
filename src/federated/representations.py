"""
Representation extraction for federated learning.

Extracts different representation types from LSTM autoencoder:
- Weights (W): Full model state_dict (~2-10 MB depending on architecture)
- Embeddings (E): dz-dim trajectory embeddings (~100 KB for 100 samples)
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans


class RepresentationExtractor:
    """Extract different representation types from trained models."""

    def __init__(
        self,
        embedding_dim: int = 32,
        num_prototypes: int = 5,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize representation extractor.

        Args:
            embedding_dim: Dimension of embeddings
            num_prototypes: Number of prototype clusters (k in K-Means)
            device: Device for tensor operations
        """
        self.embedding_dim = embedding_dim
        self.num_prototypes = num_prototypes
        self.device = device

        # Mini-batch K-Means for online prototype extraction
        # Note: sklearn uses float64 internally, we convert to float32 when returning
        self.kmeans = MiniBatchKMeans(
            n_clusters=num_prototypes,
            random_state=42,
            batch_size=32,
            max_iter=100,
        )
        self.kmeans_initialized = False

    def extract_embeddings(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        num_samples: Optional[int] = 100,
    ) -> np.ndarray:
        """Extract embeddings from model on private data.

        Args:
            model: LSTM autoencoder model
            data_loader: DataLoader with vehicle's private data
            num_samples: Number of samples to extract (None = all data)

        Returns:
            Embeddings array of shape (num_samples, embedding_dim)
        """
        model.eval()
        embeddings_list = []
        total_extracted = 0

        # Get model's device
        model_device = next(model.parameters()).device

        with torch.no_grad():
            for batch in data_loader:
                # Batch is (batch_size, seq_len, features)
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch

                # Move data to same device as model
                x = x.to(model_device)

                # Get embeddings from encoder
                emb = model.encode(x)  # (batch_size, embedding_dim)
                embeddings_list.append(emb.cpu().numpy().astype(np.float32))

                total_extracted += emb.shape[0]

                # Stop if we've collected enough samples
                if num_samples is not None and total_extracted >= num_samples:
                    break

        # Handle empty data loader
        if len(embeddings_list) == 0:
            # Return empty array with correct shape (0, embedding_dim)
            # Get embedding_dim from model
            with torch.no_grad():
                dummy_input = torch.zeros(1, 1, model.encoder.input_size).to(model_device)
                dummy_emb = model.encode(dummy_input)
                embedding_dim = dummy_emb.shape[1]
            return np.zeros((0, embedding_dim), dtype=np.float32)

        # Concatenate all embeddings
        all_embeddings = np.concatenate(embeddings_list, axis=0)

        # Trim to exact num_samples if specified
        if num_samples is not None:
            all_embeddings = all_embeddings[:num_samples]

        return all_embeddings

    def extract_prototypes(
        self,
        embeddings: np.ndarray,
        update_clusters: bool = True,
    ) -> np.ndarray:
        """Extract prototypes (cluster centroids) from embeddings.

        Uses Mini-batch K-Means for online clustering, suitable for
        streaming/incremental updates.

        Args:
            embeddings: Embeddings array (num_samples, embedding_dim)
            update_clusters: Whether to update cluster centers (True for training)

        Returns:
            Prototype centroids of shape (num_prototypes, embedding_dim)
        """
        if embeddings.shape[0] == 0:
            # Return zero prototypes if no data
            return np.zeros((self.num_prototypes, self.embedding_dim), dtype=np.float32)

        # Normalize embeddings to prevent numerical instability in KMeans
        # This prevents cluster centers from growing unbounded
        embeddings_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = np.where(embeddings_norm == 0, 1.0, embeddings_norm)
        embeddings = embeddings / embeddings_norm

        # Keep embeddings as float64 for sklearn (sklearn uses float64 internally)
        # We'll convert to float32 only when returning cluster centers
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float64)

        if update_clusters:
            # Fit/update clusters on new embeddings (sklearn uses float64 internally)
            if not self.kmeans_initialized:
                # First fit - use fit() with float64
                self.kmeans.fit(embeddings)
                self.kmeans_initialized = True
            else:
                # Subsequent updates - use partial_fit with float64
                self.kmeans.partial_fit(embeddings)
        else:
            # Just predict without updating (for inference)
            if self.kmeans_initialized:
                self.kmeans.predict(embeddings)

        # Return cluster centers as prototypes (convert from float64 to float32)
        # Clip to prevent overflow when converting
        centroids = np.clip(self.kmeans.cluster_centers_, -1e10, 1e10).astype(np.float32)

        # Check for invalid values and replace with zeros
        if np.any(np.isnan(centroids)) or np.any(np.isinf(centroids)):
            centroids = np.nan_to_num(centroids, nan=0.0, posinf=0.0, neginf=0.0)

        return centroids

    def extract_weights(
        self,
        model: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Extract full model weights as state_dict (paper Algorithm 1, Phase 3).

        Returns a copy of the model's state_dict for transmission to neighbors.
        No sparsification — paper doesn't mention it and the RL agent controls
        the W/E ratio via the WHAT head.

        Args:
            model: Model to extract weights from

        Returns:
            State dict with model parameters (all tensors moved to CPU)
        """
        state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return state_dict

    def estimate_size(
        self,
        repr_type: str,
        num_samples: Optional[int] = None,
        model: Optional[nn.Module] = None,
    ) -> float:
        """Estimate size of representation in MB.

        Args:
            repr_type: 'E'=Embeddings, 'W'=Weights
            num_samples: Number of embedding samples (only used for 'E')
            model: Model instance (used for accurate 'W' estimation)

        Returns:
            Estimated size in MB
        """
        if repr_type == "E":
            samples = num_samples if num_samples is not None else 100
            size_bytes = samples * self.embedding_dim * 4
            return size_bytes / (1024 * 1024)

        elif repr_type == "W":
            if model is not None:
                size_bytes = sum(p.numel() * 4 for p in model.parameters())
                return size_bytes / (1024 * 1024)
            return 10.0  # Fallback estimate

        else:
            raise ValueError(f"Unknown representation type: {repr_type}")

    def compress_embeddings(
        self,
        embeddings: np.ndarray,
        bits: int = 8,
    ) -> Tuple[np.ndarray, Dict]:
        """Compress embeddings using quantization (optional, for bandwidth savings).

        Args:
            embeddings: Embeddings to compress (num_samples, embedding_dim)
            bits: Quantization bits (8 = uint8, 16 = uint16)

        Returns:
            Tuple of (compressed_embeddings, compression_info)
            where compression_info contains min/max for decompression
        """
        # Find min/max per dimension
        emb_min = embeddings.min(axis=0)
        emb_max = embeddings.max(axis=0)

        # Avoid division by zero
        emb_range = emb_max - emb_min
        emb_range = np.where(emb_range == 0, 1.0, emb_range)

        # Scale to [0, 2^bits - 1]
        if bits == 8:
            max_val = 255
            dtype = np.uint8
        elif bits == 16:
            max_val = 65535
            dtype = np.uint16
        else:
            raise ValueError(f"Unsupported bits: {bits}. Use 8 or 16.")

        # Normalize and quantize
        normalized = (embeddings - emb_min) / emb_range
        quantized = (normalized * max_val).astype(dtype)

        # Store compression info for decompression
        compression_info = {
            "min": emb_min,
            "max": emb_max,
            "bits": bits,
        }

        return quantized, compression_info

    def decompress_embeddings(
        self,
        compressed: np.ndarray,
        compression_info: Dict,
    ) -> np.ndarray:
        """Decompress quantized embeddings.

        Args:
            compressed: Compressed embeddings
            compression_info: Info from compress_embeddings()

        Returns:
            Decompressed embeddings (float32)
        """
        bits = compression_info["bits"]
        emb_min = compression_info["min"]
        emb_max = compression_info["max"]

        # Determine max quantization value
        if bits == 8:
            max_val = 255
        elif bits == 16:
            max_val = 65535
        else:
            raise ValueError(f"Unsupported bits: {bits}")

        # Denormalize
        normalized = compressed.astype(np.float32) / max_val
        emb_range = emb_max - emb_min
        decompressed = normalized * emb_range + emb_min

        return decompressed

    def extract_and_package(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        repr_type: str,
        num_samples: int = 100,
        compress: bool = False,
    ) -> Dict:
        """Extract representation and package with metadata.

        Args:
            model: LSTM autoencoder
            data_loader: Private data loader
            repr_type: 'W' (weights) or 'E' (embeddings)
            num_samples: Number of embedding samples (only for 'E')
            compress: Whether to compress embeddings (8-bit quantization)

        Returns:
            Package dict with type, data, size_mb, compressed, compression_info
        """
        if repr_type == "W":
            data = self.extract_weights(model)
            compression_info = None

        elif repr_type == "E":
            embeddings = self.extract_embeddings(model, data_loader, num_samples)

            if compress:
                compressed_data, comp_info = self.compress_embeddings(embeddings)
                data = compressed_data
                compression_info = comp_info
            else:
                data = embeddings
                compression_info = None

        else:
            raise ValueError(f"Unsupported representation type: {repr_type}. Use 'W' or 'E'.")

        package = {
            "type": repr_type,
            "data": data,
            "size_mb": self.estimate_size(
                repr_type, num_samples if repr_type == "E" else None, model=model
            ),
            "compressed": compress and repr_type == "E",
            "compression_info": compression_info,
        }

        return package

    def reset_prototypes(self):
        """Reset prototype clusters (useful for starting fresh)."""
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.num_prototypes,
            random_state=42,
            batch_size=32,
            max_iter=100,
        )
