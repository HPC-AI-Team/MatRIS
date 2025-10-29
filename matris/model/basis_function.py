from __future__ import annotations

import torch
from torch import Tensor, nn
import math
import numpy as np

class PolynomialEnvelope(nn.Module):
    """
        Polynomial cutoff function for distance
        Details are given in: https://doi.org/10.48550/arXiv.2106.08903
    """

    def __init__(self, cutoff: float = 6, envelope_exponent: float = 8) -> None:
        """Initialize the polynomial cutoff function.

        Args:
            cutoff (float): cutoff radius in graph construction
                Default = 6
            envelope_exponent (float): envelope exponent
                Default = 8
        """
        super().__init__()
        assert envelope_exponent > 0
        self.cutoff = cutoff
        self.p = float(envelope_exponent)
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, dist: Tensor) -> Tensor:
        """
        Args:
            dist (Tensor): radius distance tensor

        Returns:
            polynomial cutoff functions: decaying from 1 to 0
        """
        if self.p != 0:
            dist_scaled = dist / self.cutoff
            
            env_val = (
                1
                + self.a * dist_scaled ** self.p
                + self.b * dist_scaled ** (self.p + 1)
                + self.c * dist_scaled ** (self.p + 2)
            )
            
            return torch.where(dist_scaled < 1, env_val, torch.zeros_like(dist_scaled))
        return dist.new_ones(dist.shape)


@torch.jit.script
def SphericalHarmonics(lmax: int, x: torch.Tensor) -> torch.Tensor:
    
    sh_0_0 = torch.ones_like(x) * 0.5 * math.sqrt(1.0 / math.pi)
    if lmax == 0:
        return torch.stack(
            [
                sh_0_0,
            ],
            dim=-1,
        )

    sh_1_1 = math.sqrt(3.0 / (4.0 * math.pi)) * x
    if lmax == 1:
        return torch.stack([sh_0_0, sh_1_1], dim=-1)

    sh_2_2 = math.sqrt(5.0 / (16.0 * math.pi)) * (3.0 * x**2 - 1.0)
    if lmax == 2:
        return torch.stack([sh_0_0, sh_1_1, sh_2_2], dim=-1)

    sh_3_3 = math.sqrt(7.0 / (16.0 * math.pi)) * x * (5.0 * x**2 - 3.0)
    if lmax == 3:
        return torch.stack([sh_0_0, sh_1_1, sh_2_2, sh_3_3], dim=-1)

    raise ValueError("lmax <= 3")

class FourierExpansion(nn.Module):
    """
        Fourier expansion for three-body feature.
        Details are given in: https://doi.org/10.1038/s42256-023-00716-3
    """

    def __init__(self, max_f: int = 5, 
                 scale: float = np.pi, 
                 learnable: bool = False):
        """
        Args:
            max_f (int): the maximum frequency of the expansion. 
                Default = 5
            learnable (bool): whether to set the frequencies as learnable parameters. 
                Default = False
        """
        super().__init__()
        self.max_f = max_f
        self.scale = np.sqrt(scale)
        if learnable:
            self.frequencies = torch.nn.Parameter(
                data=torch.arange(1, max_f + 1, dtype=torch.float),
                requires_grad=True,
            )
        else:
            self.register_buffer("frequencies", torch.arange(1, max_f + 1, dtype=torch.float))

    def forward(self, feature: Tensor) -> Tensor:
        """Apply Fourier expansion to feature."""
        result = feature.new_zeros(feature.shape[0], 1 + 2 * self.max_f)
        result[:, 0] = 1 / torch.sqrt(torch.tensor([2]))
        tmp = torch.outer(feature, self.frequencies) 
        result[:, 1 : self.max_f + 1] = torch.sin(tmp)
        result[:, self.max_f + 1 :] = torch.cos(tmp)
        return result / self.scale


class BesselExpansion(torch.nn.Module):
    """
        Bessel expansion for pairwise feature
        Details are given in: https://arxiv.org/abs/2003.03123
    """
    def __init__(
        self,
        num_radial: int = 9,
        cutoff: float = 6,
        envelope_exponent: int = 8,
        learnable: bool = False,
    ):
        """
        Args:
            num_radial (int): number of radial basis.
                Default = 9
            cutoff (float):  cutoff distance.
                Default = 6
            envelope_exponent (int): envelope exponent of Envelope function.
                Default = 8
            learnable (bool): whether to set the frequencies as learnable parameters.
                Default = False
        """
        super().__init__()
        self.num_radial = num_radial
        self.inv_cutoff = 1 / cutoff
        self.norm_const = (2 * self.inv_cutoff) ** 0.5

        if learnable:
            self.frequencies = torch.nn.Parameter(
                data=torch.Tensor(
                    np.pi * torch.arange(1, self.num_radial + 1, dtype=torch.float)
                ), requires_grad=True,
            )
        else:
            self.register_buffer(
                "frequencies",
                np.pi * torch.arange(1, self.num_radial + 1, dtype=torch.float),
            )
        if envelope_exponent is not None:
            self.envelope_func = PolynomialEnvelope(
                cutoff=cutoff, envelope_exponent=envelope_exponent
            )
        else:
            self.envelope_func = None

    def forward(
        self, dist: Tensor, return_smooth_factor: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Apply Bessel expansion to feature.
        
        Args:
            dist (Tensor): tensor of distances. [nEdges]
            return_smooth_factor (bool): whether to return the smooth factor
                Default = False

        Returns:
            out (Tensor): tensor of Bessel basis [nEdges, num_radial]
            smooth_factor (Tensor): tensor of smooth factors [nEdges, 1]
        
        TODO: 'torch.script.jit' boosts performance.
        """
        dist = dist[:, None]  # [nEdges] -> [nEdges, 1]
        d_scaled = dist * self.inv_cutoff
        out = self.norm_const * torch.sin(self.frequencies * d_scaled) / dist
        if self.envelope_func is not None:
            smooth_factor = self.envelope_func(dist)
            out = smooth_factor * out
            if return_smooth_factor:
                return out, smooth_factor
        return out


class GaussianExpansion(nn.Module):
    """
        Gaussian expansion for pairwise feature.
        Code adapted from the repo: https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/models/uma/nn/radial.py#L42
    """
    def __init__(
        self,
        start: float = 0.0, 
        stop: float = 6.0, 
        num_gaussians: int = 7,
        basis_width_scalar: float = 2.0, 
    ):
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> Tensor:
        """Apply Bessel expansion to feature.
        
        Args:
            dist (Tensor): tensor of distances. [nEdges]

        Returns:
            out (Tensor): tensor of Bessel basis [nEdges, num_gaussians]
        
        TODO: 'torch.script.jit' boosts performance.
        """
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class SphericalExpansion(nn.Module):
    """
        Spherical expansion for three-body feature.
        Code adapted from the repo: https://github.com/microsoft/mattersim
    """
    def __init__(self, max_n = 4, max_l = 4, cutoff = 6):
        super().__init__()

        assert max_l <= 4, "lmax must be less than 5"
        assert max_n <= 4, "max_n must be less than 5"

        self.max_n = max_n
        self.max_l = max_l
        self.cutoff = cutoff

        # retrieve formulas
        self.register_buffer(
            "factor", torch.sqrt(torch.tensor(2.0 / (self.cutoff**3)))
        )
        self.coef = torch.zeros(4, 9, 4)
        self.coef[0, 0, :] = torch.tensor(
            [ 3.14159274101257, 6.28318548202515, 9.42477798461914, 12.5663709640503 ]
        )
        self.coef[1, :4, :] = torch.tensor(
            [
                [ -1.02446483277785, -1.00834335996107, -1.00419641763893, -1.00252381898662  ],
                [ 4.49340963363647,  7.7252516746521,   10.9041213989258,  14.0661935806274   ],
                [ 0.22799275301076,  0.130525632358311, 0.092093290316619, 0.0712718627992818 ],
                [ 4.49340963363647,  7.7252516746521,   10.9041213989258,  14.0661935806274   ], 
            ]
        )
        self.coef[2, :6, :] = torch.tensor(
            [
                [ -1.04807944170731,  -1.01861796359391,  -1.01002272174988,  -1.00628955560036  ],
                [ 5.76345920562744,   9.09501171112061,   12.322940826416,    15.5146026611328   ],
                [ 0.545547077361439,  0.335992298618515,  0.245888396928293,  0.194582402961821  ],
                [ 5.76345920562744,   9.09501171112061,   12.322940826416,    15.5146026611328   ],
                [ 0.0946561878721665, 0.0369424811413594, 0.0199537107571916, 0.0125418876146463 ],
                [ 5.76345920562744,   9.09501171112061,   12.322940826416,    15.5146026611328   ], 
            ]
        )
        self.coef[3, :8, :] = torch.tensor(
            [
                [ 1.06942831392075,   1.0292173312802,    1.01650804843248,   1.01069656069999    ], 
                [ 6.9879322052002,    10.4171180725098,   13.6980228424072,   16.9236221313477    ],
                [ 0.918235852195231,  0.592803493701152,  0.445250264272671,  0.358326327374518   ],
                [ 6.9879322052002,    10.4171180725098,   13.6980228424072,   16.9236221313477    ],
                [ 0.328507713452024,  0.142266673367543,  0.0812617757677838, 0.0529328657590962  ],
                [ 6.9879322052002,    10.4171180725098,   13.6980228424072,   16.9236221313477    ],
                [ 0.0470107184508114, 0.0136570088173405, 0.0059323726279831, 0.00312775039221944 ],
                [ 6.9879322052002,    10.4171180725098,   13.6980228424072,   16.9236221313477    ],
            ]
        )

    def forward(self, dist, three_body):
        dist = dist / self.cutoff
        rbfs = []

        for j in range(self.max_l):
            rbfs.append(torch.sin(self.coef[0, 0, j] * dist) / dist)

        if self.max_n > 1:
            for j in range(self.max_l):
                rbfs.append(
                    (
                        self.coef[1, 0, j]
                        * dist
                        * torch.cos(self.coef[1, 1, j] * dist) 
                        + self.coef[1, 2, j]
                        * torch.sin(self.coef[1, 3, j] * dist)
                    )
                    / dist**2
                )

            if self.max_n > 2:
                for j in range(self.max_l):
                    rbfs.append(
                        (
                            self.coef[2, 0, j]
                            * (dist**2)
                            * torch.sin(self.coef[2, 1, j] * dist)
                            - self.coef[2, 2, j]
                            * dist
                            * torch.cos(self.coef[2, 3, j] * dist) 
                            + self.coef[2, 4, j]
                            * torch.sin(self.coef[2, 5, j] * dist)
                        )
                        / (dist**3)
                    )

                if self.max_n > 3:
                    for j in range(self.max_l):
                        rbfs.append(
                            (
                                self.coef[3, 0, j]
                                * (dist**3)
                                * torch.cos(self.coef[3, 1, j] * dist)
                                - self.coef[3, 2, j]
                                * (dist**2)
                                * torch.sin(self.coef[3, 3, j] * dist)
                                - self.coef[3, 4, j]
                                * dist
                                * torch.cos(self.coef[3, 5, j] * dist)
                                + self.coef[3, 6, j]
                                * torch.sin(self.coef[3, 7, j] * dist)
                            )
                            / dist**4
                        )

        rbfs = torch.stack(rbfs, dim=-1)
        rbfs = rbfs * self.factor
        
        cbfs = SphericalHarmonics(self.max_l - 1, torch.cos(three_body))

        cbfs = cbfs.repeat_interleave(self.max_n, dim=1)
        
        return rbfs * cbfs


class SinusoidalTimeExpansion(nn.Module):
    """ Encode time """
    
    def __init__(self, dim: int = 128):
        """
        Args:
            dim (int): the embedding size of Time.
        """
        super().__init__()

        self.dim = dim
        half_dim = self.dim // 2
        # Inverse frequencies for sinusoidal embedding
        self.register_buffer(
            "inv_freq",
            torch.exp(torch.arange(half_dim, dtype=torch.float32) * (-math.log(10000) / (half_dim - 1)))
        )
    
    def forward(self, feature):
        shape = feature.shape
        feature = feature.view(-1).to(torch.float32)
        sinusoid_in = torch.ger(feature, self.inv_freq)
        time_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        # Restore original shape with embedding dimension
        time_emb = time_emb.view(*shape, self.dim)

        return time_emb