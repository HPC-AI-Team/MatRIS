from __future__ import annotations

import torch
from torch import Tensor, nn
from typing import Any, Dict
from .functions import (
    MLP,
    GatedMLP,
    aggregate,
    get_normalization,
    Dimwise_softmax,
)
from torch.utils.checkpoint import checkpoint

THRESHOLD_VALUE = 60000 # Safe value for MatRIS-10M (A100-80GB)

class Graph_Attention_Layer(nn.Module):
    
    def __init__(
        self,
        node_feat_dim: int = 128,
        edge_feat_dim: int = 128,
        hidden_dim: int = 128,
        use_bias: bool = False,
        dropout: float = 0.0,
        mlp_type: str = "GateMLP", # MLP, GateMLP
        activation_type: str = "silu",
        norm_type: str = "layer",
        use_fp16: bool = False, 
    ):
        super().__init__()
        
        self.source_weight_linear = nn.Linear(
            in_features = edge_feat_dim, out_features = edge_feat_dim, bias = False
        )
        self.target_weight_linear = nn.Linear(
            in_features = edge_feat_dim, out_features = edge_feat_dim, bias = False
        )
        if mlp_type.lower() == "mlp":
            self.node_nonlinear_update = nn.Sequential(
                MLP(
                    input_dim=edge_feat_dim * 2 + node_feat_dim,
                    hidden_dim=hidden_dim,
                    output_dim=node_feat_dim,
                    dropout=dropout,
                    bias=use_bias,
                    activation=activation_type,
                ),
                get_normalization(name=norm_type, dim=node_feat_dim) 
            )
            self.edge_nonlinear_update = nn.Sequential(
                MLP(
                    input_dim=node_feat_dim * 2 + edge_feat_dim,
                    hidden_dim=hidden_dim,
                    output_dim=edge_feat_dim,
                    dropout=dropout,
                    bias=use_bias,
                    activation=activation_type,
                    use_fp16=use_fp16,
                ),
                get_normalization(name=norm_type, dim=edge_feat_dim) 
            )
        elif mlp_type.lower() == "gatemlp":
            self.node_nonlinear_update = GatedMLP(
                input_dim=edge_feat_dim * 2 + node_feat_dim,
                hidden_dim=hidden_dim,
                output_dim=node_feat_dim,
                norm_type=norm_type,
                dropout=dropout,
                activation=activation_type,
            )
            self.edge_nonlinear_update = GatedMLP(
                input_dim=node_feat_dim * 2 + edge_feat_dim,
                hidden_dim=hidden_dim,
                output_dim=edge_feat_dim,
                norm_type=norm_type,
                dropout=dropout,
                activation=activation_type,
                use_fp16=use_fp16,
            )
        else:
            raise NotImplementedError

        self.node_res_weight = torch.nn.Parameter(torch.ones(1, node_feat_dim), requires_grad=True)
        self.edge_res_weight = torch.nn.Parameter(torch.ones(1, edge_feat_dim), requires_grad=True)
    
    def forward(self, 
        node_feat: Tensor, 
        edge_feat: Tensor, 
        graph: Dict, # atom graph or line graph
        directed2undirected: Tensor = None,
    ): 
        source_node_index = graph['source_index']
        target_node_index = graph['target_index']
        # gather
        source_node_feat = torch.index_select(node_feat, 0, source_node_index)
        target_node_feat = torch.index_select(node_feat, 0, target_node_index)
        if directed2undirected is not None:
            # Atom Graph Update
            edge_feat_0 = torch.index_select(edge_feat, 0, directed2undirected) # [edge, dim] -> [2*edge, dim]
        else:
            # Line Graph Update
            edge_feat_0 = edge_feat

        #======= combine feature =======
        attn_edge_feat = torch.cat([edge_feat_0, target_node_feat, source_node_feat], dim=1)
        attn_edge_feat = self.edge_nonlinear_update(attn_edge_feat)

        # ======= update atom feature ======= 
        source_alpha_0 = self.source_weight_linear(edge_feat_0)
        target_alpha_0 = self.target_weight_linear(edge_feat_0)
        
        # Softmax
        num_segment = None #torch.unique(source_node_index).numel()
        source_alpha = Dimwise_softmax(source_alpha_0, source_node_index, num_segment)
        target_alpha = Dimwise_softmax(target_alpha_0, target_node_index, num_segment)
        
        source_weight = source_alpha * attn_edge_feat # refer to sa_{ij} * e'_{ij} in MatRIS paper
        target_weight = target_alpha * attn_edge_feat # refer to ta_{ij} * e'_{ij} in MatRIS paper
        
        if directed2undirected is not None:
            attn_edge_feat = aggregate(data=attn_edge_feat, segment=directed2undirected, bin_count=None, average=True, num_segment=None) #[2*edge, dim] -> [edge, dim]
        # Compute Attention output
        attn_source_feat = aggregate(data=source_weight, 
                                     segment=source_node_index, 
                                     bin_count=graph['source_bincount'],#bincount_source, 
                                     average=False, 
                                     num_segment=len(node_feat)) 

        attn_target_feat = aggregate(data=target_weight, 
                                     segment=target_node_index, 
                                     bin_count=graph['target_bincount'],#bincount_target, 
                                     average=False, 
                                     num_segment=len(node_feat)) 

        fusion_node_feat = torch.cat([node_feat, attn_target_feat, attn_source_feat], dim=1)
        attn_node_feat = self.node_nonlinear_update(fusion_node_feat)
        
        # Resdual
        attn_node_feat = attn_node_feat + self.node_res_weight * node_feat
        attn_edge_feat = attn_edge_feat + self.edge_res_weight * edge_feat

        return attn_node_feat, attn_edge_feat


class Refinement(nn.Module):
    
    def __init__(
        self,
        node_feat_dim: int = 128,
        edge_feat_dim: int = 128,
        hidden_dim: int = 128,
        num_basis: int = 7,
        dropout: float = 0.0,
        mlp_type: str = "GateMLP",    
        activation_type: str = "silu",
        norm_type: str = "layer",
        use_bias: bool = False,
        graph_type: Literal["atom graph", "line graph"] = "atom graph",
        atom_feat_dim: int = 128,
        use_smoothed_for_delta_edge: bool = False,
        use_fp16: bool = False, 
    ):
        super().__init__()
        self.graph_type = graph_type
        self.use_smoothed_for_delta_edge = use_smoothed_for_delta_edge
        
        if graph_type == "atom graph":
            input_dim = 2 * node_feat_dim + edge_feat_dim
        else:
            input_dim = atom_feat_dim + 2 * node_feat_dim + edge_feat_dim 
        if mlp_type.lower() == "mlp":
            self.edge_nonlinear_update = nn.Sequential(
                MLP(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=edge_feat_dim,
                    dropout=dropout,
                    bias=use_bias,
                    activation=activation_type,
                    use_fp16=use_fp16,
                ),
                get_normalization(name=norm_type, dim=edge_feat_dim)
            )
        elif mlp_type.lower() == "gatemlp":
            self.edge_nonlinear_update = GatedMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=edge_feat_dim,
                dropout=dropout,
                norm_type=norm_type,
                activation=activation_type,
                use_fp16=use_fp16,
            )
        else:
            raise NotImplementedError
        
        self.node_FFN = MLP(
            input_dim=edge_feat_dim,
            hidden_dim=node_feat_dim,
            output_dim=node_feat_dim,
            bias=use_bias,
        )
        self.edge_FFN = MLP(
            input_dim=edge_feat_dim,
            hidden_dim=edge_feat_dim,
            output_dim=edge_feat_dim,
            bias=use_bias,
            use_fp16=use_fp16,
        )
        self.learnable_envelope = nn.Linear(
            in_features = num_basis, out_features = edge_feat_dim, bias = False
        )
        
        self.node_res_weight = torch.nn.Parameter(torch.ones(1, node_feat_dim), requires_grad=True)
        self.edge_res_weight = torch.nn.Parameter(torch.ones(1, edge_feat_dim), requires_grad=True)

    def forward(
        self,
        node_feat: Tensor,
        edge_feat: Tensor,
        smooth_weight: Tensor,
        graph: Dict,
        directed2undirected: Tensor = None,
        atom_feat: Tensor = None, # Line graph
    ) -> Tensor:
        # Gather
        # when graph=="line graph", make sure atom_deat is not None.
        is_atom_graph = (self.graph_type == "atom graph")
        
        if is_atom_graph: 
            edge_feat_0 = torch.index_select(edge_feat, 0, directed2undirected) 
        else:
            edge_feat_0 = edge_feat

        source_node_feat = torch.index_select(node_feat, 0, graph['source_index'])
        target_node_feat = torch.index_select(node_feat, 0, graph['target_index'])
        # Envelope 
        if is_atom_graph:
            smooth_weight = torch.index_select(smooth_weight, 0, directed2undirected)
            smooth_weight = self.learnable_envelope(smooth_weight)
            # Fusion feature
            refine_fusion_feat = torch.cat([edge_feat_0, target_node_feat, source_node_feat], dim=1) 
        else:
            base_envelope = self.learnable_envelope(smooth_weight) 
            base_weights_i = torch.index_select(base_envelope, 0, graph['source_index'])
            base_weights_j = torch.index_select(base_envelope, 0, graph['target_index'])
            smooth_weight = base_weights_i * base_weights_j
            # Fusion feature
            three_body_atom_feat = torch.index_select(atom_feat, 0, graph['atom_list'])
            refine_fusion_feat = torch.cat([edge_feat_0, three_body_atom_feat, target_node_feat, source_node_feat], dim=1) 
        
        # Nonlinear            
        refine_fusion_feat_nonlinear = self.edge_nonlinear_update(refine_fusion_feat)
        refine_fusion_feat_smooth = refine_fusion_feat_nonlinear * smooth_weight 
         
        refine_node_feas = aggregate(refine_fusion_feat_smooth, 
                                     graph['target_index'], 
                                     graph['target_bincount'],
                                     average=False, 
                                     num_segment=len(node_feat))

        input2edgeFFN = (
            refine_fusion_feat_smooth
            if is_atom_graph and self.use_smoothed_for_delta_edge
            else refine_fusion_feat_nonlinear
        )
        
        delta_node_feat = self.node_FFN(refine_node_feas)
        delta_edge_feat = self.edge_FFN(input2edgeFFN)
        
        if is_atom_graph:  
            delta_edge_feat = aggregate(data=delta_edge_feat, segment=directed2undirected, bin_count=None, average=True, num_segment=None) # [2*edge, dim] -> [edge, dim]

        update_node_feat = delta_node_feat + self.node_res_weight * node_feat
        update_edge_feat = delta_edge_feat + self.edge_res_weight * edge_feat
        
        return update_node_feat, update_edge_feat


class Interaction_Block(nn.Module):
    """
    Interaction Block for MatRIS that processes both atom graphs and line graphs.
    
    This block performs attention-based message passing and refinement on two hierarchical graph structures:
    1. Atom graph: Nodes represent atoms, edges represent bonds
    2. Line graph: Nodes represent bonds, edges represent three-body interactions (angles)
    
    Attributes:
        attn_block_atom_graph (Graph_Attention_Layer): Attention layer for atom graph
        attn_block_line_graph (Graph_Attention_Layer): Attention layer for line graph
        refine_block_atom_graph (Refinement): Refinement layer for atom graph  
        refine_block_line_graph (Refinement): Refinement layer for line graph
    """
    
    def __init__(self,
                 node_feat_dim: int = 128,
                 edge_feat_dim: int = 128,
                 three_body_feat_dim: int = 128,
                 num_radial: int = 7,
                 num_angular: int = 7,
                 dropout: float = 0.0, 
                 use_bias: bool = False,
                 use_smoothed_for_delta_edge: bool = False,
                 mlp_type: str = "GateMLP",
                 norm_type: str = "layer",
                 activation_type: str = "silu",
                 ):
        """
        Initialize the Interaction Block.

        Args:
            node_feat_dim (int): Dimension of node features (atom features)
            edge_feat_dim (int): Dimension of edge features (bond features)  
            three_body_feat_dim (int): Dimension of three-body features (angle features)
            mlp_type (str): Type of MLP to use in the layers
            norm_type (str): Type of normalization to apply
            activation_type (str): Type of activation function to use
        """
        super().__init__()
        
        self.attn_block_atom_graph = Graph_Attention_Layer(
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                hidden_dim=node_feat_dim,
                use_bias=use_bias,
                mlp_type=mlp_type,
                norm_type=norm_type,
                activation_type=activation_type,
            )

        self.attn_block_line_graph = Graph_Attention_Layer(
                node_feat_dim=edge_feat_dim,
                edge_feat_dim=three_body_feat_dim,
                hidden_dim=edge_feat_dim,
                use_bias=use_bias,
                mlp_type=mlp_type,
                norm_type=norm_type,
                activation_type=activation_type,
                use_fp16=False,
            )
        
        self.refine_block_atom_graph = Refinement(
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                hidden_dim=node_feat_dim,  
                num_basis=num_radial,      
                dropout=dropout,            
                activation_type=activation_type,
                norm_type=norm_type,
                use_bias=use_bias,
                mlp_type=mlp_type,
                graph_type="atom graph",
                use_smoothed_for_delta_edge=use_smoothed_for_delta_edge,
            )
        
        self.refine_block_line_graph = Refinement(
                node_feat_dim=edge_feat_dim,
                edge_feat_dim=three_body_feat_dim,
                hidden_dim=edge_feat_dim,  
                num_basis=num_angular,     
                dropout=dropout,          
                activation_type=activation_type,
                norm_type=norm_type,
                use_bias=use_bias,
                mlp_type=mlp_type,
                graph_type="line graph",
                atom_feat_dim=node_feat_dim,
                use_fp16=False, 
            )
    
    def forward(
        self,
        batch_graph: Dict,
        node_feat: Tensor, 
        edge_feat: Tensor, 
        threebody_feat: Tensor | None,
        smooth_weight: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass of the Interaction Block.
        
        Args:
            batch_graph: Graph object containing:
                - atom_graph_dict: Atom graph structure
                - line_graph_dict: Bond graph (line graph) structure  
                - directed2undirected: Mapping from directed to undirected edges
                - bond_bases_bg: Smooth weights for bond graph
                - bond_bases_ag: Smooth weights for atom graph
            node_feat (Tensor): Node features [num_atoms, node_feat_dim]
            edge_feat (Tensor): Edge features [num_bonds, edge_feat_dim] 
            threebody_feat (Tensor): Three-body features [num_angles, three_body_feat_dim] or None
            bincount_atom_graph (Dict): Bincount information for atom graph
            bincount_line_graph (Dict): Bincount information for line graph
        """
        # Initialize variables to handle both cases (with and without threebody features)
        attn_edge_feat = edge_feat 
        attn_threebody_feat = threebody_feat
        update_edge_feat = edge_feat
        update_threebody_feat = threebody_feat 
        use_checkpoint = (
            isinstance(threebody_feat, torch.Tensor)
            and threebody_feat.shape[0] > THRESHOLD_VALUE
        )

        # Process line graph (bond graph) with attention if threebody features exist
        if threebody_feat is not None: 
            attn_edge_feat, attn_threebody_feat = self.wrapper_attn_layer(
                attn_layer=self.attn_block_line_graph,
                node_feat=edge_feat,
                edge_feat=threebody_feat,
                graph=batch_graph['line_graph_dict'],
                use_checkpoint=use_checkpoint, 
            )

        # Process atom graph with attention
        attn_node_feat, attn_edge_feat = self.wrapper_attn_layer(
            attn_layer=self.attn_block_atom_graph,
            node_feat=node_feat, 
            edge_feat=attn_edge_feat, 
            graph=batch_graph['atom_graph_dict'], 
            directed2undirected=batch_graph['directed2undirected'],
        ) 
        
        # Refine line graph features if threebody features exist
        if threebody_feat is not None:
            update_edge_feat, update_threebody_feat = self.wrapper_refine_layer(
                refine_layer=self.refine_block_line_graph,
                node_feat=attn_edge_feat,
                edge_feat=attn_threebody_feat,
                smooth_weight=smooth_weight['line graph'],
                graph=batch_graph['line_graph_dict'],
                atom_feat=attn_node_feat,
                use_checkpoint=use_checkpoint,
            )
        
        # Refine atom graph features
        update_node_feat, update_edge_feat = self.wrapper_refine_layer(
            refine_layer=self.refine_block_atom_graph,
            node_feat=attn_node_feat,
            edge_feat=update_edge_feat,
            smooth_weight=smooth_weight['atom graph'],
            graph=batch_graph['atom_graph_dict'],
            directed2undirected=batch_graph['directed2undirected'],
        )
        
        return update_node_feat, update_edge_feat, update_threebody_feat
    
    def wrapper_attn_layer(self,
                            attn_layer: nn.Module,
                            node_feat: Tensor, 
                            edge_feat: Tensor, 
                            graph: Dict,
                            directed2undirected: Tensor = None,
                            use_checkpoint: bool = False,
                       ):
        if use_checkpoint:
            attn_node_feat, attn_edge_feat = checkpoint(
                attn_layer,
                node_feat, 
                edge_feat, 
                graph,
                directed2undirected,
                use_reentrant=False,
            ) 
        else:
            attn_node_feat, attn_edge_feat = attn_layer(
                node_feat=node_feat, 
                edge_feat=edge_feat, 
                graph=graph,
                directed2undirected=directed2undirected,
            )
        
        return attn_node_feat, attn_edge_feat 
        
    def wrapper_refine_layer(self, 
                            refine_layer: nn.Module,
                            node_feat: Tensor,
                            edge_feat: Tensor,
                            smooth_weight: Tensor,
                            graph: Dict,
                            directed2undirected: Tensor = None,
                            atom_feat: Tensor = None,
                            use_checkpoint: bool = False,
                        ):
        if use_checkpoint:
            update_node_feat, update_edge_feat = checkpoint(
                refine_layer,
                node_feat,
                edge_feat,
                smooth_weight,
                graph,
                directed2undirected,
                atom_feat,
                use_reentrant=False,
            )
        else:
            update_node_feat, update_edge_feat = refine_layer(
                    node_feat=node_feat,
                    edge_feat=edge_feat,
                    smooth_weight=smooth_weight,
                    graph=graph,
                    directed2undirected=directed2undirected,
                    atom_feat=atom_feat,
                )
        return update_node_feat, update_edge_feat 
        
    