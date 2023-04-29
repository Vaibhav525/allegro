from typing import List, Optional
import math

import torch
from torch import fx

from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.util.codegen import CodeGenMixin
from e3nn.math import normalize2mom

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.nn.nonlinearities import ShiftedSoftPlus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@compile_mode("script")
class ScalarMLPGAT(GraphModuleMixin, torch.nn.Module):
    """Apply an MLP to some scalar field."""

    field: str
    out_field: str

    def __init__(
        self,
        mlp_latent_dimensions: List[int],
        mlp_output_dimension: Optional[int],
        mlp_nonlinearity: Optional[str] = "silu",
        mlp_initialization: str = "uniform",
        mlp_dropout_p: float = 0.0,
        mlp_batchnorm: bool = False,
        field: str = AtomicDataDict.NODE_FEATURES_KEY,
        out_field: Optional[str] = None,
        irreps_in=None,
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field if out_field is not None else field
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[self.field],
        )
        assert len(self.irreps_in[self.field]) == 1
        assert self.irreps_in[self.field][0].ir == (0, 1)  # scalars
        in_dim = self.irreps_in[self.field][0].mul
        
        self.gat_model = GATModel2fast(1024,1024)#.to(self.device)
        self._module = ScalarMLPFunction(
            mlp_input_dimension=in_dim,
            mlp_latent_dimensions=mlp_latent_dimensions,
            mlp_output_dimension=mlp_output_dimension,
            mlp_nonlinearity=mlp_nonlinearity,
            mlp_initialization=mlp_initialization,
            mlp_dropout_p=mlp_dropout_p,
            mlp_batchnorm=mlp_batchnorm,
        )
        self.irreps_out[self.out_field] = o3.Irreps(
            [(self._module.out_features, (0, 1))]
        )
        

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        #print("gat_applied")
        data["edge_features"]  = self.gat_model(data["edge_features"], data["edge_index"])
        data[self.out_field] = self._module(data[self.field]) # self.field is  #edge_features
        return data



# import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATModel, self).__init__()
        self.gat = GATConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Args:
            x: graph_dict["edge_embedding"]
            edge_index: graph_dict["edge_index"]

        Returns: new_node_features after attention

        """
        # Transform the original graph to the new graph
        artificial_edge_index = []
        #edge_index.shape -> shape torch.Size([2, 1808]), has both (a,b) and (b,a) edge for an link b/q a and b
        for i in range(edge_index.shape[1]): # 1808
            src, tgt = edge_index[:, i] # take to nodes, src is top row of index, tgt is lower row
            artificial_edge_index.append([src, i])
            artificial_edge_index.append([tgt, i])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        artificial_edge_index = torch.tensor(artificial_edge_index, dtype=torch.long).t().contiguous().to(device)#torch.Size([2, 3616])
        # artificial_edge_index -> shape torch.Size([2, 3616])
        
        
        artificial_node_features = x # torch.Size([1808, 8]) # node embedding of new graph (edge embedding of old graph)
        artificial_node_features_after_attention = self.gat(artificial_node_features, artificial_edge_index)
        #new_node_features = F.relu(artificial_node_features_after_attention)
        
        
        edge_features = artificial_node_features_after_attention# Reconstruct the original graph using the new graph's node embeddings as edge embeddings
        return edge_features # torch.Size([1808, 8])





class GATModel2(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATModel2, self).__init__()
        self.gat = GATConv(in_channels, out_channels)
        

    def forward(self, x, edge_index):
        """
        Args:
            x: graph_dict["edge_embedding"]
            edge_index: graph_dict["edge_index"]

        Returns: new_node_features after attention

        """
        num_edges = edge_index.shape[1] # 2,1880
        edge_connections = set()

        for i in range(num_edges):
            for j in range(i+1, num_edges):
                if (edge_index[0, i] == edge_index[0, j]) or (edge_index[0, i] == edge_index[1, j]) or \
                (edge_index[1, i] == edge_index[0, j]) or (edge_index[1, i] == edge_index[1, j]):
                    edge_connections.add((i, j))
                    edge_connections.add((j, i))
        
        # ab edge --> ac 

        #new_edge_index = torch.tensor(list(edge_connections), dtype=torch.long).t().contiguous().to(device)
        artificial_edge_index = torch.tensor(list(edge_connections), dtype=torch.long).t().contiguous().to(device)
        #artificial_edge_index = torch.tensor(artificial_edge_index, dtype=torch.long).t().contiguous().to(device)#torch.Size([2, 3616])
        # artificial_edge_index -> shape torch.Size([2, 3616])
        
        
        artificial_node_features = x # torch.Size([1808, 8]) # node embedding of new graph (edge embedding of old graph)
        artificial_node_features_after_attention = self.gat(artificial_node_features, artificial_edge_index)
        #new_node_features = F.relu(artificial_node_features_after_attention)
        
        
        edge_features = artificial_node_features_after_attention# Reconstruct the original graph using the new graph's node embeddings as edge embeddings
        return edge_features # torch.Size([1808, 8])
    
class GATModel2fast(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATModel2fast, self).__init__()
        self.gat = GATConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Args:
            x: graph_dict["edge_embedding"]
            edge_index: graph_dict["edge_index"]

        Returns: new_node_features after attention

        """
        num_edges = edge_index.shape[1]

        # Remove self-loops from the edge_index
        self_loop_mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, self_loop_mask]
        num_edges = edge_index.shape[1]

        # Create a boolean tensor with shape [num_edges, num_edges]
        # Each element at position (i, j) is True if edge i (n_ab) and edge j (n_cd) share the same 'a' (first node)
        bool_tensor = (edge_index[0, :].view(-1, 1) == edge_index[0, :].view(1, -1))

        # Set the diagonal elements of bool_tensor to False
        bool_tensor[torch.eye(num_edges, dtype=torch.bool)] = False

        # Create a mask to extract the upper triangular part (excluding the diagonal)
        mask = torch.triu(torch.ones_like(bool_tensor, dtype=torch.bool), diagonal=1)

        artificial_edge_index = torch.nonzero(bool_tensor & mask, as_tuple=False).T.contiguous().to(device)#torch.Size([2, 3616])
        # artificial_edge_index -> shape torch.Size([2, 3616])
        
        
        artificial_node_features = x # torch.Size([1808, 8]) # node embedding of new graph (edge embedding of old graph)
        artificial_node_features_after_attention = self.gat(artificial_node_features, artificial_edge_index)
        #new_node_features = F.relu(artificial_node_features_after_attention)
        
        
        edge_features = artificial_node_features_after_attention# Reconstruct the original graph using the new graph's node embeddings as edge embeddings
        return edge_features 

class ScalarMLPFunction(CodeGenMixin, torch.nn.Module):
    """Module implementing an MLP according to provided options."""

    in_features: int
    out_features: int

    def __init__(
        self,
        mlp_input_dimension: Optional[int],
        mlp_latent_dimensions: List[int],
        mlp_output_dimension: Optional[int],
        mlp_nonlinearity: Optional[str] = "silu",
        mlp_initialization: str = "normal",
        mlp_dropout_p: float = 0.0,
        mlp_batchnorm: bool = False,
    ):
        super().__init__()
        nonlinearity = {
            None: None,
            "silu": torch.nn.functional.silu,
            "ssp": ShiftedSoftPlus,
        }[mlp_nonlinearity]
        if nonlinearity is not None:
            nonlin_const = normalize2mom(nonlinearity).cst
        else:
            nonlin_const = 1.0

        dimensions = (
            ([mlp_input_dimension] if mlp_input_dimension is not None else [])
            + mlp_latent_dimensions
            + ([mlp_output_dimension] if mlp_output_dimension is not None else [])
        )
        assert len(dimensions) >= 2  # Must have input and output dim
        num_layers = len(dimensions) - 1

        self.in_features = dimensions[0]
        self.out_features = dimensions[-1]

        # Code
        params = {}
        graph = fx.Graph()
        tracer = fx.proxy.GraphAppendingTracer(graph)

        def Proxy(n):
            return fx.Proxy(n, tracer=tracer)

        features = Proxy(graph.placeholder("x"))
        norm_from_last: float = 1.0

        base = torch.nn.Module()

        for layer, (h_in, h_out) in enumerate(zip(dimensions, dimensions[1:])):
            # do dropout
            if mlp_dropout_p > 0:
                # only dropout if it will do something
                # dropout before linear projection- https://stats.stackexchange.com/a/245137
                features = Proxy(graph.call_module("_dropout", (features.node,)))

            # make weights
            w = torch.empty(h_in, h_out)

            if mlp_initialization == "normal":
                w.normal_()
            elif mlp_initialization == "uniform":
                # these values give < x^2 > = 1
                w.uniform_(-math.sqrt(3), math.sqrt(3))
            elif mlp_initialization == "orthogonal":
                # this rescaling gives < x^2 > = 1
                torch.nn.init.orthogonal_(w, gain=math.sqrt(max(w.shape)))
            else:
                raise NotImplementedError(
                    f"Invalid mlp_initialization {mlp_initialization}"
                )

            # generate code
            params[f"_weight_{layer}"] = w
            w = Proxy(graph.get_attr(f"_weight_{layer}"))
            w = w * (
                norm_from_last / math.sqrt(float(h_in))
            )  # include any nonlinearity normalization from previous layers
            features = torch.matmul(features, w)

            if mlp_batchnorm:
                # if we call batchnorm, do it after the nonlinearity
                features = Proxy(graph.call_module(f"_bn_{layer}", (features.node,)))
                setattr(base, f"_bn_{layer}", torch.nn.BatchNorm1d(h_out))

            # generate nonlinearity code
            if nonlinearity is not None and layer < num_layers - 1:
                features = nonlinearity(features)
                # add the normalization const in next layer
                norm_from_last = nonlin_const

        graph.output(features.node)

        for pname, p in params.items():
            setattr(base, pname, torch.nn.Parameter(p))

        if mlp_dropout_p > 0:
            # with normal dropout everything blows up
            base._dropout = torch.nn.AlphaDropout(p=mlp_dropout_p)

        self._codegen_register({"_forward": fx.GraphModule(base, graph)})

    def forward(self, x):
        return self._forward(x)
