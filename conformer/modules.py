from torch import nn

from .layers import Swish, Transpose


class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        expand_dim = d_model * expansion_factor
        #print("d_model:",d_model)
        #print("expand_dim:",expand_dim)
        #self.model = nn.Sequential(
        self.layer_norm=nn.LayerNorm(d_model)#,
        self.lin1=nn.Linear(d_model, expand_dim)#,
        self.swi=Swish()#,
        self.drop=nn.Dropout(dropout)#,
        self.lin2=nn.Linear(expand_dim, d_model)#,
        self.drop2=nn.Dropout(dropout)#,
        #)
        #print("----FeedForwardModule is initialized------------------")

    def forward(self, x):
        #print("inp:",x.shape)

        #print("layer_norm:",self.layer_norm(x).shape)
        x=self.layer_norm(x)
        #print("lin1:",self.lin1(x).shape)
        x=self.lin1(x)
        #print("swi:",self.swi(x).shape)
        x=self.swi(x)
        #print("drop:",self.drop(x).shape)
        x=self.drop(x)
        #print("lin2:",self.lin2(x).shape)
        x=self.lin2(x)
        #print("drop2:",self.drop2(x).shape)
        x=self.drop2(x)
        return x#self.model(x)


class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size, dropout):
        assert (
            kernel_size % 2
        ), f"Expected `kernel_size` to be odd, but got {kernel_size}"
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(d_model),
            # pointwise conv is same as linear, but matmul is quicker
            # see https://stackoverflow.com/questions/55576314
            nn.Linear(d_model, d_model * 2),
            Transpose(1, 2),
            nn.GLU(dim=1),
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=d_model,
            ),
            nn.BatchNorm1d(d_model),
            Swish(),
            Transpose(1, 2),
            # same logic as above
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.model(x)

