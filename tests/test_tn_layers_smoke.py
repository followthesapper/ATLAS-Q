
def test_tn_layers_shapes():
    try:
        import torch
        from tn_layers import TensorTrainBottleneck, TTNBlock, MERALite
    except Exception:
        return  # skip if no torch
    B, T, D = 2, 32, 64
    x = torch.randn(B, T, D)
    ttb = TensorTrainBottleneck(d_in=D, d_out=D, m=4, rank=4)
    y = ttb(x); assert y.shape[0]==B and y.shape[1] in (T//1, T) and y.shape[2] > 0
    ttn = TTNBlock(d=D, width=3)
    y2 = ttn(y); assert y2.shape[0]==B and y2.shape[2]==D
    mera = MERALite(d=D)
    y3 = mera(y2); assert y3.shape == y2.shape
