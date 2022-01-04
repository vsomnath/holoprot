from holoprot.layers.mpn.wln import WLNConv, WLNResConv
from holoprot.layers.mpn.rnn import GRUConv, LSTMConv

WLNs = {'wln': WLNConv, 'wlnres': WLNResConv}
RNNs = {'gru': GRUConv, 'lstm': LSTMConv}

def mpn_layer_from_config(config, encoder):
    if encoder in WLNs:
        layer_class = WLNs.get(encoder)
        mpn_layer = layer_class(node_fdim=config['node_fdim'],
                                edge_fdim=config['edge_fdim'],
                                hsize=config['hsize'],
                                depth=config['depth'],
                                dropout=config['dropout_p'],
                                activation=config['activation'],
                                jk_pool=config.get("jk_pool", None))
    elif encoder in RNNs:
        raise NotImplementedError("RNN layers currently do not work with the codebase")
        layer_class = RNNs.get(encoder)
        mpn_layer = layer_class(node_fdim=config['node_fdim'],
                                edge_fdim=config['edge_fdim'],
                                hsize=config['hsize'],
                                depth=config['depth'],
                                dropout=config['dropout_p'],
                                rnn_agg=config.get("rnn_agg", None),
                                activation=config['activation'])
    elif encoder == 'gtrans':
        raise NotImplementedError()

    else:
        raise ValueError(f"Encoder {encoder} is not supported yet.")
    return mpn_layer
