# Data

## Source

The paper uses data from a Super-App company in Latin America that runs a delivery app and offers
credit cards. The data cover nine months. Delivery-app usage defines monthly snapshots of the
customer network of each of three cities, and credit-card usage defines the node features. A node
is labelled an influencer when the customer referred another customer who then obtained a credit
card. The paper describes the node features in Appendix A (Table 6), the edge types in Table 2 and
the network characteristics in Appendix B (Figure 7).

## Licence and access

The data provider is anonymous, and the paper withholds the names of the cities for privacy. The
data are not distributed with this repository, and neither the repository nor the paper gives a
licence or an access route for them. Do not commit data files to this repository.

The script that built the network files from the raw data (`make_data.py`, named in an earlier
version of the README) is not in the repository.

## Files the code expects

`train.py` reads these paths relative to the working directory, which must be the repository root.
`<data_name>` is the value of `--data_name` (default `TGN_paper`), one per city.

| Path | Content, as the code uses it |
|---|---|
| `data/time_map.pkl` | Pickled dictionary keyed by month (`pandas.Timestamp`). |
| `data_pay/<data_name>_<type>.pt` | Dictionary saved with `torch.save`: one graph per month, keyed by month index, each with `x` (node features), `edge_index`, `edge_attr` (edge features) and `y` (influencer labels). `<type>` is `GAT` when `--GNN` is `GAT`, `GIN` or `SAGE`, `page_rank` with `--page_rank`, and `GCN` otherwise. |
| `data_folder/<data_name>/financial_for_profit_<data_name>.csv` | Customer profit, with the columns `START_DATE`, `APPLICATION_USER_ID` and `PROFIT`. Used only for the profit output. |
| `data_folder/<data_name>/inverse_node_map.pkl` | Pickled map from node index to customer. Used only for the profit output. |
| `data_folder/<data_name>/dict_v.pkl` | Pickled map, by month, from a customer to a list of customer IDs whose profit the profit output adds up. Used only for the profit output. |

The descriptions in the table are read from `train.py` and `utils/utils.py`. The repository does not
document the file formats further.
