import torch
from utils import load_data
from DAGAD import DAGAD_GCN, DAGAD_GAT, Validation
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    job_que = ["Flickr_GCN", "Flickr_GAT", "ACM_GCN", "ACM_GAT", "BlogCatalog_GCN", "BlogCatalog_GAT"]
    wds = [0.0005, 0.0002, 0.0002, 0.0004, 0.001, 0.0009]
    for i, job in enumerate(job_que):
        dataset = job.split("_")[0]
        data = load_data(dataset)
        data = data.to(device)
        input_dim = data.x.shape[1]
        mod_type = job.split("_")[1]
        if mod_type == 'GCN':
            model = DAGAD_GCN(input_dim, 64, 32, 2, device)
        if mod_type == 'GAT':
            model = DAGAD_GAT(input_dim, 64, 32, 8, 2, device)
        model = model.to(device)
        print(f"DAGAD-{mod_type} on {dataset}.")
        result = Validation(model, data, 200, 0.005, 1.5, 0.5, 0.7, wds[i])
        print(result)
