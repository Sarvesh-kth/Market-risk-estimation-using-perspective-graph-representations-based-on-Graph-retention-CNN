import torch
from rkm1 import RiskPredictionModel  
import os
import warnings
warnings.filterwarnings('ignore')
def load_model(model_path, in_features_dict, num_classes, num_heads, hidden_features):
    device = torch.device('cpu')
    model = RiskPredictionModel(in_features_dict, hidden_features, num_classes, num_heads).to(device)
    saved_state_dict = torch.load(model_path, map_location=device)
    model_state_dict = model.state_dict()

    for key in model_state_dict:
        if key in saved_state_dict:
            if saved_state_dict[key].shape == model_state_dict[key].shape:
                model_state_dict[key] = saved_state_dict[key]
            else:

                try:
                    resized_tensor = saved_state_dict[key].resize_(model_state_dict[key].shape)
                    model_state_dict[key] = resized_tensor
                    print(f"Resized {key} to match the model's expected shape.")
                except RuntimeError:
                    print(f"Failed to resize {key}; reinitializing to default.")
                    model_state_dict[key] = torch.randn(model_state_dict[key].shape) * 0.01  
        else:
            print(f"Missing key {key} in saved model; using default initialization.")

    model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    return model, device



def predict_risk(model, device, graph_path):
    graph = torch.load(graph_path, map_location=device)
    with torch.no_grad():
        outputs = model(graph)
        _, predicted_category = torch.max(outputs, dim=1)  
    return predicted_category.item()


file_path = 'riskmodel2_epoch_120.pth'  
graphs_directory = "modgraph/modified4"
in_features_dict = {'company': 1, 'year': 4, 'quarter': 11, 'month': 1, 'day': 14}
num_classes = 5
num_heads = 4

ticker = 'ABBV'  
graph_path = os.path.join(graphs_directory, f"{ticker}_modified.pt")

model, device = load_model(file_path, in_features_dict, num_classes, num_heads,32)
risk_category = predict_risk(model, device, graph_path)

classs = { 0 : ' Extremely Low Risk ', 1 : ' Low Risk ', 2: ' Moderate Risk ', 3: ' High Risk ', 4 : ' Extremely High Risk '}
print(f"Predicted Risk Category for {ticker}: {risk_category} : {classs[risk_category]}")
