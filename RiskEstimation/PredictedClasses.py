import os
import torch
import pandas as pd
import csv
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from rkm1 import RiskPredictionModel
import numpy as np

def load_model(model_path, in_features_dict, num_classes, num_heads):
    device = torch.device('cpu')
    model = RiskPredictionModel(in_features_dict, 32, num_classes, num_heads).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def predict_and_evaluate(model, device, graph_directory, data, results_path):
    loss_function = CrossEntropyLoss().to(device)
    predictions = []
    true_labels = []
    results = []

    with open(results_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Ticker', 'Predicted', 'Actual'])

        for filename in tqdm(os.listdir(graph_directory)):
            if filename.endswith('.pt'):
                ticker = filename[:-12]
                if ticker in data.index:
                    graph_path = os.path.join(graph_directory, filename)
                    graph = torch.load(graph_path, map_location=device)
                    outputs = model(graph)
                    _, predicted = torch.max(outputs.data, 1)
                    target = data.loc[ticker, 'target']
                    target_number = int(target[2])

                    predictions.append(predicted.item())
                    true_labels.append(target_number)
                    results.append([ticker, predicted.item(), target_number])


                    writer.writerow([ticker, predicted.item(), target_number])

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy, results

if __name__ == "__main__":
    model_path = 'riskmodel2_epoch_120.pth'
    graphs_directory = "modgraph/modified4"
    in_features_dict = {'company': 1, 'year': 4, 'quarter': 11, 'month': 1, 'day': 14}
    num_classes = 5
    num_heads = 4

    data_path = 'clustered_data.xlsx'
    data = pd.read_excel(data_path)
    data.set_index('Ticker', inplace=True)
    target_columns = ['rl0', 'rl1', 'rl2','rl3','rl4']
    data['target'] = data[target_columns].idxmax(axis=1)

    results_path = 'prediction_results.csv'
    model, device = load_model(model_path, in_features_dict, num_classes, num_heads)
    total_accuracy, results = predict_and_evaluate(model, device, graphs_directory, data, results_path)
    print(f"Total Model Accuracy: {total_accuracy:.4f}")
    print("Results saved to:", results_path)
