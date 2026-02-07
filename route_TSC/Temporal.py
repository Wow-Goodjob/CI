import traci
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
import matplotlib.pyplot as plt

def collect_enhanced_data(sumo_cfg, total_steps=3600):
    traci.start(["sumo", "-c", sumo_cfg])
    edge_ids = [e for e in traci.edge.getIDList() if not e.startswith(':')]  # 排除内部路段
    num_edges = len(edge_ids)

    data = np.zeros((total_steps, num_edges * 2))

    for step in range(total_steps):
        traci.simulationStep()
        for i, edge in enumerate(edge_ids):
            veh_count = traci.edge.getLastStepVehicleNumber(edge)
            mean_speed = traci.edge.getLastStepMeanSpeed(edge)

            data[step, i * 2] = veh_count
            data[step, i * 2 + 1] = mean_speed
    traci.close()
    return data, edge_ids


raw_data, edge_list = collect_enhanced_data('./res/hangzhou/train.sumocfg')

INPUT_DIM = raw_data.shape[1]
HIDDEN_SIZE = 128
PRED_LEN = 10
LOOKBACK = 30


class MultiFeatureLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, pred_len, num_layers=2):
        super(MultiFeatureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, pred_len * input_dim)
        self.input_dim = input_dim
        self.pred_len = pred_len

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)

        last_hidden = hn[-1]
        prediction = self.fc(last_hidden)
        return prediction.view(-1, self.pred_len, self.input_dim)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiFeatureLSTM(INPUT_DIM, HIDDEN_SIZE, PRED_LEN).to(device)

def create_sequences(data,lookback=30,pred_len=10):
    xs,ys=[],[]
    for i in range(len(data)-lookback-pred_len):
        x=data[i:i+lookback]
        y=data[i+lookback:i+lookback+pred_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs),np.array(ys)

def prepare_datasets(raw_data,train_split=2400,lookback=30,pred_len=10,batch_size=64):
    train_raw=raw_data[:train_split]
    test_raw=raw_data[train_split:]

    scaler=MinMaxScaler(feature_range=(0,1))
    scaler.fit(train_raw)

    data_scaled=scaler.transform(raw_data)
    X_Train,Y_Train=create_sequences(data_scaled[:train_split])
    X_test,Y_test=create_sequences(data_scaled[train_split:])

    train_dataset=TensorDataset(torch.FloatTensor(X_Train),torch.FloatTensor(Y_Train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test))

    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader,test_loader,scaler

def train(model,train_loader,num_epochs=50,lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion=torch.nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=lr)
    loss_num=[]
    for epoch in range(num_epochs):
        model.train()
        total_loss=0

        for batch_x,batch_y in train_loader:
            batch_x,batch_y=batch_x.to(device),batch_y.to(device)
            outputs=model(batch_x)
            loss=criterion(outputs,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
        if (epoch+1)%10==0:
            avg_loss=total_loss/len(train_loader)
            print(f"Epoch[{epoch+1}],Loss:{avg_loss:.6f}")
        loss_num.append(total_loss)
    return model,loss_num

def predict_traffic(model,input_seq,scaler):
    model.eval()
    device=next(model.parameters()).device
    input_scaled=scaler.transform(input_seq)
    input_tensor=torch.FloatTensor(input_scaled).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor=model(input_tensor)
    output_scaled=output_tensor.cpu().numpy().squeeze(0)
    prediction_real=scaler.inverse_transform(output_scaled)
    return prediction_real

train_loader,test_loader,scaler=prepare_datasets(raw_data)
model,loss=train(model,train_loader)
plt.plot(loss)
def plot_long_term_performance(model,raw_data,scaler,test_start=2400,test_end=3500,edge_index=0):
    model.eval()
    device=next(model.parameters()).device

    predictions=[]
    truths=[]
    for i in range(test_start,test_end):
        input_seq=raw_data[i:i+30]
        pred_res=predict_traffic(model,input_seq,scaler)
        first_step_pred=pred_res[0,:]
        real_val=raw_data[i+30]
        predictions.append(first_step_pred)
        truths.append(real_val)
    predictions=np.array(predictions)
    truths=np.array(truths)
    col_idx=edge_index*2
    plt.figure(figsize=(14,5))
    plt.plot(truths[:,col_idx],color='blue',alpha=0.7)
    plt.plot(predictions[:,col_idx],color='red',linestyle='--',alpha=0.8)
    plt.legend()
    plt.show()
def plot_long_term_performance_speed(model,raw_data,scaler,test_start=2400,test_end=3500,edge_index=0):
    model.eval()
    device=next(model.parameters()).device

    predictions=[]
    truths=[]
    for i in range(test_start,test_end):
        input_seq=raw_data[i:i+30]
        pred_res=predict_traffic(model,input_seq,scaler)
        first_step_pred=pred_res[0,:]
        real_val=raw_data[i+30]
        speed_index=edge_index*2+1
        predictions.append(first_step_pred[speed_index])
        truths.append(real_val[speed_index])
    predictions=np.array(predictions)
    truths=np.array(truths)
    col_idx=edge_index*2
    plt.figure(figsize=(14,5))
    plt.plot(truths,color='blue',alpha=0.7)
    plt.plot(predictions,color='red',linestyle='--',alpha=0.8)
    plt.legend()
    plt.show()
plot_long_term_performance(model,raw_data,scaler)
plot_long_term_performance_speed(model,raw_data,scaler)