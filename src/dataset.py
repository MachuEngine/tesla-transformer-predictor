import torch
from torch.utils.data import DataLoader, Dataset
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터 로딩 및 전처리
def load_and_preprocess_data(ticker, start_date, end_date, features=['Close', 'Volume', 'Open', 'High', 'Low'], split_ratio=0.8):
    """
    yfinance를 사용하여 주가 데이터를 다운로드하고, 지정한 features를 선택 후 정규화 및 학습/테스트 분할을 수행합니다.

    data : 데이터프레임 df에서 features에 해당하는 열들을 선택한 후, numpy array로 변환한 값. 즉 2차원 크기의 넘파이 배열임
        [
          [120.15, 30000000, 119.50, 121.00, 119.10],
          [121.45, 35000000, 120.80, 122.50, 120.50],
          [119.90, 40000000, 120.00, 121.50, 118.80],
        ]
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    data = df[features].values  # shape: [num_samples, num_features]

    # data 확인
    print(f"Data Sample: {data[:5]}")
    # size 확인 
    print(f"Data Shape: {data.shape}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data) # -> 각 feature 별로 0~1 사이의 값으로 정규화
    split_idx = int(len(data_scaled) * split_ratio)
    train_data = data_scaled[:split_idx]
    test_data = data_scaled[split_idx:]

    # train_data, test_data 확인
    print(f"Train Data Shape: {train_data.shape}")
    print(f"Test Data Shape: {test_data.shape}")

    return train_data, test_data, scaler

# 2. Dataset 및 DataLoader 구성
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        src = self.data[idx:idx + self.seq_length]  # [seq_length, num_features]
        tgt = self.data[idx + self.seq_length]       # [num_features]
        return src, tgt

def create_dataloader(data, seq_length, batch_size, shuffle=True):
    dataset = TimeSeriesDataset(data, seq_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader