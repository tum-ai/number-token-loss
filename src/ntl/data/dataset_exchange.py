import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
from src.ntl.data.timefeatures import time_features


class Dataset_Custom(Dataset):
    def __init__(
        self,
        flag="train",
        size=None,
        features="S", # "M" or "S" depending if multivariate or univariate
        data_path=None,
        target="OT",
        scale=False,
        timeenc=0,
        freq="h",
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 96
            self.label_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Print data mean and std
        train_data = df_data[border1s[0] : border2s[0]]
        print("Data mean: ", train_data.mean())
        print("Data std: ", train_data.std())

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_end = s_end + self.label_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_end:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[s_end:r_end]
        
        # Create question and answer

        seq_x_str = [" ".join(map(str, row)) for row in seq_x]
        # pad numbers to same length
        seq_x_str = [f"{float(num):.2f}" for num in seq_x_str]
        seq_x_mark_str = ["".join(map(str, row)) for row in seq_x_mark]
        # question = ", ".join([f"{mark}: {seq}" for mark, seq in zip(seq_x_mark_str, seq_x_str)])
        question = " ".join([f"{seq}" for seq in seq_x_str])

        seq_y_str = [" ".join(map(str, row)) for row in seq_y]
        # pad numbers to same length
        seq_y_str = [f"{float(num):.2f}" for num in seq_y_str]
        seq_y_mark_str = ["".join(map(str, row)) for row in seq_y_mark]
        # answer = ", ".join([f"{mark}: {seq}" for mark, seq in zip(seq_y_mark_str, seq_y_str)])
        answer = " ".join([f"{seq}" for seq in seq_y_str])

        return {
            "question": question,
            "answer": answer}

    def __len__(self):
        return len(self.data_x) - (self.seq_len + self.label_len)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
