import numpy as np
from sklearn.metrics import roc_auc_score

class MetricUpdater:
    def __init__(self):
        self.f1 = [0, 0, 0]
        self.auc_pred = []
        self.auc_label = []
        self.auc = 0
        self.counter = 0

    def update_f1(self, predicts, label):
        # argmax之后只有01了，不需要sig

        # acc = paddle.metric.accuracy(predicts, y_data)
        pred = predicts  # [0]

        TP = np.sum(np.array(pred * label))

        FP = np.sum(np.array(pred * (1 - label)))

        FN = np.sum(np.array((1 - pred) * label))

        # TN = np.sum(np.array((1 - pred) * (1 - label)))
        self.f1[0] += TP
        self.f1[1] += FP
        self.f1[2] += FN

        self.auc_pred.append(predicts.flatten())
        self.auc_label.append(label.flatten())
        try:
            auc = roc_auc_score(predicts.flatten(), label.flatten().astype("int64"))
        except Exception as e:
            auc = 0
        self.auc += auc
        self.counter += 1

    def caculate(self):
        try:
            if self.f1[0] + self.f1[1] != 0.0:
                precision = self.f1[0] / (self.f1[0] + self.f1[1]) + 1e-7
            else:
                precision = 0.0001
            if self.f1[0] + self.f1[2] != 0.0:
                recall = self.f1[0] / (self.f1[0] + self.f1[2]) + 1e-7
            else:
                recall = 0.0001
            f1 = 2 * precision * recall / (precision + recall)
        except Exception as e:
            f1 = 0
        try:
            label_vec = np.array(self.auc_label).flatten()
            pred_vec = np.array(self.auc_pred).flatten()
            auc = roc_auc_score(label_vec,pred_vec)
            # auc = self.auc / self.counter
            # print(self.auc / self.counter)
        except Exception as e:
            auc = 0
        return f1, auc

    def clear(self):
        self.f1 = [0, 0, 0]
        self.auc_pred = []
        self.auc_label = []