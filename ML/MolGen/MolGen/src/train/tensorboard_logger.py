from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self, log_dir):
        """TensorBoardLoggerクラスのコンストラクタ。

        Args:
            log_dir (str): TensorBoardログを保存するディレクトリ。
        """
        self.writer = SummaryWriter(log_dir)

    def log(self, label, value, step):
        """指定されたラベルで値をTensorBoardに記録するメソッド。

        Args:
            label (str): 記録する値のラベル（例: 'loss', 'accuracy'）。
            value (float): 記録する値。
            step (int): トレーニングのステップ数。
        """
        self.writer.add_scalar(label, value, step)

    def close(self):
        """TensorBoardのリソースを解放するメソッド。"""
        self.writer.close()