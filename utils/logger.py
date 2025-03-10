import datetime


class Logger:
    def __init__(self, log_file=None):
        self.log_file = log_file  # 如果提供文件名，日志会写入文件

    def _get_timestamp(self):
        """获取当前的时间戳，格式为 yyyy-mm-dd hh:mm:ss"""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log(self, message):
        """打印日志并写入文件（如果提供了 log_file）"""
        timestamp = self._get_timestamp()
        log_message = f"[{timestamp}] {message}"

        # 打印到控制台
        print(log_message)

        # 如果提供了文件名，写入文件
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_message + '\n')

    def close(self):
        """关闭日志文件"""
        if self.log_file:
            self.log("日志文件已关闭")
            self.log_file = None

