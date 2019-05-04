import os
# import file
import time

class Log():
    LOG_DIRECTORY = "log/"
    def __init__(self, file_name="log_{}.txt".format(int(time.time())) ):
        if not os.path.exists(self.LOG_DIRECTORY): os.makedirs(self.LOG_DIRECTORY)
        self.log_file = open(self.LOG_DIRECTORY+file_name, "w")

    def write(self, msg):
        print(msg)
        self.log_file.write(str(msg))
        self.log_file.write("\n")

    def close(self):
        self.log_file.close()

if __name__ == "__main__":
    pass
