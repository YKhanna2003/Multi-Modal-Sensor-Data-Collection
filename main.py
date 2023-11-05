import os
import zipfile

class MLModel:
    def __init__(self,):
       pass
    
    def setup(self,target_directory):
        os.system('kaggle datasets download -d wanghaohan/confused-eeg')
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        zip_file_path = 'confused-eeg.zip'
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(target_directory)
        print(f"Successfully extracted '{zip_file_path}' to '{target_directory}'.")

if __name__ == "__main__":
    m = MLModel()
    m.setup('data')
