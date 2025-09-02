from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

dataset = "sulianova/cardiovascular-disease-dataset"
api.dataset_download_files(dataset, path="datasets/", unzip=True)

print("Download concluído e arquivos descompactados em ./datasets/")
