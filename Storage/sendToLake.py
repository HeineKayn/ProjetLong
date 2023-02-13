from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import ClientSecretCredential
# install the following package
# pip install azure-storage-file-datalake

# Get the below details from your storage account


def clientConnect():
    storage_account_name = "malwarelake"
    storage_account_key = "ifCJVhgEG/4YVJm7/eYh2VEn4Sug9Ilj2txE1XQGy+W49vqlPBPujVzCvONnGGG3m4OOMY/E3Ty1+AStn16TjQ=="
    container_name = "test"
    directory_name = "test1"

    service_client = DataLakeServiceClient(account_url="{}://{}.dfs.core.windows.net".format(
        "https", storage_account_name), credential=storage_account_key)

    file_system_client = service_client.get_file_system_client(
        file_system=container_name)
    dir_client = file_system_client.get_directory_client(directory_name)
    dir_client.create_directory()

    data = """This is a test file"""
    file_client = dir_client.create_file(
        "YYESCAMARCHE.TXT")
    file_client.append_data(data=data, offset=0,
                            length=len(data))
    file_client.flush_data(len(data))
    print("File uploaded successfully")
clientConnect()
