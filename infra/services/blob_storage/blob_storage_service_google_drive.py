import io
from pathlib import Path
from typing import Generic, TypeVar, Optional

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from domain.models.named_entity import NamedEntity
from domain.services.blob_storage_service import BlobStorageService
from infra.services.blob_storage.auth_google_drive import GoogleDriveTokenManager

T = TypeVar('T')


class GoogleDriveBlobStorageService(BlobStorageService, Generic[T]):

    def __init__(self, folder_name: str):
        self.directory = folder_name or 'unet_depth_256x256_bgr'
        self.__credentials__ = GoogleDriveTokenManager().load_credentials()
        self.__service__ = build('drive', 'v3', credentials=self.__credentials__)

    def __create_folder__(self, name: str):
        file_metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        files = self.__service__.files()
        file = files.create(body=file_metadata, fields='id').execute()
        return file['id']

    def __get_metadata_by_id__(self, file_id: str) -> Optional[NamedEntity]:
        files = self.__service__.files().get(fileId=file_id).execute()

        if not files:
            return None

        return NamedEntity(name=files['name'], id=files['id'])

    def __search_folder__(self, name: str) -> str:
        files = self.__service__.files()
        folders_dict = files.list(
            q=f"name = '{name}' and mimeType='application/vnd.google-apps.folder'",
            fields="files(id, name)"
        ).execute()
        folders = folders_dict['files']
        return folders[0]['id'] if folders else None

    def __search_file__(self, name: str) -> Optional[str]:
        files = self.__service__.files()
        folders_dict = files.list(
            q=f"name = '{name}' and not mimeType='application/vnd.google-apps.folder'",
            fields="files(id, name)"
        ).execute()
        folders = folders_dict['files']
        return folders[0]['id'] if folders else None

    def __upload_file__(self, file_path: str, new_filename: Optional[str], folder_id: str) -> str:
        file_metadata = {
            'name': new_filename or Path(file_path).name,
            'parents': [folder_id]
        }
        media = MediaFileUpload(file_path, resumable=True)
        model_id = self.__search_file__(file_metadata['name'])

        if model_id:
            file = self.__service__.files().update(
                fileId=model_id, body={'name': file_metadata['name']},
                media_body=media, fields='id'
            ).execute()

        else:
            file = self.__service__.files().create(
                body=file_metadata, media_body=media, fields='id'
            ).execute()

        return file['id']

    def __download_file__(self, new_name: str, file_id: str):
        request = self.__service__.files().get_media(fileId=file_id)
        file_io = io.FileIO(new_name, mode='wb')
        downloader = MediaIoBaseDownload(file_io, request)
        done = False

        while not done:
            _, done = downloader.next_chunk()

    def save(self, current_file_path: str, new_file_name: Optional[str] = None) -> str:
        folder_id = self.__search_folder__(self.directory)

        if not folder_id:
            folder_id = self.__create_folder__(self.directory)

        return self.__upload_file__(current_file_path, new_file_name, folder_id)

    def download_last(self) -> Optional[NamedEntity]:
        folder_id = self.__search_folder__(self.directory)

        if not folder_id:
            return None

        files_service = self.__service__.files()
        files_dict = files_service.list(
            q=f"'{folder_id}' in parents",
            orderBy="modifiedTime desc",
            fields="files(id, name)"
        ).execute()

        files = files_dict['files']

        if not files:
            return None

        self.__download_file__(files[0]['name'], files[0]['id'])

    def get_by_name(self, file_name: str) -> Optional[NamedEntity]:
        file_id = self.__search_file__(file_name)

        if not file_id:
            return None

        return NamedEntity(id=file_id, name=file_name)

    def download(self, file_id: str) -> Optional[NamedEntity]:
        file_metadata: NamedEntity = self.__get_metadata_by_id__(file_id)
        self.__download_file__(file_metadata['name'], file_metadata['id'])
        return file_metadata

    def remove(self, file_id: str) -> None:
        self.__service__.files().delete(fileId=file_id).execute()
