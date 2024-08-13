import os

from google import oauth2 as goauth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow


class GoogleDriveTokenManager:
    __scope__ = ['https://www.googleapis.com/auth/drive']
    __credentials_file_name__ = 'google_credentials.json'

    def load_credentials(self) -> Credentials:
        token_file_name = 'token.json'
        recovered_credentials = None

        if os.path.exists(token_file_name):
            recovered_credentials = Credentials.from_authorized_user_file(token_file_name)

        if not recovered_credentials:
            return self.__authenticate_with_file__(token_file_name)

        if recovered_credentials.valid and not recovered_credentials.expired:
            return recovered_credentials

        if recovered_credentials.expired and recovered_credentials.refresh_token:
            recovered_credentials.refresh(Request())
            return self.__save_credentials__(token_file_name, recovered_credentials)

        return self.__authenticate_with_file__(token_file_name)

    def __authenticate_with_file__(self, token_file_name: str) -> Credentials:
        flow = InstalledAppFlow.from_client_secrets_file(
            self.__credentials_file_name__, self.__scope__
        )
        credentials: goauth.credentials.Credentials = flow.run_local_server(port=0)
        return self.__save_credentials__(token_file_name, credentials)

    @staticmethod
    def __save_credentials__(filename: str, credentials: Credentials) -> Credentials:
        with open(filename, 'w') as token_file:
            token_file.write(credentials.to_json())

        return credentials
