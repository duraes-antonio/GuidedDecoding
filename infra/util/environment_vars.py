import ast
import os

import dotenv
from typing_extensions import TypedDict

from infra.consts.environment import Environment


class EnvironmentVars(TypedDict):
    google_credentials: dict
    google_token: dict


def get_env_vars(env: Environment) -> EnvironmentVars:
    if env is Environment.KAGGLE:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        return {
            'google_credentials': ast.literal_eval(user_secrets.get_secret("GOOGLE_CREDENTIALS_JSON")),
            'google_token': ast.literal_eval(user_secrets.get_secret("GOOGLE_TOKEN_JSON")),
        }

    colab_env_path = '../drive/MyDrive/unet_depth/auth/.env'
    local_env_path = './.env'
    dotenv.load_dotenv(colab_env_path if env is Environment.COLAB else local_env_path)
    return {
        'google_credentials': ast.literal_eval(os.environ["GOOGLE_CREDENTIALS_JSON"]),
        'google_token': ast.literal_eval(os.environ["GOOGLE_TOKEN_JSON"]),
    }
