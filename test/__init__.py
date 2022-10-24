import os

TEST_CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./qcs_config")
TEST_QCS_SETTINGS_PATH = os.path.join(TEST_CONFIG_DIR, "settings.toml")
TEST_QCS_SECRETS_PATH = os.path.join(TEST_CONFIG_DIR, "secrets.toml")


def override_qcs_config():
    os.environ["QCS_SETTINGS_FILE_PATH"] = TEST_QCS_SETTINGS_PATH
    os.environ["QCS_SECRETS_FILE_PATH"] = TEST_QCS_SECRETS_PATH
