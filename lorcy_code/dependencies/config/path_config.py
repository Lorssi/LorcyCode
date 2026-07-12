from lorcy_code.dependencies.util import os_util

base_dir = "./lorcy_code"
RAW_DATA_ROOT = os_util.create_dir_if_not_exist('/'.join([base_dir, 'data/raw']))
INTERIM_DATA_ROOT = os_util.create_dir_if_not_exist('/'.join([base_dir, 'data/interim']))
MODEL_DATA_ROOT = os_util.create_dir_if_not_exist('/'.join([base_dir, 'data/model']))
EXTERNAL_DIR = os_util.create_dir_if_not_exist('/'.join([base_dir, 'data/external']))
RESULT_DIR = os_util.create_dir_if_not_exist('/'.join([base_dir, 'data/result']))

PRRS_ABORTION_ABNORMAL_ALERT_ROOT = os_util.create_dir_if_not_exist('/'.join([INTERIM_DATA_ROOT, 'PRRS_Abortion_Abnormal_Alert']))
FEATURE_STORE = os_util.create_dir_if_not_exist('/'.join([INTERIM_DATA_ROOT, 'PRRS_Abortion_Abnormal_Alert', 'feature_store']))
RISK_ALERT_RUNING_DATA_STORE = os_util.create_dir_if_not_exist('/'.join([INTERIM_DATA_ROOT, 'PRRS_Abortion_Abnormal_Alert', 'risk_alert']))
RISK_ALERT_MODEL_FILE_PATH = os_util.create_dir_if_not_exist('/'.join([MODEL_DATA_ROOT, 'PRRS_Abortion_Abnormal_Alert', 'risk_alert']))
RISK_ALERT_RESULT_PATH = os_util.create_dir_if_not_exist('/'.join([RESULT_DIR, 'PRRS_Abortion_Abnormal_Alert', 'risk_alert']))

# ==================== 中间表 ====================
RISK_ALERT_TRAIN_INDEX_SAMPLE_PATH = '/'.join([RISK_ALERT_RUNING_DATA_STORE, 'risk_alert_train_index_sample_data.csv'])
RISK_ALERT_FEATURE_DATASET = '/'.join([RISK_ALERT_RUNING_DATA_STORE, 'risk_alert_train_feature_dataset.csv'])
TRAIN_TEST_DATA = '/'.join([RISK_ALERT_RUNING_DATA_STORE, 'risk_alert_train_test_data.csv'])
TRAIN_TEST_DATA_Y = '/'.join([RISK_ALERT_RUNING_DATA_STORE, 'risk_alert_train_test_data_y.csv'])
TRAIN_TEST_DATA_X = '/'.join([RISK_ALERT_RUNING_DATA_STORE, 'risk_alert_train_test_data_X.csv'])
TRAIN_Y = '/'.join([RISK_ALERT_RUNING_DATA_STORE, 'risk_alert_train_y.csv'])
TRAIN_X = '/'.join([RISK_ALERT_RUNING_DATA_STORE, 'risk_alert_train_X.csv'])
TEST_Y = '/'.join([RISK_ALERT_RUNING_DATA_STORE, 'risk_alert_test_y.csv'])
TEST_X = '/'.join([RISK_ALERT_RUNING_DATA_STORE, 'risk_alert_test_X.csv'])
TRAIN_X_TRANSFORMED = '/'.join([RISK_ALERT_RUNING_DATA_STORE, 'risk_alert_transformed_train_X.csv'])
TEST_X_TRANSFORMED = '/'.join([RISK_ALERT_RUNING_DATA_STORE, 'risk_alert_transformed_test_X.csv'])
TRAIN_X_TRANSFORMED_MASK_NULL = '/'.join([RISK_ALERT_RUNING_DATA_STORE, 'risk_alert_transformed_train_X_mask_null.csv'])
TEST_X_TRANSFORMED_MASK_NULL = '/'.join([RISK_ALERT_RUNING_DATA_STORE, 'risk_alert_transformed_test_X_mask_null.csv'])

# ==================== 模型文件 ====================
TRANSFORM_PATH = '/'.join([RISK_ALERT_MODEL_FILE_PATH, 'risk_alert_nfm_transform.json'])
MODEL_PATH = '/'.join([RISK_ALERT_MODEL_FILE_PATH, 'risk_alert_nfm_model.pth'])

# ==================== 中间表 ====================
RISK_ALERT_PREDICT_INDEX_SAMPLE_PATH = '/'.join([RISK_ALERT_RUNING_DATA_STORE, 'risk_alert_predict_index_sample_data.csv'])
RISK_ALERT_PREDICT_FEATURE_DATASET = '/'.join([RISK_ALERT_RUNING_DATA_STORE, 'risk_alert_predict_feature_dataset.csv'])
RISK_ALERT_PREDICT_FEATURE_TRANSFORMED = '/'.join([RISK_ALERT_RUNING_DATA_STORE, 'risk_alert_predict_feature_transformed.csv'])
RISK_ALERT_PREDICT_FEATURE_TRANSFORMED_MASKNULL = '/'.join([RISK_ALERT_RUNING_DATA_STORE, 'risk_alert_predict_feature_transformed_masknull.csv'])

# ==================== 预测结果 ====================
RISK_ALERT_PREDICT_RESULT = '/'.join([RISK_ALERT_RESULT_PATH, 'predict_result.csv'])