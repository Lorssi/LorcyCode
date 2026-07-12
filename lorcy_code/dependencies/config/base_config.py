from lorcy_code.dependencies.util import os_util

base_dir = "./lorcy_code"

# ==================== 特征 ====================
past_7d_abortion_features = [f'abortion_rate_past_{day + 1}d' for day in range(7)]
abortion_first_diff = [f'abortion_rate_diff_{day}d' for day in range(1, 7)]
abortion_second_diff = [f'abortion_rate_diff2_{day}d' for day in range(1, 6)]
# TRANSFORM_FIT
DISCRETE_COLUMNS = ['org_inv_dk','city','l3_org_inv_dk']
CONTINUOUS_COLUMNS = ['check_out_ratio_7d', 'reserve_sow_sqty', 'abortion_rate_ma_diff'] + past_7d_abortion_features
INVARIANT_COLUMNS = ['season', 'month']