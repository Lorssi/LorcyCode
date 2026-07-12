import datetime
import time
import pytz

version = 1.3


def time_stamp_long2short(time_stamp_long):
    """
    13位时间搓转为10+3位(浮点)形式
    """
    return int(time_stamp_long) * 1.0 / 1000


def time_stamp_sort2long(time_stamp_short):
    """
    10+3位(浮点)时间搓转为13位形式
    """
    return int(time_stamp_short) * 1000


def time2time_stamp_long(year, month, day, hour=0, minute=0, sec=0):
    """
    输入时间，返回mongo支持的NumberLong数据格式(13位整数)
    """
    time_stamp_long = time2time_stamp(year, month, day, hour=0, minute=0, sec=0) * 1000
    return time_stamp_long


def time2time_stamp(year, month, day, hour=0, minute=0, sec=0):
    """
    转换为10+3位(浮点)时间戳形式
    """
    year = str(year)
    month = str(month)
    day = str(day)
    hour = str(hour)
    minute = str(minute)
    sec = str(sec)

    time_str = year + month + day + hour + minute + sec
    # print(f"time_str = {time_str}") # time_str = 2024522000
    time_tuple = time.strptime(time_str, "%Y%m%d%H%M%S")
    # print(f"time_tuple = {time_tuple}") # time_tuple = time.struct_time(tm_year=2024, tm_mon=5, tm_mday=22, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=2, tm_yday=143, tm_isdst=-1)
    time_stamp = int(time.mktime(time_tuple))

    return time_stamp  # 1716307200


def timestamp2datetime(timestamp, convert_to_local=False):
    ''' Converts UNIX timestamp(10位) to a datetime object. '''
    if isinstance(timestamp, (int ,float)):
        dt = datetime.datetime.utcfromtimestamp(timestamp)
        if convert_to_local:  # 是否转化为本地时间
            dt = dt + datetime.timedelta(hours=8)  # 中国默认时区
        return dt
    return timestamp


def time_str2long_stamp(time_str, time_str_format="%Y-%m-%d"):
    """
    """
    stamp = int(time.mktime(time.strptime(time_str, time_str_format)))
    long_stamp = time_stamp_sort2long(stamp)  # 目标日期的起始时刻

    return long_stamp


def func_time_cost(fn):
    def inner(*arg, **kwarg):
        s_time = time.time()
        res = fn(*arg, **kwarg)
        e_time = time.time()
        # print('{} 耗时：{}秒'.format(fn.__name__, e_time - s_time))
        return res

    return inner
def time_str2datetime(time_str):
    stamp = time_str2long_stamp(time_str, )
    return timestamp2datetime(time_stamp_long2short(stamp),True)


def time_str2datetime_pig(time_str):
    return datetime.datetime.strptime(time_str, "%Y-%m-%d")


if __name__ == "__main__":
    """
    测试
    """
    # pass
    stamp = time_str2long_stamp("20240522:00-00-00")
    print(stamp)
    dt = timestamp2datetime(time_stamp_long2short(stamp),True)
    print(dt)


