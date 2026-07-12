from typing import Dict, Any, List, Tuple, Union
import pandas as pd

#基类
class BaseOutput(object):
    def __init__(self,output:pd.DataFrame,extend_data: Dict[str, Any] = None):
        self.output:pd.DataFrame =output
        self.extend_data:Dict[str, Any] = extend_data if extend_data is not None else {}

    def get_output(self) -> Dict[str, Any]:
        pass








