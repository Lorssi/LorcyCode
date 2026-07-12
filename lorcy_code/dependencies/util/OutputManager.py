from typing import Dict, Any
import pandas as pd
from util.base_output import BaseOutput

class Output(BaseOutput):
    def __init__(self, output: pd.DataFrame, extend_data: Dict[str, Any] = None):
        super().__init__(output, extend_data)
        self.data = None

    def get_output(self) -> Dict[str, Any]:
        self.data = {'output': self.output}
        if self.extend_data:
            self.data.update(self.extend_data)
        return self.data

# # 示例
# output_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
# extend_data = {'extra_info': pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})}
# output_instance = Output(output_df, extend_data)
# print(output_instance.get_output())