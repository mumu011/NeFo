# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from llamafactory.train.tuner import run_exp  # use absolute import

# 减少transformers的冗长日志输出，特别是权重加载信息
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.image_processing_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.image_processing_base").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

def launch():
    run_exp()

if __name__ == "__main__":
    launch()

