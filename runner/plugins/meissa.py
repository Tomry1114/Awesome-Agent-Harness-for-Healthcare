import substrate as S
from ._external_data import make_plugin
PLUGIN = make_plugin('MeissaSFT')
S.register_plugin(PLUGIN)
