import substrate as S
from ._external_data import make_plugin
PLUGIN = make_plugin('MeissaEval')
S.register_plugin(PLUGIN)
