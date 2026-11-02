import substrate as S
from ._user_study import make_plugin
PLUGIN = make_plugin('MeissaEval')
S.register_plugin(PLUGIN)
