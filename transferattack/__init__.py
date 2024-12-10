from .ours.mumodig import MUMODIG
from .ours.mumodig_sgm import MUMODIG_SGM 
from .ours.mumodig_pnapo import MUMODIG_PNAPO
from .gradient.mifgsm import MIFGSM

attack_zoo = {

            'mumodig':MUMODIG,
            'mumodig_sgm':MUMODIG_SGM,
            'mumodig_pnapo':MUMODIG_PNAPO,
            'mifgsm': MIFGSM
        }

__version__ = '1.0.0'
