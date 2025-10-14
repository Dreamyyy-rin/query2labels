from .resnet import *
from .query2label import Qeruy2Label
query2label = Qeruy2Label
from .tresnet import tresnetm, tresnetl, tresnetxl, tresnetl_21k
from .tresnet2 import tresnetl as tresnetl_v2
from .swin_transformer import build_swin_transformer
from .query2label import Qeruy2Label
from .tresnet import tresnetm, tresnetl, tresnetxl, tresnetl_21k
from .backbone import build_backbone
from .transformer import build_transformer

def build_model(args):
    # buat instance model utama di sini
    model = Qeruy2Label(args)
    return model
